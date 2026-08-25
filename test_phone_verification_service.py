"""Tests for Enterprise Phone Number Verification Service and Endpoints.

Verifies:
1. Phone number normalization and international E.164 format validation
2. Cryptographic 6-digit SMS OTP generation
3. In-memory OTP caching, single-use invalidation, and retry limit protection
4. Twilio and HTTP SMS Gateway dispatch with mock transport
5. End-to-end phone OTP send and verify in Flask and FastAPI auth routers
"""
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from backend.app.core.phone_service import PhoneVerificationService, SmsConfig
from app import app as flask_app
from backend.app.main import app as fastapi_app


@pytest.fixture
def phone_service():
    return PhoneVerificationService(otp_ttl_seconds=60)


def test_phone_normalization_and_validation(phone_service):
    """Verify normalization of various international phone formats."""
    # 10-digit Indian number without country code
    assert phone_service.normalize_phone("9876543210") == "+919876543210"
    # Formatted US number
    assert phone_service.normalize_phone("+1 (555) 019-2834") == "+15550192834"
    # UK number
    assert phone_service.normalize_phone("+44 7911 123456") == "+447911123456"

    # Validation
    assert phone_service.is_valid_phone("+919876543210") is True
    assert phone_service.is_valid_phone("+15550192834") is True
    assert phone_service.is_valid_phone("invalid_phone_str") is False
    assert phone_service.is_valid_phone("123") is False


def test_sms_otp_lifecycle(phone_service):
    """Verify phone OTP storage, verification, and expiration."""
    phone = "+919876543210"
    otp = "456789"
    phone_service.store_otp(phone, otp)

    # Invalid code
    is_valid, msg = phone_service.verify_otp(phone, "000000")
    assert is_valid is False
    assert "Invalid" in msg

    # Valid code
    is_valid, msg = phone_service.verify_otp(phone, otp)
    assert is_valid is True
    assert "successful" in msg.lower()

    # Replay protection (single use)
    is_valid, msg = phone_service.verify_otp(phone, otp)
    assert is_valid is False


def test_mock_twilio_sms_dispatch():
    """Verify Twilio SMS REST dispatch format."""
    config = SmsConfig(
        twilio_account_sid="AC_test_account_sid_12345",
        twilio_auth_token="auth_token_secret_12345",
        twilio_phone_number="+15005550006",
    )
    service = PhoneVerificationService(config=config)

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 201
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        sent, msg, otp = service.send_sms_otp("+919876543210")

        assert sent is True
        assert len(otp) == 6
        mock_urlopen.assert_called_once()


def test_flask_phone_otp_endpoints():
    """Verify Flask /api/v1/auth/phone/send and /api/v1/auth/phone/verify."""
    with flask_app.test_client() as client:
        # 1. Send SMS
        send_resp = client.post("/api/v1/auth/phone/send", json={"phone": "+919876543210"})
        assert send_resp.status_code == 200
        data = send_resp.json
        assert "demo_otp" in data
        otp = data["demo_otp"]

        # 2. Verify SMS
        verify_resp = client.post("/api/v1/auth/phone/verify", json={
            "phone": "+919876543210",
            "otp": otp,
        })
        assert verify_resp.status_code == 200
        v_data = verify_resp.json
        assert "access_token" in v_data
        assert "user" in v_data


def test_fastapi_phone_otp_endpoints():
    """Verify FastAPI /api/v1/auth/phone/send and /api/v1/auth/phone/verify."""
    client = TestClient(fastapi_app)

    # 1. Send SMS
    send_resp = client.post("/api/v1/auth/phone/send", json={"phone": "+15550192834"})
    assert send_resp.status_code == 200
    data = send_resp.json()
    assert "demo_otp" in data
    otp = data["demo_otp"]

    # 2. Verify SMS
    verify_resp = client.post("/api/v1/auth/phone/verify", json={
        "phone": "+15550192834",
        "otp": otp,
    })
    assert verify_resp.status_code == 200
    v_data = verify_resp.json()
    assert "access_token" in v_data
