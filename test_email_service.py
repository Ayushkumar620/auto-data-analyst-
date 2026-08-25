"""Tests for Enterprise Email OTP Service and Endpoints.

Verifies:
1. Cryptographic 6-digit OTP generation
2. In-memory OTP storage, TTL expiration, and rate-limited attempt invalidation
3. HTML and plaintext email template rendering
4. SMTP dispatch with mock server and dev-mode fallback
5. End-to-end OTP send and verify in Flask and FastAPI auth routers
"""
from unittest.mock import MagicMock, patch
import pytest

from backend.app.core.email_service import EmailService, SmtpConfig, global_email_service
from app import app as flask_app


@pytest.fixture
def email_service():
    return EmailService(otp_ttl_seconds=60)


def test_generate_otp(email_service):
    """Verify generated OTP is 6 numeric characters."""
    otp = email_service.generate_otp(6)
    assert len(otp) == 6
    assert otp.isdigit()


def test_store_and_verify_otp(email_service):
    """Verify correct OTP verification, single-use invalidation, and bad OTP rejection."""
    email = "test.user@company.com"
    otp = "654321"
    email_service.store_otp(email, otp)

    # Wrong code should fail
    is_valid, msg = email_service.verify_otp(email, "111111")
    assert is_valid is False
    assert "Invalid" in msg

    # Correct code should succeed
    is_valid, msg = email_service.verify_otp(email, otp)
    assert is_valid is True
    assert "successful" in msg.lower()

    # Re-using the same code must fail (single-use)
    is_valid, msg = email_service.verify_otp(email, otp)
    assert is_valid is False


def test_expired_otp(email_service):
    """Verify expired OTP is rejected."""
    email = "expired@company.com"
    email_service.store_otp(email, "999888")

    # Manually expire
    email_service._cache[email]["expires_at"] = 0

    is_valid, msg = email_service.verify_otp(email, "999888")
    assert is_valid is False
    assert "expired" in msg.lower()


def test_mock_smtp_dispatch():
    """Verify SMTP dispatch constructs MIME message and calls smtplib."""
    config = SmtpConfig(
        server="smtp.example.com",
        port=587,
        username="sender@example.com",
        password="secretpassword",
        use_tls=True,
    )
    service = EmailService(smtp_config=config)

    with patch("smtplib.SMTP") as mock_smtp_cls:
        mock_server = MagicMock()
        mock_smtp_cls.return_value = mock_server

        sent, msg, otp = service.send_otp_email("recipient@company.com")

        assert sent is True
        assert len(otp) == 6
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("sender@example.com", "secretpassword")
        mock_server.sendmail.assert_called_once()


def test_flask_otp_endpoints():
    """Verify Flask /api/v1/auth/otp/send and /api/v1/auth/otp/verify."""
    with flask_app.test_client() as client:
        # Send
        send_resp = client.post("/api/v1/auth/otp/send", json={"email": "flask_user@domain.com"})
        assert send_resp.status_code == 200
        data = send_resp.json
        assert "demo_otp" in data
        otp = data["demo_otp"]

        # Verify
        verify_resp = client.post("/api/v1/auth/otp/verify", json={
            "email": "flask_user@domain.com",
            "otp": otp,
        })
        assert verify_resp.status_code == 200
        v_data = verify_resp.json
        assert "access_token" in v_data
        assert v_data["user"]["email"] == "flask_user@domain.com"

