"""Test Login & Auth Flow on Flask server (http://127.0.0.1:5000)
including Child Holding Magic Lamp UI, Email OTP, Welcome Banner, and Recent Workflows.
"""
import json
from app import app


def test_flask_index_contains_lamp_scene_and_workflows():
    """Verify http://127.0.0.1:5000/ contains the Lamp scene, OTP tabs, and Recent Workflows."""
    client = app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # 1. Child holding lamp scene
    assert 'id="auth-overlay"' in html
    assert 'child-lamp-scene' in html
    assert 'toggleMagicLamp()' in html

    # 2. Email OTP and Password tabs
    assert 'id="tab-btn-otp"' in html
    assert 'id="otp-form"' in html
    assert 'id="password-form"' in html

    # 3. Personalized Welcome & Recent Workflows
    assert 'welcome-user-name' in html
    assert 'recent-workflows-section' in html
    assert 'Recent Workflows &amp; Quick Actions' in html


def test_flask_auth_login_endpoint():
    """Verify /api/v1/auth/login works on Flask."""
    client = app.test_client()
    res = client.post(
        "/api/v1/auth/login",
        data=json.dumps({"email": "demo@example.com", "password": "strongpass123"}),
        content_type="application/json",
    )
    assert res.status_code == 200
    data = json.loads(res.get_data(as_text=True))
    assert "access_token" in data
    assert data["user"]["email"] == "demo@example.com"


def test_flask_email_otp_flow():
    """Verify /api/v1/auth/otp/send and verify works on Flask."""
    client = app.test_client()

    # Step 1: Send OTP
    send_res = client.post(
        "/api/v1/auth/otp/send",
        data=json.dumps({"email": "analyst@company.com"}),
        content_type="application/json",
    )
    assert send_res.status_code == 200
    send_data = json.loads(send_res.get_data(as_text=True))
    otp_code = send_data.get("demo_otp")
    assert otp_code is not None

    # Step 2: Verify OTP
    verify_res = client.post(
        "/api/v1/auth/otp/verify",
        data=json.dumps({"email": "analyst@company.com", "otp": otp_code}),
        content_type="application/json",
    )
    assert verify_res.status_code == 200
    verify_data = json.loads(verify_res.get_data(as_text=True))
    assert "access_token" in verify_data
    assert verify_data["user"]["email"] == "analyst@company.com"
