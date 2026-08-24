"""Test Login & Auth Flow on Flask server (http://127.0.0.1:5000)."""
import json
from app import app


def test_flask_index_contains_login_overlay():
    """Verify http://127.0.0.1:5000/ contains the Login modal and credentials fields."""
    client = app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert 'id="auth-overlay"' in html
    assert 'modal-email' in html
    assert 'modal-password' in html
    assert 'Auto-Fill Localhost Demo Credentials' in html


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
