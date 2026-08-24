"""Test Unified Single-Localhost Deployment.

Verifies that the entire application (React SPA + FastAPI backend) is served
from a single unified localhost port (e.g. http://localhost:8000), opening
with the Login page flow and routing smoothly to the Command Studio & Dashboard.
"""
from fastapi.testclient import TestClient
from backend.app.main import app


def test_unified_root_serves_spa_html():
    """GET / serves the single-page application index.html."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "<div id=\"root\">" in response.text or "<!doctype html>" in response.text.lower()


def test_unified_routes_serve_spa_html():
    """GET /login, /dashboard, /chat serve index.html for client-side routing."""
    client = TestClient(app)
    for route in ["/login", "/dashboard", "/chat", "/upload", "/projects"]:
        res = client.get(route)
        assert res.status_code == 200
        assert "text/html" in res.headers.get("content-type", "")


def test_api_routes_not_shadowed_by_spa():
    """Verify backend API routes like /health and /api/v1/health work without being swallowed."""
    client = TestClient(app)
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"

    res_api = client.get("/api/v1/health")
    assert res_api.status_code == 200
    assert res_api.json()["status"] == "ok"


def test_localhost_login_flow_on_single_origin():
    """Verify login authentication works on the unified origin."""
    client = TestClient(app)
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "demo@example.com", "password": "strongpass123"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["user"]["email"] == "demo@example.com"

