from backend.app.main import app


def test_fastapi_app_metadata():
    assert app.title == "Auto Data Analyst Agent"
    assert app.version == "1.0.0"
    assert any(getattr(route, "path", None) == "/health" for route in app.routes)
    assert any(getattr(route, "path", None) == "/api/v1/health" for route in app.routes)
