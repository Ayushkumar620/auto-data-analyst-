from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_forecast_endpoint_returns_predictions():
    csv_content = "date,sales\n2024-01,100\n2024-02,120\n2024-03,150\n2024-04,180\n2024-05,210\n"
    response = client.post(
        "/api/v1/forecast",
        files={"file": ("sales.csv", csv_content, "text/csv")},
        data={"horizon": 2},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["target"] == "sales"
    assert len(payload["forecast"]) == 2


def test_report_endpoint_generates_download_bundle():
    csv_content = "date,region,sales\n2024-01,North,100\n2024-02,North,120\n2024-03,South,90\n2024-04,South,110\n"
    response = client.post(
        "/api/v1/reports/generate",
        files={"file": ("sales.csv", csv_content, "text/csv")},
        data={"output_format": "pdf"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["report"]["title"]
    assert payload["download_url"].startswith("/api/v1/reports/")
