from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_eda_endpoint_returns_structured_results():
    csv_content = "month,sales,profit\n2024-01,120,20\n2024-02,150,30\n2024-03,170,40\n2024-04,200,45\n"
    response = client.post(
        "/api/v1/datasets/eda",
        files={"file": ("sales.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["summary"]["row_count"] == 4
    assert payload["statistics"]["numeric"]
    assert payload["recommended_charts"]


def test_chart_endpoint_generates_plotly_json():
    csv_content = "region,sales\nNorth,120\nSouth,90\nEast,130\n"
    response = client.post(
        "/api/v1/datasets/chart",
        files={"file": ("sales.csv", csv_content, "text/csv")},
        data={"chart_type": "bar", "x": "region", "y": "sales", "title": "Sales by Region"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["chart_type"] == "bar"
    assert payload["data"]
    assert payload["layout"]["title"] == "Sales by Region"
