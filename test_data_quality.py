from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_profile_endpoint_returns_dataset_metrics():
    csv_content = "region,sales,month\nNorth,120,2024-01\nSouth,90,2024-02\nEast,130,2024-03\n"
    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("sales.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["rows"] == 3
    assert payload["profile"]["quality_score"] >= 0
    assert payload["column_analysis"]


def test_clean_endpoint_handles_missing_and_duplicates():
    csv_content = "region,sales\nNorth,100\nNorth,100\nSouth,\nEast,200\n"
    response = client.post(
        "/api/v1/datasets/clean",
        files={"file": ("sales.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "cleaned"
    assert payload["quality_after"] >= payload["quality_before"]
    assert isinstance(payload["cleaning_report"], list)
