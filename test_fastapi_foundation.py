from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "auto-data-analyst"


def test_upload_csv_dataset(tmp_path):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "region,sales,month\nNorth,120,2024-01\nSouth,90,2024-02\nEast,130,2024-03\n",
        encoding="utf-8",
    )

    with csv_path.open("rb") as handle:
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("sales.csv", handle, "text/csv")},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "uploaded"
    assert payload["rows"] == 3
    assert payload["columns"] == 3
    assert payload["dataset"]["name"] == "sales.csv"
