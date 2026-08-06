import io
import app as app_module


def test_upload_returns_profile():
    client = app_module.app.test_client()
    with open("sample_data.csv", "rb") as f:
        payload = f.read()

    response = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(payload), "sample_data.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["dataset_name"] == "sample_data.csv"
    assert data["rows"] > 0
    assert data["columns"] > 0
    assert data["preview"]
