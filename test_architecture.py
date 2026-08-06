import io
import app as app_module


def test_workspace_and_project_flow():
    client = app_module.app.test_client()

    workspace_response = client.post(
        "/api/workspaces",
        json={"name": "Sales Analytics", "owner": "demo"},
    )
    assert workspace_response.status_code == 200
    workspace = workspace_response.get_json()
    assert workspace["name"] == "Sales Analytics"

    project_response = client.post(
        f"/api/workspaces/{workspace['id']}/projects",
        json={"name": "Revenue Review"},
    )
    assert project_response.status_code == 200
    project = project_response.get_json()
    assert project["workspace_id"] == workspace["id"]
    assert project["name"] == "Revenue Review"


def test_upload_creates_dataset_object():
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
    assert data["dataset_id"]
    assert data["project_id"]
    assert data["workspace_id"]
