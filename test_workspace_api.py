from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_workspace_creation_api():
    response = client.post(
        "/api/v1/workspaces",
        json={"name": "Sales Workspace", "owner": "demo"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["name"] == "Sales Workspace"
    assert payload["owner"] == "demo"


def test_project_creation_api():
    created = client.post(
        "/api/v1/workspaces",
        json={"name": "Ops Workspace", "owner": "demo"},
    ).json()

    response = client.post(
        f"/api/v1/workspaces/{created['id']}/projects",
        json={"name": "Revenue Review"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_id"] == created["id"]
    assert payload["name"] == "Revenue Review"


def test_upload_with_workspace_project_context_links_dataset():
    workspace = client.post(
        "/api/v1/workspaces",
        json={"name": "Finance Workspace", "owner": "demo"},
    ).json()
    project = client.post(
        f"/api/v1/workspaces/{workspace['id']}/projects",
        json={"name": "Budget Review"},
    ).json()

    csv_content = "region,sales\nNorth,100\nSouth,120\n"
    response = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("sales.csv", csv_content, "text/csv")},
        data={"workspace_id": workspace["id"], "project_id": project["id"]},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_id"] == workspace["id"]
    assert payload["project_id"] == project["id"]
    assert payload["workspace_dataset_id"]
