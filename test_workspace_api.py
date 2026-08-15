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
