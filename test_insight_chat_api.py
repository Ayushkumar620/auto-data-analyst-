from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_insight_endpoint_returns_structured_insights():
    csv_content = "date,sales,profit\n2024-01,100,10\n2024-02,120,12\n2024-03,150,15\n2024-04,190,20\n"
    response = client.post(
        "/api/v1/insights/generate",
        files={"file": ("metrics.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "facts" in payload
    assert "insights" in payload
    assert any(item["type"] == "trend" for item in payload["insights"])


def test_chat_endpoint_responds_using_dataset_context():
    csv_content = "month,region,sales\n2024-01,North,120\n2024-01,South,90\n2024-02,North,150\n2024-02,South,110\n"
    response = client.post(
        "/api/v1/chat",
        files={"file": ("sales.csv", csv_content, "text/csv")},
        data={"message": "what is the total sales?"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["intent"] in {"aggregation", "statistics"}
    assert "message" in payload


def test_insight_endpoint_accepts_workspace_project_context():
    workspace = client.post(
        "/api/v1/workspaces",
        json={"name": "Insight Workspace", "owner": "demo"},
    ).json()
    project = client.post(
        f"/api/v1/workspaces/{workspace['id']}/projects",
        json={"name": "Trend Project"},
    ).json()

    csv_content = "date,sales\n2024-01,100\n2024-02,130\n2024-03,160\n"
    response = client.post(
        "/api/v1/insights/generate",
        files={"file": ("metrics.csv", csv_content, "text/csv")},
        data={"workspace_id": workspace["id"], "project_id": project["id"]},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_id"] == workspace["id"]
    assert payload["project_id"] == project["id"]
    assert payload["workspace_dataset_id"]
