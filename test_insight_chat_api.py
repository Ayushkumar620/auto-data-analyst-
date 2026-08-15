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
