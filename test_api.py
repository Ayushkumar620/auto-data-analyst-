from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ok'
    assert payload['service'] == 'auto-data-analyst'


def test_api_v1_health_endpoint():
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ok'


def test_list_datasets_endpoint():
    response = client.get('/api/v1/datasets/')
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get('datasets'), list)
