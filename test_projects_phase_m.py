"""Tests for PHASE M: project management (CRUD) with authentication."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database import Base, get_db
from backend.app.main import app


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _token(client):
    client.post("/api/v1/auth/register", json={
        "email": "user@example.com", "username": "user", "password": "strongpass123",
    })
    return client.post("/api/v1/auth/login", json={
        "email": "user@example.com", "password": "strongpass123",
    }).json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_project_lifecycle(client):
    token = _token(client)

    # Create
    created = client.post("/api/v1/projects", json={"name": "Q4 Analysis", "description": "Quarterly"},
                          headers=_auth(token))
    assert created.status_code == 201
    project_id = created.json()["id"]
    assert created.json()["name"] == "Q4 Analysis"

    # List
    listed = client.get("/api/v1/projects", headers=_auth(token))
    assert listed.status_code == 200
    names = [p["name"] for p in listed.json()["projects"]]
    assert "Q4 Analysis" in names

    # Get
    fetched = client.get(f"/api/v1/projects/{project_id}", headers=_auth(token))
    assert fetched.status_code == 200
    assert fetched.json()["description"] == "Quarterly"

    # Update
    updated = client.patch(f"/api/v1/projects/{project_id}", json={"name": "Q4 Final"},
                           headers=_auth(token))
    assert updated.status_code == 200
    assert updated.json()["name"] == "Q4 Final"

    # Delete
    deleted = client.delete(f"/api/v1/projects/{project_id}", headers=_auth(token))
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/projects/{project_id}", headers=_auth(token)).status_code == 404


def test_project_requires_authentication(client):
    assert client.post("/api/v1/projects", json={"name": "No Auth"}).status_code == 401
    assert client.get("/api/v1/projects").status_code == 401


def test_users_cannot_access_each_others_projects(client):
    token_a = _token(client)
    token_b = _token_b(client)
    created = client.post("/api/v1/projects", json={"name": "Private Project"}, headers=_auth(token_a))
    project_id = created.json()["id"]
    # User B must not see or fetch User A's project
    assert client.get(f"/api/v1/projects/{project_id}", headers=_auth(token_b)).status_code == 404


def _token_b(client):
    client.post("/api/v1/auth/register", json={
        "email": "other@example.com", "username": "other", "password": "strongpass123",
    })
    return client.post("/api/v1/auth/login", json={
        "email": "other@example.com", "password": "strongpass123",
    }).json()["access_token"]