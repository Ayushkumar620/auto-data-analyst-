"""Tests for PHASE M: secure authentication (register, login, JWT, protected route)."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.auth.security import hash_password, verify_password, create_access_token, decode_access_token


@pytest.fixture()
def client():
    # In-memory SQLite for tests (no external PostgreSQL required)
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


def test_register_returns_token_and_user(client):
    resp = client.post("/api/v1/auth/register", json={
        "email": "alice@example.com",
        "username": "alice",
        "password": "strongpass123",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["token_type"] == "bearer"
    assert data["access_token"]
    assert data["user"]["email"] == "alice@example.com"
    # Must not leak the password hash
    assert "password_hash" not in data["user"]


def test_register_rejects_duplicate_email(client):
    payload = {"email": "bob@example.com", "username": "bob", "password": "strongpass123"}
    assert client.post("/api/v1/auth/register", json=payload).status_code == 201
    conflict = client.post("/api/v1/auth/register", json=payload)
    assert conflict.status_code == 409


def test_login_returns_token(client):
    client.post("/api/v1/auth/register", json={
        "email": "carol@example.com", "username": "carol", "password": "strongpass123",
    })
    resp = client.post("/api/v1/auth/login", json={
        "email": "carol@example.com", "password": "strongpass123",
    })
    assert resp.status_code == 200
    assert resp.json()["access_token"]


def test_login_rejects_wrong_password(client):
    client.post("/api/v1/auth/register", json={
        "email": "dave@example.com", "username": "dave", "password": "strongpass123",
    })
    resp = client.post("/api/v1/auth/login", json={
        "email": "dave@example.com", "password": "wrongpassword",
    })
    assert resp.status_code == 401


def test_protected_me_requires_token(client):
    assert client.get("/api/v1/auth/me").status_code == 401
    client.post("/api/v1/auth/register", json={
        "email": "erin@example.com", "username": "erin", "password": "strongpass123",
    })
    login = client.post("/api/v1/auth/login", json={
        "email": "erin@example.com", "password": "strongpass123",
    }).json()
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {login['access_token']}"})
    assert me.status_code == 200
    assert me.json()["email"] == "erin@example.com"


def test_password_hashing_and_verification():
    hashed = hash_password("secretphrase")
    assert hashed != "secretphrase"
    assert verify_password("secretphrase", hashed)
    assert not verify_password("wrong", hashed)


def test_jwt_roundtrip_and_invalid_token():
    token = create_access_token(42, {"email": "x@y.com"})
    payload = decode_access_token(token)
    assert payload["sub"] == "42"
    with pytest.raises(ValueError):
        decode_access_token("not.a.jwt")