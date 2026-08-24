#!/usr/bin/env python
"""Comprehensive integration test for Auto Data Analyst using FastAPI TestClient."""

import sys
from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))

from backend.app.main import app

client = TestClient(app)


@pytest.fixture(scope="module")
def unique_credentials():
    ts = int(time.time() * 1000)
    return {
        "email": f"integ_{ts}@example.com",
        "username": f"integ_{ts}",
        "password": "TestPass123!@#"
    }


@pytest.fixture(scope="module")
def auth_token(unique_credentials):
    # Register
    reg_resp = client.post("/api/v1/auth/register", json=unique_credentials)
    if reg_resp.status_code in [200, 201]:
        data = reg_resp.json()
        if "access_token" in data:
            return data["access_token"]

    # Fallback to login
    login_resp = client.post(
        "/api/v1/auth/login",
        json={
            "email": unique_credentials["email"],
            "password": unique_credentials["password"],
        },
    )
    assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
    return login_resp.json()["access_token"]


def test_1_auth_register(unique_credentials):
    """Test user registration."""
    response = client.post("/api/v1/auth/register", json=unique_credentials)
    assert response.status_code in [200, 201, 400]
    if response.status_code in [200, 201]:
        data = response.json()
        assert "access_token" in data or "id" in data or "email" in data


def test_2_auth_login(auth_token):
    """Test user login returns token."""
    assert auth_token is not None
    assert len(auth_token) > 10


def test_3_create_project(auth_token):
    """Test creating a project."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.post(
        "/api/v1/projects",
        headers=headers,
        json={
            "name": f"Integration Test Project {int(time.time())}",
            "description": "A test project for integration testing"
        }
    )
    assert response.status_code in [200, 201]
    data = response.json()
    assert "id" in data


def test_4_upload_dataset():
    """Test dataset upload."""
    csv_content = b"name,age,salary\nJohn,30,50000\nJane,28,55000\nBob,35,60000\n"
    files = {"file": ("test_data.csv", csv_content, "text/csv")}
    response = client.post("/api/v1/datasets/upload", files=files)
    assert response.status_code in [200, 201]
    data = response.json()
    assert "dataset" in data or "name" in data or "id" in data


def test_5_eda_analysis():
    """Test EDA analysis."""
    csv_content = b"name,age,salary\nJohn,30,50000\nJane,28,55000\nBob,35,60000\n"
    files = {"file": ("test_data.csv", csv_content, "text/csv")}
    response = client.post("/api/v1/datasets/eda", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "statistics" in data or "summary" in data or "correlations" in data


def test_6_insights_generation():
    """Test insights generation."""
    csv_content = b"name,age,salary\nJohn,30,50000\nJane,28,55000\nBob,35,60000\n"
    files = {"file": ("test_data.csv", csv_content, "text/csv")}
    response = client.post("/api/v1/insights/generate", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "insights" in data or "summary" in data or "status" in data


def test_7_forecasting():
    """Test forecasting."""
    csv_content = (
        b"date,value\n"
        b"2024-01-01,100\n"
        b"2024-01-02,105\n"
        b"2024-01-03,103\n"
        b"2024-01-04,108\n"
        b"2024-01-05,110\n"
    )
    files = {"file": ("time_series.csv", csv_content, "text/csv")}
    data = {"horizon": 3, "target": "value", "date_column": "date"}
    response = client.post("/api/v1/forecast", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data.get("status") == "success" or "forecast" in res_data


def test_8_list_projects(auth_token):
    """Test listing projects."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.get("/api/v1/projects", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "projects" in data or isinstance(data, list)


if __name__ == "__main__":
    print("\nRunning complete integration tests via TestClient...")
    creds = {
        "email": f"integ_main_{int(time.time())}@example.com",
        "username": f"integ_main_{int(time.time())}",
        "password": "TestPass123!@#"
    }
    test_1_auth_register(creds)
    tok = auth_token(creds)
    test_2_auth_login(tok)
    test_3_create_project(tok)
    test_4_upload_dataset()
    test_5_eda_analysis()
    test_6_insights_generation()
    test_7_forecasting()
    test_8_list_projects(tok)
    print("Integration tests completed successfully!")

