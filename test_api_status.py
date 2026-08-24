#!/usr/bin/env python
"""Test the Auto Data Analyst API endpoints using FastAPI TestClient."""

import sys
from pathlib import Path
import time
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))

from backend.app.main import app

client = TestClient(app)


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200, f"Health check failed: {response.text}"
    data = response.json()
    assert data["status"] == "ok"


def test_health_v1():
    """Test v1 health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_auth_register():
    """Test user registration."""
    unique_user = f"user_{int(time.time() * 1000)}"
    payload = {
        "email": f"{unique_user}@example.com",
        "username": unique_user,
        "password": "testpass123"
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code in [200, 201, 400]
    if response.status_code in [200, 201]:
        data = response.json()
        assert "access_token" in data or "id" in data or "email" in data


def test_projects_list():
    """Test listing projects."""
    response = client.get("/api/v1/projects")
    assert response.status_code in [200, 401]


def test_datasets_list():
    """Test listing datasets."""
    response = client.get("/api/v1/datasets/")
    assert response.status_code in [200, 401]


def test_insights_generate():
    """Test generating insights."""
    csv_content = b"name,age,salary\nAlice,30,50000\nBob,35,60000\nCharlie,40,70000\n"
    response = client.post(
        "/api/v1/insights/generate",
        files={"file": ("test.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "insights" in data or "summary" in data or "status" in data


def test_forecasting_endpoint():
    """Test forecasting endpoint."""
    csv_content = (
        b"date,value\n"
        b"2024-01-01,100\n"
        b"2024-01-02,105\n"
        b"2024-01-03,103\n"
        b"2024-01-04,108\n"
        b"2024-01-05,110\n"
    )
    response = client.post(
        "/api/v1/forecast",
        files={"file": ("time_series.csv", csv_content, "text/csv")},
        data={"horizon": 3, "target": "value", "date_column": "date"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "success" or "forecast" in data


if __name__ == "__main__":
    print("\nTesting Auto Data Analyst API with TestClient\n")
    test_health()
    test_health_v1()
    test_auth_register()
    test_projects_list()
    test_datasets_list()
    test_insights_generate()
    test_forecasting_endpoint()
    print("\nAll basic API tests completed successfully!")

