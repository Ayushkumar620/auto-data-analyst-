#!/usr/bin/env python
"""Comprehensive integration test for Auto Data Analyst."""
"""Comprehensive integration test for Auto Data Analyst using FastAPI TestClient."""

import sys
import json
from pathlib import Path
import tempfile
import time
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path('.').resolve()))
sys.path.insert(0, str(Path(__file__).parent))

import requests
from requests.auth import HTTPBasicAuth
from backend.app.main import app

BASE_URL = "http://localhost:8000"
API_V1 = f"{BASE_URL}/api/v1"
client = TestClient(app)

# Test data
TEST_USER = {
    "email": f"testuser_{int(time.time())}@example.com",
    "username": f"testuser_{int(time.time())}",
    "password": "TestPass123!@#"
}

def test_1_auth_register():
    """Test user registration."""
    print("\n1️⃣  Testing User Registration...")
    response = requests.post(
        f"{API_V1}/auth/register",
        json=TEST_USER
    )
    print(f"   Status: {response.status_code}")
    if response.status_code in [200, 201]:
        data = response.json()
        print(f"   ✅ User registered successfully")
        return data.get("access_token")
    else:
        print(f"   ❌ Registration failed: {response.text}")
        return None
@pytest.fixture(scope="module")
def unique_credentials():
    ts = int(time.time() * 1000)
    return {
        "email": f"integ_{ts}@example.com",
        "username": f"integ_{ts}",
        "password": "TestPass123!@#"
    }

def test_2_auth_login():
    """Test user login."""
    print("\n2️⃣  Testing User Login...")
    response = requests.post(
        f"{API_V1}/auth/login",

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
            "email": TEST_USER["email"],
            "password": TEST_USER["password"]
        }
            "email": unique_credentials["email"],
            "password": unique_credentials["password"],
        },
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
    assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
    return login_resp.json()["access_token"]


def test_1_auth_register(unique_credentials):
    """Test user registration."""
    response = client.post("/api/v1/auth/register", json=unique_credentials)
    assert response.status_code in [200, 201, 400]
    if response.status_code in [200, 201]:
        data = response.json()
        token = data.get("access_token")
        print(f"   ✅ Login successful, got token")
        return token
    else:
        print(f"   ❌ Login failed: {response.text}")
        return None
        assert "access_token" in data or "id" in data or "email" in data

def test_3_create_project(token):

def test_2_auth_login(auth_token):
    """Test user login returns token."""
    assert auth_token is not None
    assert len(auth_token) > 10


def test_3_create_project(auth_token):
    """Test creating a project."""
    print("\n3️⃣  Testing Project Creation...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{API_V1}/projects",
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.post(
        "/api/v1/projects",
        headers=headers,
        json={
            "name": "Test Project",
            "name": f"Integration Test Project {int(time.time())}",
            "description": "A test project for integration testing"
        }
    )
    print(f"   Status: {response.status_code}")
    if response.status_code in [200, 201]:
        data = response.json()
        project_id = data.get("id")
        print(f"   ✅ Project created: {project_id}")
        return project_id
    else:
        print(f"   ❌ Project creation failed: {response.text}")
        return None
    assert response.status_code in [200, 201]
    data = response.json()
    assert "id" in data


def test_4_upload_dataset():
    """Test dataset upload."""
    print("\n4️⃣  Testing Dataset Upload...")
    # Create a small CSV file
    csv_content = "name,age,salary\nJohn,30,50000\nJane,28,55000\nBob,35,60000\n"
    
    files = {"file": ("test_data.csv", csv_content.encode("utf-8"), "text/csv")}
    response = requests.post(
        f"{API_V1}/datasets/upload",
        files=files
    )
    print(f"   Status: {response.status_code}")
    if response.status_code in [200, 201]:
        data = response.json()
        print(f"   ✅ Dataset uploaded successfully")
        return data
    else:
        print(f"   ❌ Upload failed: {response.text}")
        return None
    csv_content = b"name,age,salary\nJohn,30,50000\nJane,28,55000\nBob,35,60000\n"
    files = {"file": ("test_data.csv", csv_content, "text/csv")}
    response = client.post("/api/v1/datasets/upload", files=files)
    assert response.status_code in [200, 201]
    data = response.json()
    assert "dataset" in data or "name" in data or "id" in data

def test_5_eda_analysis(upload_data):

def test_5_eda_analysis():
    """Test EDA analysis."""
    print("\n5️⃣  Testing EDA Analysis...")
    if not upload_data:
        print("   ⏭️  Skipped (no upload data)")
        return
    
    # Create a small CSV file
    csv_content = "name,age,salary\nJohn,30,50000\nJane,28,55000\nBob,35,60000\n"
    files = {"file": ("test_data.csv", csv_content.encode("utf-8"), "text/csv")}
    
    response = requests.post(
        f"{API_V1}/datasets/eda",
        files=files
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ EDA analysis completed")
        return data
    else:
        print(f"   ❌ EDA analysis failed: {response.text}")
        return None
    csv_content = b"name,age,salary\nJohn,30,50000\nJane,28,55000\nBob,35,60000\n"
    files = {"file": ("test_data.csv", csv_content, "text/csv")}
    response = client.post("/api/v1/datasets/eda", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "statistics" in data or "summary" in data or "correlations" in data


def test_6_insights_generation():
    """Test insights generation."""
    print("\n6️⃣  Testing Insights Generation...")
    csv_content = "name,age,salary\nJohn,30,50000\nJane,28,55000\nBob,35,60000\n"
    files = {"file": ("test_data.csv", csv_content.encode("utf-8"), "text/csv")}
    
    response = requests.post(
        f"{API_V1}/insights/generate",
        files=files
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Insights generated successfully")
        return data
    else:
        print(f"   ❌ Insights generation failed: {response.text}")
        return None
    csv_content = b"name,age,salary\nJohn,30,50000\nJane,28,55000\nBob,35,60000\n"
    files = {"file": ("test_data.csv", csv_content, "text/csv")}
    response = client.post("/api/v1/insights/generate", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "insights" in data or "summary" in data or "status" in data


def test_7_forecasting():
    """Test forecasting."""
    print("\n7️⃣  Testing Forecasting...")
    csv_content = "date,value\n2024-01-01,100\n2024-01-02,105\n2024-01-03,103\n2024-01-04,108\n2024-01-05,110\n"
    files = {
        "file": ("time_series.csv", csv_content.encode("utf-8"), "text/csv"),
    }
    data = {
        "horizon": 3,
        "target": "value",
        "date_column": "date"
    }
    
    response = requests.post(
        f"{API_V1}/forecast",
        files=files,
        data=data
    csv_content = (
        b"date,value\n"
        b"2024-01-01,100\n"
        b"2024-01-02,105\n"
        b"2024-01-03,103\n"
        b"2024-01-04,108\n"
        b"2024-01-05,110\n"
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   ✅ Forecasting completed successfully")
        return response.json()
    else:
        print(f"   ❌ Forecasting failed: {response.text}")
        return None
    files = {"file": ("time_series.csv", csv_content, "text/csv")}
    data = {"horizon": 3, "target": "value", "date_column": "date"}
    response = client.post("/api/v1/forecast", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data.get("status") == "success" or "forecast" in res_data

def test_8_list_projects(token):

def test_8_list_projects(auth_token):
    """Test listing projects."""
    print("\n8️⃣  Testing List Projects...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(
        f"{API_V1}/projects",
        headers=headers
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        count = len(data) if isinstance(data, list) else 0
        print(f"   ✅ Projects listed ({count} projects)")
        return data
    else:
        print(f"   ❌ List projects failed: {response.text}")
        return None
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.get("/api/v1/projects", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "projects" in data or isinstance(data, list)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 INTEGRATION TEST SUITE - AUTO DATA ANALYST")
    print("="*60)
    
    try:
        # Test authentication
        token = test_1_auth_register()
        if token:
            print(f"   Token: {token[:20]}...")
        
        login_token = test_2_auth_login()
        if not login_token:
            print("\n⚠️  Could not login, stopping tests")
            sys.exit(1)
        
        # Test with authenticated endpoints
        project_id = test_3_create_project(login_token)
        test_8_list_projects(login_token)
        
        # Test with non-authenticated endpoints
        upload_data = test_4_upload_dataset()
        test_5_eda_analysis(upload_data)
        test_6_insights_generation()
        test_7_forecasting()
        
        print("\n" + "="*60)
        print("✅ INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
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

