#!/usr/bin/env python
"""Test the Auto Data Analyst API endpoints."""

import sys
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

import requests

BASE_URL = "http://localhost:8000"
API_V1 = f"{BASE_URL}/api/v1"

def test_health():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200, f"Health check failed: {response.text}"
    data = response.json()
    assert data["status"] == "ok"
    print("✅ Health endpoint works")

def test_auth_register():
    """Test user registration."""
    payload = {
        "email": "test@example.com",
        "username": "testuser",
        "password": "testpass123"
    }
    response = requests.post(f"{API_V1}/auth/register", json=payload)
    print(f"   Register response: {response.status_code}")
    if response.status_code in [200, 201]:
        print("✅ Auth register endpoint exists")
        return response.json()
    else:
        print(f"⚠️  Auth register returned {response.status_code}: {response.text}")
        return None

def test_projects_list():
    """Test listing projects."""
    response = requests.get(f"{API_V1}/projects")
    print(f"   Projects list response: {response.status_code}")
    if response.status_code == 200:
        print("✅ Projects list endpoint works")
    elif response.status_code == 401:
        print("⚠️  Projects endpoint requires authentication (expected)")
    else:
        print(f"⚠️  Projects returned {response.status_code}: {response.text}")

def test_datasets_list():
    """Test listing datasets."""
    response = requests.get(f"{API_V1}/datasets")
    print(f"   Datasets list response: {response.status_code}")
    if response.status_code == 200:
        print("✅ Datasets list endpoint works")
    elif response.status_code == 401:
        print("⚠️  Datasets endpoint requires authentication (expected)")
    else:
        print(f"⚠️  Datasets returned {response.status_code}: {response.text}")

def test_insights_list():
    """Test listing insights."""
    response = requests.get(f"{API_V1}/insights")
    print(f"   Insights list response: {response.status_code}")
    if response.status_code == 200:
        print("✅ Insights endpoint works")
    elif response.status_code == 401:
        print("⚠️  Insights endpoint requires authentication (expected)")
    else:
        print(f"⚠️  Insights returned {response.status_code}: {response.text}")

def test_forecasting_list():
    """Test listing forecasts."""
    response = requests.get(f"{API_V1}/forecasting")
    print(f"   Forecasting list response: {response.status_code}")
    if response.status_code == 200:
        print("✅ Forecasting endpoint works")
    elif response.status_code == 401:
        print("⚠️  Forecasting endpoint requires authentication (expected)")
    else:
        print(f"⚠️  Forecasting returned {response.status_code}: {response.text}")

if __name__ == "__main__":
    print("\n🧪 Testing Auto Data Analyst API\n")
    
    try:
        test_health()
        print()
        test_auth_register()
        print()
        test_projects_list()
        test_datasets_list()
        test_insights_list()
        test_forecasting_list()
        
        print("\n✅ Basic API tests completed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
