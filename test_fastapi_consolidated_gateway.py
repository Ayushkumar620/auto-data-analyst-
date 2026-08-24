"""Tests for Consolidated FastAPI Gateway Routers.

Verifies:
1. POST /api/v1/analyze - Autonomous command analysis with DAG execution graph
2. POST /api/v1/sql/introspect & /query - Live SQL connector gateway
3. POST /api/v1/sandbox/execute - AST-isolated Python code sandbox
4. POST /api/v1/vision/extract - Multi-modal vision feature extraction
5. POST /api/v1/reports/executive-pdf & /executive-deck - Presentation engine
"""
import sqlite3
from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_sqlite_db_path(tmp_path):
    db_file = str(tmp_path / "gateway_test.sqlite")
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("CREATE TABLE products (id INT PRIMARY KEY, name TEXT, price REAL);")
    cur.executemany("INSERT INTO products VALUES (?, ?, ?);", [
        (1, "Widget", 19.99),
        (2, "Gadget", 49.99),
    ])
    conn.commit()
    conn.close()
    return db_file


def test_api_v1_analyze_endpoint(client):
    """Verify POST /api/v1/analyze executes command and returns DAG and explanation."""
    payload = {
        "command": "Find top product by price and explain why",
        "dataset": [
            {"product": "Laptop", "price": 1200.0, "units": 10},
            {"product": "Phone", "price": 800.0, "units": 25},
            {"product": "Tablet", "price": 500.0, "units": 15},
        ],
        "session_id": "test_gateway_sess",
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "command" in data
    assert "user_intent" in data
    assert "execution_graph" in data
    assert "final_explanation" in data
    assert len(data["execution_graph"]) >= 5


def test_api_v1_sql_endpoints(client, sample_sqlite_db_path):
    """Verify POST /api/v1/sql/introspect and POST /api/v1/sql/query."""
    # 1. Introspect
    intro_resp = client.post("/api/v1/sql/introspect", json={"connection_uri": sample_sqlite_db_path})
    assert intro_resp.status_code == 200
    intro_data = intro_resp.json()
    assert "products" in intro_data["tables"]

    # 2. Query
    query_resp = client.post("/api/v1/sql/query", json={
        "connection_uri": sample_sqlite_db_path,
        "query": "SELECT * FROM products",
    })
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    assert q_data["rows_returned"] == 2
    assert q_data["records"][0]["name"] == "Widget"


def test_api_v1_sandbox_endpoint(client):
    """Verify POST /api/v1/sandbox/execute executes calculations safely."""
    payload = {
        "code": "result = df['val'].sum() * 2",
        "dataset": [{"val": 10}, {"val": 20}],
    }
    resp = client.post("/api/v1/sandbox/execute", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["result"] == 60


def test_api_v1_vision_endpoint(client):
    """Verify POST /api/v1/vision/extract extracts features from image matrices."""
    # Synthetic 2D image 8x8
    img_2d = np.ones((8, 8), dtype=float).tolist()
    payload = {
        "images": [img_2d, img_2d],
        "labels": ["sample_a", "sample_b"],
    }
    resp = client.post("/api/v1/vision/extract", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_images"] == 2
    assert data["labels_found"] == ["sample_a", "sample_b"]
    assert len(data["sample_records"]) == 2


def test_api_v1_executive_reports(client):
    """Verify POST /api/v1/reports/executive-pdf and executive-deck."""
    payload = {
        "title": "Quarterly Executive Brief",
        "command": "Analyze revenue growth",
        "explanation": "Revenue grew 18% driven by Enterprise tier expansion.",
        "kpis": {"Revenue": 1500000.0, "Growth": 0.18},
        "evidence_list": [{"claim_type": "FACT", "method": "SQL Aggregation", "artifact": "Total $1.5M"}],
    }

    # Deck
    deck_resp = client.post("/api/v1/reports/executive-deck", json=payload)
    assert deck_resp.status_code == 200
    deck_data = deck_resp.json()
    assert deck_data["total_slides"] >= 4

    # PDF
    pdf_resp = client.post("/api/v1/reports/executive-pdf", json=payload)
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert pdf_resp.content.startswith(b"%PDF")
