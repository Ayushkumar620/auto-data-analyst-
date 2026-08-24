"""Tests for Interactive Real-Time DAG Execution Visualizer.

Verifies:
1. AutonomousCommandOrchestrator DAG graph node synthesis with rich metadata (intent, knowledge, plan, execution, validation, evidence)
2. Flask API `/api/analyze` response containing complete execution_graph DAG array
3. Web template index.html containing DAG rendering functions
4. Web styling style.css containing responsive DAG grid and animation keyframes
"""
import io
import json
import numpy as np
import pandas as pd
import pytest

from agent.command_orchestrator import AutonomousCommandOrchestrator, CommandExecutionResult
from app import app


@pytest.fixture
def sample_csv():
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "Region": np.random.choice(["North", "South", "East", "West"], n),
        "Sales": np.random.uniform(500, 3000, n),
        "Profit": np.random.uniform(50, 600, n),
    })
    csv_bytes = io.BytesIO()
    df.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)
    return csv_bytes


def test_orchestrator_dag_node_generation():
    """Verify 6-stage DAG nodes generated with metadata."""
    orchestrator = AutonomousCommandOrchestrator()
    df = pd.DataFrame({
        "Region": ["North", "South", "East", "West"],
        "Sales": [1000.0, 2000.0, 1500.0, 3000.0],
        "Profit": [100.0, 300.0, 200.0, 450.0],
    })

    result: CommandExecutionResult = orchestrator.execute_command(
        command="Analyze sales by region and find top drivers",
        dataframe=df,
        session_id="dag_test_sess",
    )

    assert result.execution_graph is not None
    assert len(result.execution_graph) == 6

    # Verify node IDs and structure
    node_ids = [node["id"] for node in result.execution_graph]
    assert "node_intent" in node_ids
    assert "node_knowledge" in node_ids
    assert "node_planner" in node_ids
    assert "node_execution" in node_ids
    assert "node_validation" in node_ids
    assert "node_evidence" in node_ids

    # Verify details and badges
    for node in result.execution_graph:
        assert "title" in node
        assert "agent" in node
        assert "status" in node
        assert "badge" in node
        assert "details" in node
        assert "icon" in node


def test_flask_api_analyze_returns_execution_graph(sample_csv):
    """Verify Flask /api/analyze endpoint includes execution_graph in JSON response."""
    client = app.test_client()

    data = {
        "file": (sample_csv, "test_sales.csv"),
        "command": "summary",
        "session_id": "test_flask_dag_session",
    }

    response = client.post(
        "/api/analyze",
        data=data,
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    res_json = json.loads(response.data)

    assert "execution_graph" in res_json
    assert isinstance(res_json["execution_graph"], list)
    assert len(res_json["execution_graph"]) == 6
    assert res_json["execution_graph"][0]["agent"] == "IntentAnalyzer"


def test_frontend_dag_artifacts():
    """Verify index.html and style.css contain DAG visualization components."""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    assert "renderExecutionGraphHtml" in html_content
    assert "execution-dag-card" in html_content
    assert "dag-flow-grid" in html_content

    with open("static/css/style.css", "r", encoding="utf-8") as f:
        css_content = f.read()

    assert ".dag-container" in css_content
    assert ".dag-flow-grid" in css_content
    assert ".dag-node-card" in css_content
    assert "pulseArrow" in css_content

