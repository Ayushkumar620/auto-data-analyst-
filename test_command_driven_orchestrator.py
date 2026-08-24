"""Comprehensive test suite for the Command-Driven Autonomous Agent Architecture.

Validates that the agent autonomously executes arbitrary natural-language outcome commands
without requiring predefined menus, buttons, or manual tool selection.
"""
import io
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.command_orchestrator import AutonomousCommandOrchestrator, CommandExecutionResult
from backend.app.main import app


@pytest.fixture
def enterprise_df():
    """Synthetic enterprise sales, customer, profit, and transaction dataset."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="W")
    countries = np.random.choice(["India", "US", "UK", "Germany"], n, p=[0.4, 0.3, 0.15, 0.15])
    customers = [f"Customer_{i%20}" for i in range(n)]
    revenue = np.random.uniform(5000, 50000, n)
    profit = revenue * 0.25 - np.random.uniform(500, 3000, n)
    # Add churn and missing values
    churn = (profit < 2000).astype(int)
    revenue[5] = np.nan  # Missing value to test automatic cleaning

    # Inject unusual outlier transaction
    revenue[10] = 350000.0

    return pd.DataFrame({
        "transaction_date": dates,
        "country": countries,
        "customer_name": customers,
        "revenue": revenue,
        "profit": profit,
        "churn": churn,
    })


# ==============================================================================
# 1. Core Command Execution & Lifecycle Stages Tests
# ==============================================================================

def test_command_analyze_sales_data(enterprise_df):
    """Command: 'Analyze my sales data.' -> Autonomous EDA, aggregations & insights."""
    orchestrator = AutonomousCommandOrchestrator()
    result = orchestrator.execute_command("Analyze my sales data.", enterprise_df)

    assert isinstance(result, CommandExecutionResult)
    assert result.user_intent in ("eda", "summary", "insight")
    assert len(result.required_operations) >= 2
    assert len(result.selected_agents) >= 1
    assert result.validation_summary["status"] in ("PASSED", "PASSED_WITH_WARNINGS")
    assert len(result.final_explanation) > 0
    assert result.visualization is not None


def test_command_why_did_profit_decrease(enterprise_df):
    """Command: 'Why did profit decrease last year?' -> Driver variance analysis with non-causal caveats."""
    orchestrator = AutonomousCommandOrchestrator()
    result = orchestrator.execute_command("Why did profit decrease last year?", enterprise_df)

    assert isinstance(result, CommandExecutionResult)
    assert "profit" in result.final_explanation.lower() or "revenue" in result.final_explanation.lower()
    assert any("variance" in op or "driver" in op for op in result.required_operations)
    # Non-causal caveat check
    assert "correlation" in result.final_explanation.lower() or "pattern" in result.final_explanation.lower()


def test_command_clean_and_find_top_customers(enterprise_df):
    """Command: 'Clean this dataset and find the top 10 customers.' -> Multi-tool composition."""
    orchestrator = AutonomousCommandOrchestrator()
    result = orchestrator.execute_command("Clean this dataset and find the top 10 customers.", enterprise_df)

    assert isinstance(result, CommandExecutionResult)
    assert any("CleaningAgent" in agent for agent in result.selected_agents) or any("clean" in op for op in result.required_operations)
    assert "customer" in result.final_explanation.lower()
    assert "top" in result.final_explanation.lower() or "1." in result.final_explanation


def test_command_compare_revenue_between_countries(enterprise_df):
    """Command: 'Compare revenue between India and the US.' -> Cross-cohort dimension aggregation."""
    orchestrator = AutonomousCommandOrchestrator()
    result = orchestrator.execute_command("Compare revenue between India and the US.", enterprise_df)

    assert isinstance(result, CommandExecutionResult)
    assert "revenue" in result.final_explanation.lower()
    assert "india" in result.final_explanation.lower() or "us" in result.final_explanation.lower() or "country" in result.final_explanation.lower()


def test_command_predict_sales(enterprise_df):
    """Command: 'Predict next month's sales.' -> Time series / predictive modeling."""
    orchestrator = AutonomousCommandOrchestrator()
    result = orchestrator.execute_command("Predict next month's sales.", enterprise_df)

    assert isinstance(result, CommandExecutionResult)
    assert result.user_intent in ("prediction", "forecasting")
    assert len(result.required_operations) >= 2


def test_command_build_best_model_for_churn(enterprise_df):
    """Command: 'Build the best model to predict customer churn.' -> Algorithm benchmarking & selection."""
    orchestrator = AutonomousCommandOrchestrator()
    result = orchestrator.execute_command("Build the best model to predict customer churn.", enterprise_df)

    assert isinstance(result, CommandExecutionResult)
    assert result.user_intent == "prediction"
    assert any("benchmark" in op or "model" in op for op in result.required_operations)
    if result.model_selection_summary:
        assert "model_name" in result.model_selection_summary


def test_command_find_unusual_transactions(enterprise_df):
    """Command: 'Find unusual transactions and explain them.' -> Anomaly & outlier diagnostics."""
    orchestrator = AutonomousCommandOrchestrator()
    result = orchestrator.execute_command("Find unusual transactions and explain them.", enterprise_df)

    assert isinstance(result, CommandExecutionResult)
    assert any("outlier" in op or "anomal" in op for op in result.required_operations)


def test_command_create_financial_report(enterprise_df):
    """Command: 'Create a report showing the financial performance.' -> Narrative report synthesis."""
    orchestrator = AutonomousCommandOrchestrator()
    result = orchestrator.execute_command("Create a report showing the financial performance.", enterprise_df)

    assert isinstance(result, CommandExecutionResult)
    assert len(result.final_explanation) > 0
    assert result.dataset_summary["rows"] == 100


# ==============================================================================
# 2. FastAPI Command-Driven REST API Endpoint Test
# ==============================================================================

def test_fastapi_chat_command_endpoint(enterprise_df):
    """Verify POST /api/v1/chat/command executes natural-language commands directly."""
    client = TestClient(app)

    csv_buffer = io.BytesIO()
    enterprise_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    files = {"file": ("enterprise.csv", csv_buffer.getvalue(), "text/csv")}
    data = {"command": "Compare revenue by country and clean data"}

    response = client.post("/api/v1/chat/command", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()

    assert "command" in res_json
    assert "user_intent" in res_json
    assert "required_operations" in res_json
    assert "selected_agents" in res_json
    assert "final_explanation" in res_json
    assert "validation_summary" in res_json
    assert len(res_json["required_operations"]) >= 2

