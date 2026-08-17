"""Tests for PHASE I: dataset-aware AI chat + approved tool system."""
import pandas as pd
import pytest

from backend.app.chat.agent import ChatAgent
from backend.app.chat.tools import ToolRegistry
from backend.app.chat.executor import DataExecutor


def _sales_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-05", "2024-01-12", "2024-02-02", "2024-02-09",
                                "2024-03-01", "2024-03-08"]),
        "region": ["North", "South", "North", "South", "North", "South"],
        "product": ["A", "B", "A", "B", "A", "B"],
        "sales": [1000, 500, 1200, 600, 900, 450],
        "profit": [100, 50, 120, 60, 90, 45],
    })


def _respond(dataframe, message):
    return ChatAgent().respond(dataframe, message)


def test_schema_question():
    response = _respond(_sales_df(), "what columns and schema does this dataset have")
    assert response.status == "success"
    assert response.intent == "schema"
    assert "revenue" in response.message


def test_which_region_highest_revenue():
    response = _respond(_sales_df(), "which region has the highest sales")
    assert response.intent == "aggregation"
    assert response.status == "success"
    assert "North" in response.message
    assert response.evidence["winner"] == "North"


def test_which_product_highest_profit():
    response = _respond(_sales_df(), "which product has the highest profit")
    assert response.intent == "aggregation"
    assert response.status == "success"
    assert response.evidence["winner"] == "A"


def test_why_did_revenue_decrease():
    # All revenue periods we can measure fluctuate; verify trend intent fires
    response = _respond(_sales_df(), "why did sales decrease")
    assert response.intent == "trend"
    assert response.status == "success"
    assert "sales" in response.message


def test_show_monthly_sales():
    response = _respond(_sales_df(), "show monthly sales")
    assert response.intent == "visualization"
    assert response.status == "success"
    # Should return a line chart visualization of revenue
    assert response.visualization is not None


def test_find_unusual_transactions():
    response = _respond(_sales_df(), "find unusual transactions in sales")
    assert response.intent == "anomaly_detection"
    assert response.status == "success"
    assert "anomal" in response.message.lower()


def test_predict_next_month_sales():
    response = _respond(_sales_df(), "predict next month's sales")
    assert response.intent == "forecast"
    assert response.status == "success"
    assert "forecast" in response.message.lower()


def test_correlation_question():
    response = _respond(_sales_df(), "are revenue and profit correlated")
    assert response.intent == "correlation"
    assert response.status == "success"
    assert response.evidence["correlation"] is not None


def test_ambiguous_question_asks_clarification():
    response = _respond(pd.DataFrame({"value": [1, 2, 3]}), "which is best")
    assert response.status == "needs_clarification"


def test_empty_dataset_returns_supportive_error():
    from backend.app.chat.agent import ChatAgent
    response = ChatAgent().respond(pd.DataFrame(), "total revenue")
    assert response.status == "unsupported"


def test_tool_registry_only_allows_approved_tools():
    registry = ToolRegistry()
    names = registry.names
    expected = {
        "get_dataset_schema", "get_column_statistics", "filter_data",
        "aggregate_data", "group_by", "calculate_growth", "detect_anomalies",
        "calculate_correlation", "create_bar_chart", "create_line_chart",
        "create_scatter_chart", "create_histogram", "run_eda",
        "generate_insights", "forecast",
    }
    assert expected <= set(names)
    with pytest.raises(ValueError):
        registry.execute("arbitrary_code", _sales_df())


def test_tool_evidence_is_deterministic_and_referenced():
    executor = DataExecutor()
    schema = executor.get_dataset_schema(_sales_df())
    assert schema["rows"] == 6
    assert "revenue" in schema["columns"]
    group = executor.group_by(_sales_df(), "region", "revenue", "sum")
    assert group[0]["region"] == "North"
    assert group[0]["revenue"] == 3100
    stats = executor.get_column_statistics(_sales_df(), "revenue")
    assert stats["mean"] == pytest.approx(775.0)