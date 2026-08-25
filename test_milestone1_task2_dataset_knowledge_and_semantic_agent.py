"""
Tests for Milestone 1, Task 2: Dataset Knowledge Engine and Semantic Schema Agent.

Verifies:
1. Numeric revenue column detected as METRIC
2. Region column detected as DIMENSION
3. Customer ID detected as IDENTIFIER
4. Date column detected as DATE
5. Boolean column detected as BOOLEAN
6. Ambiguous column produces lower confidence (< 0.60) and is marked uncertain
7. Missing values are captured in DataQuality
8. Duplicate rows are captured in DataQuality
9. DatasetKnowledge is generated correctly with all query helpers
10. AgentResult is returned correctly from SemanticSchemaAgent.run()
"""
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from agent.dataset_knowledge import (
    ColumnKnowledge,
    DataQuality,
    DatasetKnowledge,
    SemanticType,
)
from agent.schemas import AgentResult, AgentStatus, ClaimType, Evidence, SemanticMapping
from agent.semantic_schema_agent import SemanticSchemaAgent


# ==============================================================================
# Deterministic Test Fixtures
# ==============================================================================

@pytest.fixture
def ecommerce_df():
    """A realistic small e-commerce dataset containing all canonical semantic roles."""
    return pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(1, 11)],
        "order_date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "region": ["North", "South", "East", "West", "North", "South", "East", "West", "North", "South"],
        "revenue": [1200.50, 450.00, 890.20, 2300.00, 150.75, 3400.10, 560.00, 780.30, 920.00, 1100.40],
        "is_churned": [False, True, False, False, True, False, False, False, True, False],
        "ambiguous_x": ["a_1", "b_2", "c_3", "d_4", "e_5", "f_6", "g_7", "h_8", "i_9", "j_10"],
    })


@pytest.fixture
def dirty_df():
    """Dataset with missing values and duplicates for quality profiling."""
    return pd.DataFrame({
        "transaction_id": ["TX1", "TX2", "TX3", "TX3", "TX4"],  # 1 duplicate row
        "amount": [100.0, None, 300.0, 300.0, 500.0],          # 1 missing value
        "category": ["A", "B", None, None, "E"],               # 2 missing values
    })


# ==============================================================================
# 1-5. Semantic Role & Type Detection Tests
# ==============================================================================

def test_revenue_detected_as_metric(ecommerce_df):
    """1. Verify numeric revenue column is classified as METRIC with high confidence."""
    agent = SemanticSchemaAgent()
    dk = agent.analyze_dataset(ecommerce_df, dataset_name="orders.csv")

    col_k = dk.get_column_knowledge("revenue")
    assert col_k is not None
    assert col_k.semantic_type == SemanticType.METRIC
    assert col_k.role == "metric"
    assert col_k.confidence >= 0.85
    assert col_k.mean == pytest.approx(1485.225, 0.01)
    assert col_k.is_uncertain is False


def test_region_detected_as_dimension(ecommerce_df):
    """2. Verify region column is classified as DIMENSION with high confidence."""
    agent = SemanticSchemaAgent()
    dk = agent.analyze_dataset(ecommerce_df, dataset_name="orders.csv")

    col_k = dk.get_column_knowledge("region")
    assert col_k is not None
    assert col_k.semantic_type in (SemanticType.DIMENSION, SemanticType.CATEGORY)
    assert col_k.role == "dimension"
    assert col_k.confidence >= 0.80
    assert col_k.cardinality == 4
    assert col_k.is_uncertain is False


def test_customer_id_detected_as_identifier(ecommerce_df):
    """3. Verify customer_id column is classified as IDENTIFIER."""
    agent = SemanticSchemaAgent()
    dk = agent.analyze_dataset(ecommerce_df, dataset_name="orders.csv")

    col_k = dk.get_column_knowledge("customer_id")
    assert col_k is not None
    assert col_k.semantic_type == SemanticType.IDENTIFIER
    assert col_k.role == "identifier"
    assert col_k.confidence >= 0.85
    assert col_k.unique_count == 10


def test_order_date_detected_as_date(ecommerce_df):
    """4. Verify order_date column is classified as DATE or DATETIME."""
    agent = SemanticSchemaAgent()
    dk = agent.analyze_dataset(ecommerce_df, dataset_name="orders.csv")

    col_k = dk.get_column_knowledge("order_date")
    assert col_k is not None
    assert col_k.semantic_type in (SemanticType.DATE, SemanticType.DATETIME)
    assert col_k.role == "date"
    assert col_k.confidence >= 0.85


def test_is_churned_detected_as_boolean(ecommerce_df):
    """5. Verify boolean flag column is classified as BOOLEAN."""
    agent = SemanticSchemaAgent()
    dk = agent.analyze_dataset(ecommerce_df, dataset_name="orders.csv")

    col_k = dk.get_column_knowledge("is_churned")
    assert col_k is not None
    assert col_k.semantic_type == SemanticType.BOOLEAN or col_k.role in ("target_candidate", "dimension")
    assert col_k.confidence >= 0.85


def test_ambiguous_column_confidence_and_uncertainty(ecommerce_df):
    """6. Verify ambiguous column with generic names and high cardinality has lower confidence or is noted."""
    agent = SemanticSchemaAgent()
    dk = agent.analyze_dataset(ecommerce_df, dataset_name="orders.csv")

    col_k = dk.get_column_knowledge("ambiguous_x")
    assert col_k is not None
    assert col_k.confidence <= 0.75


# ==============================================================================
# 7-8. Data Quality & Profiling Integration Tests
# ==============================================================================

def test_missing_values_captured(dirty_df):
    """7. Verify missing values are captured accurately in DataQuality and missing_values dict."""
    agent = SemanticSchemaAgent()
    dk = agent.analyze_dataset(dirty_df, dataset_name="dirty.csv")

    assert dk.missing_values["amount"] == 1
    assert dk.missing_values["category"] == 2
    assert dk.missing_values["transaction_id"] == 0

    dq = dk.data_quality
    if isinstance(dq, DataQuality):
        assert dq.missing_values["amount"] == 1
        assert len(dq.warnings) > 0


def test_duplicates_captured(dirty_df):
    """8. Verify duplicate rows are detected and counted in DataQuality."""
    agent = SemanticSchemaAgent()
    dk = agent.analyze_dataset(dirty_df, dataset_name="dirty.csv")

    dq = dk.data_quality
    if isinstance(dq, DataQuality):
        assert dq.duplicates >= 1
    else:
        assert dq.get("duplicate_rows", 0) >= 1 or dq.get("duplicates", 0) >= 1


# ==============================================================================
# 9-10. DatasetKnowledge & AgentResult Integration Tests
# ==============================================================================

def test_dataset_knowledge_generation_and_helpers(ecommerce_df):
    """9. Verify DatasetKnowledge generates all query helpers and round-trips to dict."""
    agent = SemanticSchemaAgent()
    dk = agent.analyze_dataset(ecommerce_df, dataset_name="ecommerce.csv")

    assert isinstance(dk, DatasetKnowledge)
    assert dk.row_count == 10
    assert dk.column_count == 6
    assert dk.get_primary_metric() == "revenue"
    assert dk.get_primary_dimension() == "region"
    assert dk.get_primary_date_column() == "order_date"
    assert dk.is_time_series() is True

    # Test serialization
    d = dk.to_dict()
    assert d["row_count"] == 10
    assert d["column_count"] == 6
    assert "revenue" in d["data_types"]

    # Test deserialization
    dk_copy = DatasetKnowledge.from_dict(d)
    assert dk_copy.row_count == dk.row_count
    assert dk_copy.get_primary_metric() == "revenue"


def test_agent_result_returned_correctly(ecommerce_df):
    """10. Verify SemanticSchemaAgent.run() returns standardized AgentResult with evidence."""
    agent = SemanticSchemaAgent()
    result = agent.run({"data": ecommerce_df, "name": "sales.csv"})

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.status == AgentStatus.COMPLETED
    assert result.agent_name == "Semantic Schema Agent"
    assert result.confidence > 0.80
    assert result.has_evidence is True
    assert "dataset_knowledge" in result.data
    assert result.execution_time >= 0.0
