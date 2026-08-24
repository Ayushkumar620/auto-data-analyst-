"""Comprehensive test suite for Phase 2: Dataset Intelligence & Semantic Schema Agent."""
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from agent.schemas import ClaimType, Evidence, SemanticMapping
from backend.app.core.dataset_knowledge import DatasetKnowledge
from backend.app.core.semantic import SemanticSchemaAgent, BUSINESS_CONCEPTS


# ==============================================================================
# 1. DatasetKnowledge Class & Methods Tests
# ==============================================================================

def test_dataset_knowledge_instantiation_and_helpers():
    """Test creating DatasetKnowledge and utilizing query helpers."""
    ev = Evidence(
        source="SemanticSchemaAgent",
        method="concept_matching",
        data_ref={"column": "revenue_usd"},
        confidence=0.95,
        claim_type=ClaimType.FACT,
    )
    metric_map = SemanticMapping(
        column_name="revenue_usd",
        semantic_concept="revenue",
        concept_category="financial",
        confidence=0.95,
        evidence=[ev],
        aliases=["sales", "turnover", "income"],
        description="Total financial revenue in USD",
    )
    date_map = SemanticMapping(
        column_name="order_date",
        semantic_concept="timestamp",
        concept_category="temporal",
        confidence=0.98,
        evidence=[ev],
        aliases=["date", "datetime"],
        description="Order placement date",
    )
    dim_map = SemanticMapping(
        column_name="region",
        semantic_concept="location",
        concept_category="geography",
        confidence=0.85,
        evidence=[ev],
        aliases=["zone", "territory"],
        description="Geographic region",
    )

    dk = DatasetKnowledge(
        dataset_id="retail_dataset_001",
        dataset_type="financial",
        columns=["order_date", "revenue_usd", "region"],
        data_types={"order_date": "datetime64[ns]", "revenue_usd": "float64", "region": "object"},
        semantic_meanings={"revenue_usd": "Total financial revenue", "order_date": "Date"},
        metrics=[metric_map],
        dimensions=[dim_map],
        date_columns=[date_map],
        identifiers=[],
        semantic_mappings=[metric_map, date_map, dim_map],
        categorical_columns=["region"],
        numeric_columns=["revenue_usd"],
        relationships=[],
        missing_values={"revenue_usd": {"count": 0, "percentage": 0.0}},
        data_quality={"quality_score": 98, "issues": []},
        confidence_scores={"revenue_usd": 0.95, "order_date": 0.98, "region": 0.85},
        overall_confidence=0.93,
    )

    # Test query helpers
    assert dk.get_primary_metric() == "revenue_usd"
    assert dk.get_primary_date_column() == "order_date"
    assert dk.get_primary_dimension() == "region"
    assert dk.is_time_series() is True

    # Test find_columns_by_concept
    assert dk.find_columns_by_concept("revenue") == ["revenue_usd"]
    assert dk.find_columns_by_concept("sales") == ["revenue_usd"]  # alias match
    assert dk.find_columns_by_concept("turnover") == ["revenue_usd"]  # alias match
    assert dk.find_columns_by_concept("location") == ["region"]

    # Test get_column_mapping
    mapping = dk.get_column_mapping("revenue_usd")
    assert mapping is not None
    assert mapping.semantic_concept == "revenue"
    assert mapping.concept_category == "financial"


def test_dataset_knowledge_serialization_roundtrip():
    """Test that DatasetKnowledge correctly serializes to dict and deserializes back."""
    ev = Evidence(
        source="SemanticSchemaAgent",
        method="concept_matching",
        data_ref={"column": "profit"},
        confidence=0.92,
        claim_type=ClaimType.FACT,
    )
    metric_map = SemanticMapping(
        column_name="profit",
        semantic_concept="profit",
        concept_category="financial",
        confidence=0.92,
        evidence=[ev],
        aliases=["net_profit", "earnings"],
    )
    dk = DatasetKnowledge(
        dataset_id="test_ds",
        dataset_type="tabular",
        columns=["profit"],
        data_types={"profit": "float64"},
        metrics=[metric_map],
        semantic_mappings=[metric_map],
        numeric_columns=["profit"],
        confidence_scores={"profit": 0.92},
        overall_confidence=0.92,
    )

    d = dk.to_dict()
    assert isinstance(d, dict)
    assert d["dataset_id"] == "test_ds"
    assert len(d["metrics"]) == 1
    assert d["metrics"][0]["semantic_concept"] == "profit"

    reconstituted = DatasetKnowledge.from_dict(d)
    assert reconstituted.dataset_id == "test_ds"
    assert len(reconstituted.metrics) == 1
    assert reconstituted.metrics[0].column_name == "profit"
    assert reconstituted.metrics[0].semantic_concept == "profit"
    assert reconstituted.overall_confidence == 0.92


# ==============================================================================
# 2. SemanticSchemaAgent Concept Matching & Disambiguation Tests
# ==============================================================================

def test_semantic_matching_revenue_synonyms():
    """Verify that sales, sales_amount, revenue, net_revenue, and turnover map to revenue with high confidence."""
    agent = SemanticSchemaAgent()
    revenue_variations = [
        "sales",
        "sales_amount",
        "revenue",
        "net_revenue",
        "turnover",
        "gross_revenue",
        "total_sales",
        "revenue_usd",
    ]

    for col in revenue_variations:
        series = pd.Series([100.0, 200.0, 150.0, 300.0, 250.0])
        mapping = agent.match_concept(col, series=series)
        assert mapping.semantic_concept == "revenue", f"Failed for column name: {col}"
        assert mapping.concept_category == "financial"
        assert mapping.confidence >= 0.8, f"Confidence too low for {col}: {mapping.confidence}"
        assert len(mapping.evidence) > 0


def test_semantic_matching_cost_and_profit_disambiguation():
    """Verify that profit and cost concepts are not confused with revenue."""
    agent = SemanticSchemaAgent()

    # Profit variations
    for col in ["profit", "net_profit", "gross_profit", "ebitda", "operating_margin"]:
        series = pd.Series([20.0, -5.0, 35.0, 50.0, 10.0])
        mapping = agent.match_concept(col, series=series)
        assert mapping.semantic_concept == "profit", f"Failed for {col}"
        assert mapping.concept_category == "financial"

    # Cost variations
    for col in ["cogs", "cost", "expenses", "operating_expenses", "total_cost"]:
        series = pd.Series([80.0, 90.0, 100.0, 75.0, 85.0])
        mapping = agent.match_concept(col, series=series)
        assert mapping.semantic_concept == "cost", f"Failed for {col}"
        assert mapping.concept_category == "financial"


def test_semantic_matching_volume_pricing_customer_hr():
    """Verify volume, pricing, customer, HR, and churn concept classifications."""
    agent = SemanticSchemaAgent()

    # Quantity / Units
    for col in ["quantity", "qty", "units_sold", "item_count"]:
        series = pd.Series([1, 5, 2, 10, 3])
        mapping = agent.match_concept(col, series=series)
        assert mapping.semantic_concept == "quantity"
        assert mapping.concept_category == "volume"

    # Unit Price
    for col in ["unit_price", "price", "selling_price"]:
        series = pd.Series([19.99, 49.99, 9.99, 99.00])
        mapping = agent.match_concept(col, series=series)
        assert mapping.semantic_concept == "price"
        assert mapping.concept_category == "pricing"

    # Salary / Wages
    for col in ["salary", "annual_salary", "wage", "base_pay"]:
        series = pd.Series([60000, 75000, 50000, 90000])
        mapping = agent.match_concept(col, series=series)
        assert mapping.semantic_concept == "salary"
        assert mapping.concept_category == "hr"

    # Churn / Attrition
    for col in ["churn", "is_churned", "attrition", "cancellation_flag"]:
        series = pd.Series([0, 1, 0, 0, 1])
        mapping = agent.match_concept(col, series=series)
        assert mapping.semantic_concept == "churn"
        assert mapping.concept_category == "customer"


# ==============================================================================
# 3. Complete build_knowledge Integration Tests
# ==============================================================================

def test_build_knowledge_on_ecommerce_dataset():
    """Test build_knowledge constructs a rich, connected DatasetKnowledge object."""
    df = pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=20, freq="D"),
        "customer_id": [f"CUST_{i:03d}" for i in range(20)],
        "sales_amount": [100.0 + i * 10 for i in range(20)],
        "cost_of_goods": [60.0 + i * 6 for i in range(20)],
        "net_profit": [40.0 + i * 4 for i in range(20)],  # Exact identity: sales_amount - cost_of_goods = net_profit
        "units_sold": [2, 3, 1, 4, 2, 5, 3, 2, 4, 3, 2, 5, 4, 3, 6, 2, 4, 3, 5, 4],
        "category": ["Electronics" if i % 2 == 0 else "Apparel" for i in range(20)],
        "region": ["North", "South", "East", "West"] * 5,
    })

    agent = SemanticSchemaAgent()
    knowledge = agent.build_knowledge(df, dataset_id="ecommerce_2024")

    assert isinstance(knowledge, DatasetKnowledge)
    assert knowledge.dataset_id == "ecommerce_2024"
    assert knowledge.dataset_type in ("time_series", "financial")
    assert len(knowledge.columns) == 8
    assert len(knowledge.numeric_columns) == 4  # sales_amount, cost_of_goods, net_profit, units_sold
    assert "order_date" in [m.column_name for m in knowledge.date_columns]
    assert "customer_id" in [m.column_name for m in knowledge.identifiers]

    # Check metrics
    metric_names = [m.column_name for m in knowledge.metrics]
    assert "sales_amount" in metric_names
    assert "cost_of_goods" in metric_names
    assert "net_profit" in metric_names
    assert "units_sold" in metric_names

    # Check dimensions
    dim_names = [d.column_name for d in knowledge.dimensions]
    assert "category" in dim_names
    assert "region" in dim_names

    # Check data quality score
    assert "quality_score" in knowledge.data_quality
    assert knowledge.data_quality["quality_score"] >= 90

    # Check relationship discovery (e.g. sales_amount - cost_of_goods = net_profit or correlations)
    assert len(knowledge.relationships) > 0
    formulas = [r.get("formula") for r in knowledge.relationships if "formula" in r]
    # Check that a derived metric formula was detected
    assert any("sales_amount - cost_of_goods" in f or "cost_of_goods + net_profit" in f for f in formulas if f)

    # Check overall confidence
    assert 0.8 <= knowledge.overall_confidence <= 1.0

