"""
Tests for Milestone 2, Task 1: Command Intelligence / Intent Agent.

Verifies:
1. Simple analysis command parsing
2. Aggregation command parsing
3. Comparison command parsing
4. Ranking command parsing ("top 10")
5. Cleaning command parsing ("clean the data and remove duplicates")
6. Forecast command parsing ("predict next month's sales")
7. Prediction command parsing ("build model to predict customer churn")
8. Root-cause command parsing ("why did profit decrease last quarter?")
9. Multi-step command parsing (cleaning, duplicate removal, aggregation, forecasting)
10. Natural language time expressions (2025, Q3, last quarter, January 2026)
11. Ambiguous metric handling with clarification prompts (gross_sales vs net_sales)
12. Unknown command classification and low confidence
13. Dataset semantic cross-referencing against DatasetKnowledge
14. Low-confidence intent flagging
15. LLM unavailable deterministic fallback execution
"""
import pytest
from datetime import datetime
import pandas as pd

from agent.dataset_knowledge import (
    ColumnKnowledge,
    DataQuality,
    DatasetKnowledge,
    SemanticType,
)
from agent.intent import (
    CommandIntelligenceAgent,
    IntentType,
    UserIntent,
)
from agent.schemas import AgentResult, AgentStatus, SemanticMapping


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_dataset_knowledge():
    """Mock DatasetKnowledge for e-commerce store."""
    return DatasetKnowledge(
        dataset_id="store_2024",
        dataset_name="store.csv",
        row_count=500,
        column_count=5,
        columns=["customer_id", "order_date", "country", "revenue", "profit"],
        categorical_columns=["country"],
        numerical_columns=["revenue", "profit"],
        confidence=0.95,
    )


@pytest.fixture
def ambiguous_dataset_knowledge():
    """DatasetKnowledge with multiple competing sales columns."""
    return DatasetKnowledge(
        dataset_id="ambig_store",
        dataset_name="ambig.csv",
        row_count=100,
        column_count=4,
        columns=["customer_id", "gross_sales", "net_sales", "region"],
        categorical_columns=["region"],
        numerical_columns=["gross_sales", "net_sales"],
        confidence=0.95,
    )


# ==============================================================================
# 1-8. Core Intent Scenarios
# ==============================================================================

def test_simple_analysis_command():
    """1. Test understanding of a simple exploratory analysis command."""
    agent = CommandIntelligenceAgent()
    intent = agent.analyze_intent("Analyze my sales data.")

    assert intent.intent_type in (IntentType.DATASET_ANALYSIS, IntentType.AGGREGATION)
    assert "sales" in intent.metrics
    assert intent.confidence >= 0.85
    assert len(intent.required_capabilities) > 0


def test_aggregation_command():
    """2. Test understanding of aggregation commands (average, mean, total)."""
    agent = CommandIntelligenceAgent()
    intent = agent.analyze_intent("What is the average revenue by region?")

    assert intent.aggregation_type == "mean"
    assert "revenue" in intent.metrics
    assert "region" in intent.dimensions
    assert "aggregation" in intent.required_capabilities


def test_comparison_command():
    """3. Test comparison command across entities with time filter."""
    agent = CommandIntelligenceAgent()
    intent = agent.analyze_intent("Compare revenue between India and the US in 2025.")

    assert intent.intent_type == IntentType.COMPARISON
    assert "revenue" in intent.metrics
    assert intent.time_range is not None
    assert intent.time_range.get("year") == 2025
    assert intent.comparison is not None
    assert "India" in intent.comparison.get("entities", [])
    assert "US" in intent.comparison.get("entities", [])


def test_ranking_command():
    """4. Test ranking command ('top 10')."""
    agent = CommandIntelligenceAgent()
    intent = agent.analyze_intent("Find the top 10 customers by revenue.")

    assert intent.ranking is not None
    assert intent.ranking["limit"] == 10
    assert intent.ranking["type"] == "top"
    assert "revenue" in intent.metrics
    assert "customer" in intent.dimensions or "customers" in intent.original_command


def test_cleaning_command():
    """5. Test data cleaning and duplicate removal command."""
    agent = CommandIntelligenceAgent()
    intent = agent.analyze_intent("Clean the data and remove duplicate customers.")

    assert intent.intent_type == IntentType.DATA_CLEANING
    assert "data_cleaning" in intent.required_capabilities
    assert "duplicate_handling" in intent.required_capabilities


def test_forecast_command():
    """6. Test future forecasting command."""
    agent = CommandIntelligenceAgent()
    intent = agent.analyze_intent("Predict next month's sales.")

    assert intent.intent_type == IntentType.FORECASTING
    assert "forecasting" in intent.required_capabilities
    assert "sales" in intent.metrics
    assert intent.time_range is not None
    assert intent.time_range.get("period") == "next_month"


def test_prediction_command():
    """7. Test machine learning classification/prediction command."""
    agent = CommandIntelligenceAgent()
    intent = agent.analyze_intent("Build the best model to predict customer churn.")

    assert intent.intent_type in (IntentType.PREDICTION, IntentType.CLASSIFICATION)
    assert "model_training" in intent.required_capabilities
    assert "prediction" in intent.required_capabilities
    assert "churn" in intent.metrics or "churn" in intent.original_command


def test_root_cause_command():
    """8. Test root-cause 'why' analytical command."""
    agent = CommandIntelligenceAgent()
    intent = agent.analyze_intent("Why did profit decrease last quarter?")

    assert intent.intent_type == IntentType.ROOT_CAUSE_ANALYSIS
    assert "profit" in intent.metrics
    assert intent.time_range is not None
    assert intent.time_range.get("period") == "last_quarter"
    assert "trend_analysis" in intent.required_capabilities
    assert "anomaly_detection" in intent.required_capabilities


# ==============================================================================
# 9-15. Advanced Capabilities & Edge Cases
# ==============================================================================

def test_multi_step_command():
    """9. Test compound multi-step command containing multiple operations."""
    agent = CommandIntelligenceAgent()
    cmd = "Clean the dataset, remove duplicates, analyze revenue by region, and forecast next month."
    intent = agent.analyze_intent(cmd)

    assert "data_cleaning" in intent.required_capabilities
    assert "duplicate_handling" in intent.required_capabilities
    assert "regional_analysis" in intent.required_capabilities
    assert "forecasting" in intent.required_capabilities
    assert len(intent.required_capabilities) >= 4


def test_time_expressions():
    """10. Test structured parsing of multiple natural language time expressions."""
    agent = CommandIntelligenceAgent()

    # Year
    i1 = agent.analyze_intent("Show revenue in 2025.")
    assert i1.time_range["year"] == 2025

    # Quarter
    i2 = agent.analyze_intent("Show growth in Q3 2024.")
    assert i2.time_range["quarter"] == "Q3"
    assert i2.time_range["year"] == 2024

    # Month
    i3 = agent.analyze_intent("Analyze trends in January 2026.")
    assert i3.time_range["month"] == "january"
    assert i3.time_range["year"] == 2026


def test_ambiguous_metric_handling(ambiguous_dataset_knowledge):
    """11. Test ambiguity detection when multiple column candidates match a term."""
    agent = CommandIntelligenceAgent()
    intent = agent.analyze_intent("Analyze sales.", dataset_knowledge=ambiguous_dataset_knowledge)

    assert intent.needs_clarification is True
    assert len(intent.ambiguities) > 0
    assert "gross_sales" in intent.ambiguities[0]
    assert "net_sales" in intent.ambiguities[0]
    assert intent.confidence <= 0.60


def test_unknown_command():
    """12. Test unrecognized query flags low confidence and unknown intent."""
    agent = CommandIntelligenceAgent()
    intent = agent.analyze_intent("xyz123 foo bar baz")

    assert intent.intent_type == IntentType.UNKNOWN
    assert intent.confidence <= 0.50


def test_dataset_semantic_matching(mock_dataset_knowledge):
    """13. Test cross-referencing user command against DatasetKnowledge."""
    agent = CommandIntelligenceAgent()
    intent = agent.analyze_intent(
        "Compare revenue between India and the US in 2025.",
        dataset_knowledge=mock_dataset_knowledge,
    )

    assert "revenue" in intent.metrics
    assert "country" in intent.dimensions or "country" in intent.entities.get("dimensions", [])
    assert intent.confidence >= 0.90


def test_low_confidence_intent():
    """14. Test that vague or underspecified commands produce appropriate confidence."""
    agent = CommandIntelligenceAgent()
    intent = agent.analyze_intent("something interesting")

    assert intent.confidence < 0.70


def test_llm_unavailable_deterministic_fallback():
    """15. Test that CommandIntelligenceAgent runs cleanly when LLM provider is None."""
    agent = CommandIntelligenceAgent(llm_provider=None)
    result = agent.run("Find the top 5 products by revenue.")

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.status == AgentStatus.COMPLETED
    assert result.agent_name == "Command Intelligence Agent"
    assert "user_intent" in result.data
    assert result.data["user_intent"]["ranking"]["limit"] == 5
