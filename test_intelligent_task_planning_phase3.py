"""Comprehensive test suite for Phase 3: Intelligent Task Planning & Intent Routing."""
from datetime import datetime
import pandas as pd
import pytest

from agent.schemas import AgentResult, AgentStatus, ClaimType
from agent.intent import AnalyticalIntent, IntentAnalyzer, IntentClassificationResult
from agent.dynamic_planner import DynamicTaskPlanner, PlanStep, TaskPlan
from agent.planner import PlannerAgent
from backend.app.core.dataset_knowledge import DatasetKnowledge
from backend.app.core.semantic import SemanticSchemaAgent


# ==============================================================================
# 1. IntentAnalyzer Tests
# ==============================================================================

def test_intent_analyzer_single_intents():
    """Test IntentAnalyzer on standard single-objective queries."""
    analyzer = IntentAnalyzer()

    # EDA Intent
    res_eda = analyzer.analyze("Show me descriptive statistics and column summary")
    assert res_eda.primary_intent == AnalyticalIntent.EDA
    assert res_eda.confidence >= 0.7

    # Cleaning Intent
    res_clean = analyzer.analyze("Preprocess the data and impute all missing values")
    assert res_clean.primary_intent == AnalyticalIntent.CLEANING
    assert res_clean.needs_cleaning is True

    # Prediction Intent
    res_pred = analyzer.analyze("Train a supervised classification model to predict churn")
    assert res_pred.primary_intent == AnalyticalIntent.PREDICTION

    # Forecasting Intent
    res_fc = analyzer.analyze("Forecast sales for the next 6 months")
    assert res_fc.primary_intent == AnalyticalIntent.FORECASTING
    assert res_fc.time_horizon == 6

    # Visualization Intent
    res_viz = analyzer.analyze("Create a bar chart of revenue by region")
    assert res_viz.primary_intent == AnalyticalIntent.VISUALIZATION
    assert res_viz.chart_type == "bar"

    # Anomalies Intent
    res_anom = analyzer.analyze("Detect unusual anomalies and outlier spikes in transaction_amount")
    assert res_anom.primary_intent == AnalyticalIntent.ANOMALIES


def test_intent_analyzer_multi_step_complex_query():
    """Test IntentAnalyzer on a multi-objective prompt with top-k and semantic targets."""
    df = pd.DataFrame({
        "customer_id": [f"CUST_{i}" for i in range(20)],
        "monthly_charges": [50.0 + i for i in range(20)],
        "tenure_months": [12 + i for i in range(20)],
        "is_churned": [0 if i % 2 == 0 else 1 for i in range(20)],
    })
    agent = SemanticSchemaAgent()
    knowledge = agent.build_knowledge(df, dataset_id="churn_data")

    analyzer = IntentAnalyzer()
    query = "Clean the dataset, train a model to predict churn, and explain the top 3 drivers"
    result = analyzer.analyze(query, knowledge=knowledge, dataframe=df)

    assert result.primary_intent == AnalyticalIntent.PREDICTION
    assert result.needs_cleaning is True
    assert result.needs_explanation is True
    assert result.top_k == 3
    assert result.target_column == "is_churned"  # Mapped concept churn -> is_churned
    assert len(result.reasoning) > 0


# ==============================================================================
# 2. DynamicTaskPlanner DAG Plan Creation Tests
# ==============================================================================

@pytest.fixture
def sample_dataset():
    return pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=25, freq="D"),
        "customer_id": [f"CUST_{i:03d}" for i in range(25)],
        "sales_amount": [100.0 + i * 15 for i in range(25)],
        "cost_of_goods": [60.0 + i * 8 for i in range(25)],
        "net_profit": [40.0 + i * 7 for i in range(25)],
        "units_sold": [2, 3, 1, 4, 2, 5, 3, 2, 4, 3, 2, 5, 4, 3, 6, 2, 4, 3, 5, 4, 3, 2, 4, 3, 5],
        "category": ["Electronics" if i % 2 == 0 else "Apparel" for i in range(25)],
    })


def test_create_plan_multi_step_dag(sample_dataset):
    """Test creating a multi-step execution plan with dependencies and validation."""
    planner = DynamicTaskPlanner()
    query = "Clean the dataset, train a model to predict net_profit, and explain the top 3 drivers"
    plan = planner.create_plan(query, dataframe=sample_dataset)

    assert isinstance(plan, TaskPlan)
    assert len(plan.steps) >= 3

    step_actions = [s.action for s in plan.steps]
    assert "clean" in step_actions
    assert "predict" in step_actions
    assert "explain_drivers" in step_actions
    assert "report" in step_actions

    # Check dependencies
    predict_step = next(s for s in plan.steps if s.action == "predict")
    clean_step = next(s for s in plan.steps if s.action == "clean")
    assert clean_step.step_id in predict_step.dependencies

    # Check serialization
    plan_dict = plan.to_dict()
    assert plan_dict["plan_id"] == plan.plan_id
    assert len(plan_dict["steps"]) == len(plan.steps)


def test_create_plan_forecasting_workflow(sample_dataset):
    """Test dynamic plan creation for a time series forecasting query."""
    planner = DynamicTaskPlanner()
    query = "Forecast sales for the next 7 periods and display a line chart"
    plan = planner.create_plan(query, dataframe=sample_dataset)

    assert isinstance(plan, TaskPlan)
    step_actions = [s.action for s in plan.steps]
    assert "forecast" in step_actions
    assert "report" in step_actions

    fc_step = next(s for s in plan.steps if s.action == "forecast")
    assert fc_step.parameters.get("periods") == 7


# ==============================================================================
# 3. Dynamic Plan Execution & PlannerAgent Integration Tests
# ==============================================================================

def test_execute_plan_end_to_end(sample_dataset):
    """Test full execution of a dynamic DAG plan returning a validated AgentResult."""
    planner = DynamicTaskPlanner()
    query = "Clean the dataset, train a model to predict net_profit, and explain the top 3 drivers"
    plan = planner.create_plan(query, dataframe=sample_dataset)
    result = planner.execute_plan(plan, dataframe=sample_dataset)

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.agent == "DynamicTaskPlanner"
    assert "steps_executed" in result.output
    assert "report" in result.output
    assert len(result.evidence) > 0
    assert 0.0 < result.confidence <= 1.0


def test_planner_agent_plan_and_execute(sample_dataset):
    """Test PlannerAgent.plan_and_execute high-level entrypoint."""
    planner_agent = PlannerAgent(data=sample_dataset)
    query = "Forecast sales for the next 4 periods and summarize results"

    plan = planner_agent.plan(query)
    assert isinstance(plan, TaskPlan)
    assert len(plan.steps) >= 2

    result = planner_agent.plan_and_execute(query)
    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.validation is not None
    assert result.validation.passed is True
