"""Comprehensive test suite for Phase 4: Intelligent ML Model Selection & Comparison Engine."""
import numpy as np
import pandas as pd
import pytest

from agent.schemas import AgentResult, AgentStatus, ClaimType
from agent.model_selection_agent import ModelSelectionAgent
from agent.planner import PlannerAgent
from backend.app.ml.model_selection import (
    MLModelComparisonEngine,
    ModelComparisonReport,
    ModelEvaluationResult,
    ProblemType,
)


# ==============================================================================
# 1. Problem Type & Dataset Characteristics Inspection Tests
# ==============================================================================

def test_problem_type_detection():
    """Verify problem type detection across binary, multiclass, and continuous targets."""
    engine = MLModelComparisonEngine()

    # Binary string / bool
    s_binary = pd.Series(["Yes", "No", "Yes", "Yes", "No"])
    assert engine.detect_problem_type(s_binary) == ProblemType.BINARY_CLASSIFICATION

    s_bool = pd.Series([True, False, True, False])
    assert engine.detect_problem_type(s_bool) == ProblemType.BINARY_CLASSIFICATION

    # Binary 0/1 integers
    s_int_bin = pd.Series([0, 1, 1, 0, 1, 0])
    assert engine.detect_problem_type(s_int_bin) == ProblemType.BINARY_CLASSIFICATION

    # Multiclass string
    s_multi = pd.Series(["Low", "Medium", "High", "Medium", "Low", "Critical"])
    assert engine.detect_problem_type(s_multi) == ProblemType.MULTICLASS_CLASSIFICATION

    # Continuous regression
    s_reg = pd.Series([10.5, 23.2, 45.1, 12.8, 98.4, 55.0, 67.2, 89.1, 34.6, 78.9])
    assert engine.detect_problem_type(s_reg) == ProblemType.REGRESSION


def test_inspect_dataset_characteristics():
    """Verify dataset metadata and class balance calculation."""
    engine = MLModelComparisonEngine()
    X = pd.DataFrame({
        "num_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "cat_1": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    })
    y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # Imbalanced 80/20

    chars = engine.inspect_dataset(X, y)
    assert chars["n_samples"] == 10
    assert chars["n_features"] == 2
    assert chars["numeric_features"] == 1
    assert chars["categorical_features"] == 1
    assert chars["is_imbalanced"] is True


# ==============================================================================
# 2. Candidate Model Pool Generation Tests
# ==============================================================================

def test_get_candidate_models():
    """Verify model candidate pools for classification and regression."""
    engine = MLModelComparisonEngine()

    clf_models = engine.get_candidate_models(ProblemType.BINARY_CLASSIFICATION, n_samples=100, n_features=5)
    clf_names = [m[0] for m in clf_models]
    assert "Logistic Regression" in clf_names
    assert "Decision Tree" in clf_names
    assert "Random Forest" in clf_names
    assert "Gradient Boosting" in clf_names
    assert "K-Nearest Neighbors" in clf_names
    assert "Support Vector Machine (SVC)" in clf_names

    reg_models = engine.get_candidate_models(ProblemType.REGRESSION, n_samples=100, n_features=5)
    reg_names = [m[0] for m in reg_models]
    assert "Linear Regression" in reg_names
    assert "Ridge Regression" in reg_names
    assert "Lasso Regression" in reg_names
    assert "Decision Tree" in reg_names
    assert "Random Forest" in reg_names
    assert "Gradient Boosting" in reg_names


# ==============================================================================
# 3. Model Benchmarking on Regression & Classification Datasets
# ==============================================================================

@pytest.fixture
def regression_df():
    np.random.seed(42)
    n = 60
    x1 = np.random.uniform(10, 100, n)
    x2 = np.random.uniform(1, 10, n)
    # y is a non-linear combination
    y = 5.0 * x1 + 2.5 * (x2 ** 2) + np.random.normal(0, 5, n)
    return pd.DataFrame({
        "feature_x1": x1,
        "feature_x2": x2,
        "category_tag": np.random.choice(["Tier1", "Tier2", "Tier3"], n),
        "target_value": y,
    })


@pytest.fixture
def classification_df():
    np.random.seed(42)
    n = 60
    tenure = np.random.uniform(1, 72, n)
    monthly_fee = np.random.uniform(20, 120, n)
    # probability of churn increases with high monthly fee and low tenure
    prob = 1 / (1 + np.exp(-(0.05 * monthly_fee - 0.08 * tenure)))
    churn = (prob > 0.5).astype(int)
    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly_fee,
        "contract_type": np.random.choice(["Month-to-Month", "One-Year", "Two-Year"], n),
        "is_churned": churn,
    })


def test_benchmark_models_regression(regression_df):
    """Test full benchmarking pipeline on regression data."""
    engine = MLModelComparisonEngine()
    report = engine.benchmark_models(
        dataframe=regression_df,
        target_column="target_value",
        cv_folds=3,
    )

    assert isinstance(report, ModelComparisonReport)
    assert report.problem_type == ProblemType.REGRESSION
    assert len(report.candidate_evaluations) >= 5
    assert report.best_model is not None
    assert report.best_model.status == "success"
    assert "r2_score" in report.best_model.metrics
    assert "rmse" in report.best_model.metrics
    assert len(report.leaderboard) == len([e for e in report.candidate_evaluations if e.status == "success"])
    assert len(report.selection_rationale) > 20

    # Best model should have high R2
    assert report.best_model.primary_metric_value > 0.5


def test_benchmark_models_classification(classification_df):
    """Test full benchmarking pipeline on classification data."""
    engine = MLModelComparisonEngine()
    report = engine.benchmark_models(
        dataframe=classification_df,
        target_column="is_churned",
        cv_folds=3,
    )

    assert isinstance(report, ModelComparisonReport)
    assert report.problem_type == ProblemType.BINARY_CLASSIFICATION
    assert len(report.candidate_evaluations) >= 5
    assert report.best_model is not None
    assert "accuracy" in report.best_model.metrics
    assert "f1_score" in report.best_model.metrics
    assert report.best_model.primary_metric_value >= 0.5


# ==============================================================================
# 4. ModelSelectionAgent & Planner Integration Tests
# ==============================================================================

def test_model_selection_agent_run(regression_df):
    """Test ModelSelectionAgent execution and AgentResult output."""
    agent = ModelSelectionAgent()
    result = agent.run({"data": regression_df, "target": "target_value", "cv_folds": 3})

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.agent == "Model Selection Agent"
    assert "best_model" in result.output
    assert "leaderboard" in result.output
    assert "selection_rationale" in result.output
    assert len(result.evidence) >= 1
    assert result.evidence[0].claim_type == ClaimType.FACT
    assert result.evidence[0].method == "cross_validation_benchmark"


def test_planner_agent_model_selection_action(classification_df):
    """Test routing action 'model_selection' through PlannerAgent."""
    planner = PlannerAgent(data=classification_df)
    result = planner.run_agent({"action": "model_selection", "target": "is_churned", "cv_folds": 3})

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.validation is not None
    assert result.validation.passed is True
    assert "best_model" in result.output
    assert "leaderboard" in result.output
