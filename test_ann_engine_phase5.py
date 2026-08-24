"""Comprehensive test suite for Phase 5: Modular Artificial Neural Network (ANN) Engine."""
import numpy as np
import pandas as pd
import pytest

from agent.schemas import AgentResult, AgentStatus, ClaimType
from agent.ann_agent import ANNAgent
from agent.planner import PlannerAgent
from backend.app.ml.ann_engine import (
    ANNEngine,
    ANNHyperparameters,
    ANNTrainingResult,
)
from backend.app.ml.model_selection import ProblemType


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def regression_dataset():
    np.random.seed(42)
    n = 60
    x1 = np.random.uniform(5, 50, n)
    x2 = np.random.uniform(1, 10, n)
    # Non-linear relationship
    y = 2.5 * x1 + 1.8 * (x2 ** 2) + np.random.normal(0, 2, n)
    return pd.DataFrame({
        "feature_a": x1,
        "feature_b": x2,
        "category_c": np.random.choice(["Type1", "Type2", "Type3"], n),
        "target_metric": y,
    })


@pytest.fixture
def classification_dataset():
    np.random.seed(42)
    n = 60
    f1 = np.random.uniform(10, 100, n)
    f2 = np.random.uniform(1, 10, n)
    prob = 1 / (1 + np.exp(-(0.05 * f1 - 0.3 * f2)))
    labels = (prob > 0.5).astype(int)
    return pd.DataFrame({
        "score_1": f1,
        "score_2": f2,
        "group_tag": np.random.choice(["Alpha", "Beta"], n),
        "target_class": labels,
    })


# ==============================================================================
# 1. Hyperparameters & Preprocessing Tests
# ==============================================================================

def test_ann_hyperparameters_defaults_and_dict():
    """Verify ANNHyperparameters defaults and serialization."""
    hp = ANNHyperparameters(hidden_layer_sizes=(64, 32), activation="tanh", max_iter=150)
    d = hp.to_dict()
    assert d["hidden_layer_sizes"] == [64, 32]
    assert d["activation"] == "tanh"
    assert d["max_iter"] == 150
    assert d["early_stopping"] is True


def test_ann_data_preparation(regression_dataset):
    """Verify feature normalization and target preparation for neural networks."""
    engine = ANNEngine()
    X, y, features, problem_type = engine.prepare_data(
        dataframe=regression_dataset,
        target_column="target_metric",
    )

    assert problem_type == ProblemType.REGRESSION
    assert X.shape == (60, 3)  # feature_a, feature_b, category_c
    assert len(y) == 60
    assert "feature_a" in features
    # Check that X is standard scaled (mean approx 0, std approx 1)
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-2)
    assert np.allclose(X.std(axis=0), 1.0, atol=1e-2)


# ==============================================================================
# 2. ANN Regression & Classification Training Tests
# ==============================================================================

def test_ann_train_and_evaluate_regression(regression_dataset):
    """Verify ANN training, loss curve extraction, and ML comparison for regression."""
    engine = ANNEngine()
    hp = ANNHyperparameters(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42)
    result = engine.train_and_evaluate(
        dataframe=regression_dataset,
        target_column="target_metric",
        hyperparams=hp,
        compare_with_ml=True,
    )

    assert isinstance(result, ANNTrainingResult)
    assert result.status == "success"
    assert result.problem_type == ProblemType.REGRESSION
    assert "r2_score" in result.metrics
    assert "rmse" in result.metrics
    assert result.primary_metric_name == "r2_score"
    assert result.primary_metric_value > 0.4

    # Check loss curve and training epochs
    assert len(result.loss_curve) > 0
    assert result.epochs_trained > 0
    assert "Input(3) -> Dense(64, relu) -> Dense(32, relu) -> Output(1)" in result.architecture_summary

    # Check comparison with traditional ML baseline
    assert "best_traditional_ml_model" in result.comparison_with_ml
    assert "ann_score" in result.comparison_with_ml


def test_ann_train_and_evaluate_classification(classification_dataset):
    """Verify ANN training and accuracy/F1 evaluation for classification."""
    engine = ANNEngine()
    hp = ANNHyperparameters(hidden_layer_sizes=(32, 16), max_iter=150, random_state=42)
    result = engine.train_and_evaluate(
        dataframe=classification_dataset,
        target_column="target_class",
        hyperparams=hp,
        compare_with_ml=True,
    )

    assert isinstance(result, ANNTrainingResult)
    assert result.status == "success"
    assert result.problem_type == ProblemType.BINARY_CLASSIFICATION
    assert "accuracy" in result.metrics
    assert "f1_score" in result.metrics
    assert result.primary_metric_value >= 0.5
    assert len(result.loss_curve) > 0


# ==============================================================================
# 3. Architecture Tuning Tests
# ==============================================================================

def test_ann_tune_architecture(regression_dataset):
    """Verify multi-architecture search across Shallow, Deep, and Tanh configurations."""
    engine = ANNEngine()
    best_res, trials = engine.tune_architecture(
        dataframe=regression_dataset,
        target_column="target_metric",
    )

    assert isinstance(best_res, ANNTrainingResult)
    assert len(trials) == 4
    assert any(t["activation"] == "tanh" for t in trials)
    assert any("Dense(128" in t["architecture"] for t in trials)
    assert best_res.primary_metric_value > 0.4


# ==============================================================================
# 4. ANNAgent & Planner Integration Tests
# ==============================================================================

def test_ann_agent_execution(regression_dataset):
    """Verify ANNAgent runs and emits rich AgentResult with Evidence."""
    agent = ANNAgent()
    result = agent.run({
        "data": regression_dataset,
        "target": "target_metric",
        "layers": [64, 32],
        "epochs": 100,
    })

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.agent == "ANN Agent"
    assert "loss_curve" in result.output
    assert "architecture_summary" in result.output
    assert "comparison_with_ml" in result.output
    assert len(result.evidence) >= 1
    assert result.evidence[0].method == "ann_training_and_evaluation"


def test_planner_agent_ann_action(classification_dataset):
    """Verify PlannerAgent routing for 'ann' action."""
    planner = PlannerAgent(data=classification_dataset)
    result = planner.run_agent({"action": "ann", "target": "target_class", "epochs": 100})

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.validation is not None
    assert result.validation.passed is True
    assert "loss_curve" in result.output
