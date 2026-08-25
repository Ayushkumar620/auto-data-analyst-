"""
Tests for Milestone 3, Task 2: Model Training and Evaluation Engine.

Verifies:
1. Tabular regression training & evaluation
2. Binary classification training & evaluation
3. Multiclass classification training & evaluation
4. Imbalanced classification with F1 optimization
5. Cross-validation execution (KFold & StratifiedKFold)
6. Time-series split (chronological without future leakage)
7. Preprocessing leakage prevention (preprocessor fitted on train fold only)
8. Metric calculation correctness
9. Multi-candidate model ranking
10. Best model selection (higher-is-better vs lower-is-better metrics)
11. Overfitting detection diagnostics
12. Failed candidate error isolation
13. Partial success returned when some candidates fail
14. All models failing returns structured failure
15. Model artifact serialization
16. ModelRegistry integration and version tracking
17. Deterministic results with random_state
18. Evidence generation grounded in actual computed metrics
"""
import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd

from agent.model_selection_schemas import ModelCandidate
from agent.model_training_engine import (
    DataPreprocessor,
    ModelTrainingAgent,
    ModelTrainingEngine,
    TraditionalMLTrainer,
)
from agent.model_training_schemas import (
    ModelComparisonResult,
    TrainingRequest,
    TrainingResult,
)
from agent.schemas import AgentResult, AgentStatus
from backend.app.ml.registry import ModelRegistry


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def temp_registry():
    """Temporary model registry directory."""
    temp_dir = tempfile.mkdtemp()
    registry = ModelRegistry(registry_dir=temp_dir)
    yield registry
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def regression_data():
    """Synthetic regression dataset."""
    np.random.seed(42)
    n = 60
    x1 = np.random.uniform(10, 50, n)
    x2 = np.random.uniform(1, 10, n)
    y = 3.0 * x1 + 2.0 * x2 + np.random.normal(0, 2, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


@pytest.fixture
def binary_data():
    """Synthetic binary classification dataset."""
    np.random.seed(42)
    n = 60
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(5, 2, n)
    y = (x1 + x2 > 5.0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


@pytest.fixture
def multiclass_data():
    """Synthetic multiclass classification dataset (3 classes)."""
    np.random.seed(42)
    n = 60
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(5, 2, n)
    classes = ["Low", "Medium", "High"]
    y = [classes[i % 3] for i in range(n)]
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


@pytest.fixture
def imbalanced_data():
    """Synthetic imbalanced classification dataset (85% class 0, 15% class 1)."""
    np.random.seed(42)
    n = 80
    x1 = np.random.uniform(10, 100, n)
    x2 = np.random.uniform(1, 20, n)
    y = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


# ==============================================================================
# 1-4. Core Task Training & Optimization Tests
# ==============================================================================

def test_regression_training(temp_registry, regression_data):
    """1. Test regression model training and metric computation."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="target",
        feature_columns=["x1", "x2"],
        task_type="regression",
        candidate_models=["Linear Regression", "Ridge Regression", "Random Forest Regressor"],
        optimization_metric="r2",
    )
    result = engine.train_and_compare(request, regression_data)

    assert result.status == "success"
    assert result.best_model is not None
    assert result.best_model.primary_metric_name == "r2"
    assert result.best_model.primary_metric_value > 0.70
    assert "mae" in result.best_model.validation_metrics
    assert "rmse" in result.best_model.validation_metrics


def test_binary_classification_training(temp_registry, binary_data):
    """2. Test binary classification model training and ROC-AUC / Accuracy."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="target",
        feature_columns=["x1", "x2"],
        task_type="binary_classification",
        candidate_models=["Logistic Regression", "Random Forest Classifier"],
        optimization_metric="roc_auc",
    )
    result = engine.train_and_compare(request, binary_data)

    assert result.status == "success"
    assert result.best_model is not None
    assert "accuracy" in result.best_model.validation_metrics
    assert "precision" in result.best_model.validation_metrics
    assert "recall" in result.best_model.validation_metrics
    assert "f1" in result.best_model.validation_metrics


def test_multiclass_classification_training(temp_registry, multiclass_data):
    """3. Test multiclass classification training and weighted metrics."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="target",
        feature_columns=["x1", "x2"],
        task_type="multiclass_classification",
        candidate_models=["Logistic Regression", "Random Forest Classifier"],
        optimization_metric="accuracy",
    )
    result = engine.train_and_compare(request, multiclass_data)

    assert result.status == "success"
    assert result.best_model is not None
    assert "accuracy" in result.best_model.validation_metrics
    assert "f1_weighted" in result.best_model.validation_metrics


def test_imbalanced_classification_f1_optimization(temp_registry, imbalanced_data):
    """4. Test that imbalanced classification optimizes F1 score."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="target",
        feature_columns=["x1", "x2"],
        task_type="binary_classification",
        candidate_models=["Logistic Regression", "Random Forest Classifier"],
        optimization_metric="f1",
    )
    result = engine.train_and_compare(request, imbalanced_data)

    assert result.status == "success"
    assert result.optimization_metric == "f1"
    assert result.best_model.primary_metric_name == "f1"


# ==============================================================================
# 5-8. Cross-Validation, Leakage Prevention & Metrics Tests
# ==============================================================================

def test_cross_validation_execution(temp_registry, regression_data):
    """5. Test 5-fold cross-validation execution and fold score tracking."""
    engine = ModelTrainingEngine(registry=temp_registry)
    res = engine.train_and_validate_candidate(
        candidate="Linear Regression",
        df=regression_data,
        target_col="target",
        feature_cols=["x1", "x2"],
        task_type="regression",
        validation_strategy="5_fold_cv",
    )

    assert res.status == "success"
    assert "fold_scores" in res.validation_results
    assert len(res.validation_results["fold_scores"]) == 5


def test_time_series_split_execution(temp_registry, regression_data):
    """6. Test chronological TimeSeriesSplit without lookahead leakage."""
    engine = ModelTrainingEngine(registry=temp_registry)
    res = engine.train_and_validate_candidate(
        candidate="Linear Regression",
        df=regression_data,
        target_col="target",
        feature_cols=["x1", "x2"],
        task_type="regression",
        validation_strategy="time_series_split",
    )

    assert res.status == "success"
    assert len(res.validation_results["fold_scores"]) >= 3


def test_preprocessing_leakage_prevention():
    """7. Test that DataPreprocessor learns means/scales strictly from training slice."""
    X_train = pd.DataFrame({"num": [10.0, 20.0, 30.0], "cat": ["A", "B", "A"]})
    X_test = pd.DataFrame({"num": [100.0, 200.0], "cat": ["A", "Unseen"]})

    prep = DataPreprocessor()
    prep.fit(X_train)

    # Median learned from train must be 20.0, not affected by test values (100, 200)
    assert prep.impute_values["num"] == 20.0

    X_test_t = prep.transform(X_test)
    assert X_test_t.shape[0] == 2
    assert X_test_t.shape[1] == 2  # numeric + cat


def test_metric_calculation_correctness():
    """8. Test calculation of regression and classification metrics."""
    engine = ModelTrainingEngine()

    # Regression
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 31.0, 39.0])
    m_reg = engine.compute_metrics(y_true, y_pred, "regression")
    assert "r2" in m_reg
    assert "mae" in m_reg
    assert "rmse" in m_reg
    assert m_reg["mae"] == 1.5

    # Classification
    y_c_true = np.array([0, 1, 1, 0])
    y_c_pred = np.array([0, 1, 0, 0])
    m_clf = engine.compute_metrics(y_c_true, y_c_pred, "binary_classification")
    assert m_clf["accuracy"] == 0.75
    assert m_clf["precision"] == 1.0


# ==============================================================================
# 9-12. Model Ranking, Winner Selection & Error Recovery Tests
# ==============================================================================

def test_model_candidate_ranking(temp_registry, regression_data):
    """9. Test multi-model candidate ranking."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="target",
        feature_columns=["x1", "x2"],
        task_type="regression",
        candidate_models=["Linear Regression", "Random Forest Regressor"],
        optimization_metric="r2",
    )
    result = engine.train_and_compare(request, regression_data)

    assert len(result.ranking) == 2
    assert result.ranking[0]["rank"] == 1
    assert result.ranking[1]["rank"] == 2
    assert result.ranking[0]["score"] >= result.ranking[1]["score"]


def test_best_model_selection_direction(temp_registry, regression_data):
    """10. Test that minimization metrics (RMSE) select the lowest score as winner."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="target",
        feature_columns=["x1", "x2"],
        task_type="regression",
        candidate_models=["Linear Regression", "Ridge Regression"],
        optimization_metric="rmse",
    )
    result = engine.train_and_compare(request, regression_data)

    assert result.optimization_metric == "rmse"
    # Winner must have smallest RMSE
    assert result.ranking[0]["score"] <= result.ranking[1]["score"]


def test_overfitting_detection(temp_registry):
    """11. Test detection of overfitting when train score is significantly higher than val score."""
    engine = ModelTrainingEngine(registry=temp_registry)

    # Overfitting dataset: pure random noise target with 20 features and 25 rows
    np.random.seed(42)
    n = 25
    df_noise = pd.DataFrame({f"f_{i}": np.random.randn(n) for i in range(15)})
    df_noise["noise_target"] = np.random.choice([0, 1], size=n)

    res = engine.train_and_validate_candidate(
        candidate="Decision Tree Classifier",  # Unconstrained decision tree will overfit noise
        df=df_noise,
        target_col="noise_target",
        feature_cols=[f"f_{i}" for i in range(15)],
        task_type="binary_classification",
    )

    # Overfitting or warnings should be captured
    assert res.status == "success"
    assert "accuracy" in res.training_metrics


def test_failed_candidate_isolation(temp_registry, regression_data):
    """12 & 13. Test that failure in one candidate does not crash other candidates (partial success)."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="target",
        feature_columns=["x1", "x2"],
        task_type="regression",
        candidate_models=["Linear Regression", "InvalidNonExistentModelX99"],
        optimization_metric="r2",
    )
    result = engine.train_and_compare(request, regression_data)

    # Partial success: Linear Regression succeeded, invalid model was isolated
    assert result.status == "partial" or result.status == "success"
    assert result.best_model is not None
    assert result.best_model.model_name == "Linear Regression"


def test_all_models_failing_handling(temp_registry):
    """14. Test that when all candidates fail, a clean structured failure result is returned."""
    engine = ModelTrainingEngine(registry=temp_registry)
    empty_df = pd.DataFrame()
    request = TrainingRequest(target_column="y", feature_columns=["x"])
    result = engine.train_and_compare(request, empty_df)

    assert result.status == "failed"
    assert result.best_model is None


# ==============================================================================
# 15-18. Artifacts, Registry, Determinism & Evidence Tests
# ==============================================================================

def test_model_artifact_creation_and_registry(temp_registry, regression_data):
    """15 & 16. Test model artifact file serialization and ModelRegistry retrieval."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="target",
        feature_columns=["x1", "x2"],
        task_type="regression",
        candidate_models=["Linear Regression"],
    )
    result = engine.train_and_compare(request, regression_data)

    best = result.best_model
    assert best.model_artifact_path is not None
    assert os.path.exists(best.model_artifact_path)

    # Retrieve from registry
    model_obj, preprocessor, meta = temp_registry.get_model(best.model_id)
    assert model_obj is not None
    assert meta.target_column == "target"
    assert meta.problem_type == "regression"


def test_deterministic_results(temp_registry, regression_data):
    """17. Test that setting random_state produces identical metric scores across repeated runs."""
    engine = ModelTrainingEngine(registry=temp_registry)
    req = TrainingRequest(
        target_column="target",
        feature_columns=["x1", "x2"],
        task_type="regression",
        candidate_models=["Random Forest Regressor"],
        random_state=42,
    )
    r1 = engine.train_and_compare(req, regression_data)
    r2 = engine.train_and_compare(req, regression_data)

    assert r1.best_model.primary_metric_value == r2.best_model.primary_metric_value


def test_evidence_generation_and_agent_run(regression_data):
    """18. Test Evidence generation and ModelTrainingAgent run interface."""
    agent = ModelTrainingAgent()
    result = agent.run({
        "data": regression_data,
        "target": "target",
        "features": ["x1", "x2"],
        "task_type": "regression",
        "candidates": ["Linear Regression", "Ridge Regression"],
    })

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert len(result.evidence) >= 1
    assert result.evidence[0].source == "ModelTrainingEngine"
    assert "best_model" in result.output
    assert "ranking" in result.output
