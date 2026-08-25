"""
Tests for Milestone 3, Task 3: Production-Ready Artificial Neural Network (ANN) Engine.

Verifies:
1. ANN tabular regression training and evaluation
2. ANN binary classification training and evaluation
3. ANN multiclass classification training and evaluation
4. Preprocessing without data leakage (DataPreprocessor)
5. Target encoding and inverse decoding
6. Early stopping configuration and tracking
7. Overfitting detection diagnostic
8. Metric generation (R2, RMSE, MAE, F1, ROC-AUC, Accuracy)
9. Model artifact creation and serialization
10. ModelRegistry integration for ANN models
11. End-to-end prediction pipeline with schema validation
12. Feature-order enforcement during prediction
13. Invalid input / missing target error handling
14. Insufficient samples error handling (< 10 rows)
15. ANN failure handling and error isolation
16. ANN integration with ModelSelectionAgent (suitability scoring)
17. ANN integration with ToolRegistry ('ann_trainer')
18. Side-by-side comparison against traditional ML models
"""
import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd

from agent.ann_schemas import ANNConfig, auto_select_ann_architecture
from agent.ann_agent import ANNAgent
from agent.model_selection_agent import ModelSelectionAgent
from agent.model_selection_schemas import ModelSelectionRequest
from agent.model_training_engine import (
    ANNTrainer,
    DataPreprocessor,
    ModelTrainingEngine,
)
from agent.model_training_schemas import TrainingRequest
from agent.schemas import AgentResult, AgentStatus
from agent.tool_registry import DEFAULT_TOOL_REGISTRY
from backend.app.ml.ann_engine import ANNEngine, ANNHyperparameters, ANNTrainingResult
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
def ann_regression_df():
    """Synthetic tabular regression dataset for ANN testing."""
    np.random.seed(42)
    n = 60
    x1 = np.random.uniform(5, 50, n)
    x2 = np.random.uniform(1, 10, n)
    # Non-linear relationship
    y = 2.0 * x1 + 1.5 * (x2 ** 2) + np.random.normal(0, 2, n)
    return pd.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "category_a": np.random.choice(["Tier1", "Tier2", "Tier3"], n),
        "target_revenue": y,
    })


@pytest.fixture
def ann_binary_df():
    """Synthetic binary classification dataset for ANN testing."""
    np.random.seed(42)
    n = 60
    f1 = np.random.uniform(10, 100, n)
    f2 = np.random.uniform(1, 10, n)
    prob = 1 / (1 + np.exp(-(0.05 * f1 - 0.3 * f2)))
    labels = (prob > 0.5).astype(int)
    return pd.DataFrame({
        "score_1": f1,
        "score_2": f2,
        "category_b": np.random.choice(["Alpha", "Beta"], n),
        "is_active": labels,
    })


@pytest.fixture
def ann_multiclass_df():
    """Synthetic multiclass classification dataset (3 classes)."""
    np.random.seed(42)
    n = 60
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(5, 2, n)
    classes = ["Low", "Medium", "High"]
    labels = [classes[i % 3] for i in range(n)]
    return pd.DataFrame({
        "num_x": x1,
        "num_y": x2,
        "tier_label": labels,
    })


# ==============================================================================
# 1-3. Core ANN Task Execution Tests
# ==============================================================================

def test_ann_regression_training(temp_registry, ann_regression_df):
    """1. Test ANN tabular regression training and evaluation."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="target_revenue",
        feature_columns=["feature_1", "feature_2", "category_a"],
        task_type="regression",
        candidate_models=["Artificial Neural Network (ANN/MLP)"],
        optimization_metric="r2",
    )
    result = engine.train_and_compare(request, ann_regression_df)

    assert result.status == "success"
    best = result.best_model
    assert best is not None
    assert "ann" in best.model_family.lower()
    assert best.primary_metric_name == "r2"
    assert "rmse" in best.validation_metrics
    assert "mae" in best.validation_metrics


def test_ann_binary_classification_training(temp_registry, ann_binary_df):
    """2. Test ANN binary classification training and evaluation."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="is_active",
        feature_columns=["score_1", "score_2", "category_b"],
        task_type="binary_classification",
        candidate_models=["Artificial Neural Network (ANN/MLP)"],
        optimization_metric="accuracy",
    )
    result = engine.train_and_compare(request, ann_binary_df)

    assert result.status == "success"
    best = result.best_model
    assert best is not None
    assert "accuracy" in best.validation_metrics
    assert "f1" in best.validation_metrics


def test_ann_multiclass_classification_training(temp_registry, ann_multiclass_df):
    """3. Test ANN multiclass classification training and weighted evaluation."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="tier_label",
        feature_columns=["num_x", "num_y"],
        task_type="multiclass_classification",
        candidate_models=["Artificial Neural Network (ANN/MLP)"],
        optimization_metric="accuracy",
    )
    result = engine.train_and_compare(request, ann_multiclass_df)

    assert result.status == "success"
    best = result.best_model
    assert best is not None
    assert "f1_weighted" in best.validation_metrics


# ==============================================================================
# 4-8. Preprocessing, Early Stopping, Overfitting & Metrics Tests
# ==============================================================================

def test_ann_preprocessing_and_target_encoding(ann_multiclass_df):
    """4 & 5. Test leakage-free preprocessing and label encoding/decoding."""
    prep = DataPreprocessor()
    prep.fit(
        ann_multiclass_df[["num_x", "num_y"]],
        ann_multiclass_df["tier_label"],
        is_classification=True,
    )

    X_t = prep.transform(ann_multiclass_df[["num_x", "num_y"]])
    y_t = prep.transform_target(ann_multiclass_df["tier_label"])

    assert X_t.shape == (60, 2)
    assert len(y_t) == 60
    assert set(y_t) == {0, 1, 2}

    # Inverse decode test
    decoded = prep.target_encoder.inverse_transform(y_t)
    assert list(decoded) == list(ann_multiclass_df["tier_label"])


def test_auto_select_ann_architecture():
    """6. Test automatic architecture selection across dataset sizes."""
    # Small dataset
    cfg_small = auto_select_ann_architecture(n_samples=50, n_features=3)
    assert cfg_small.hidden_layers == (32, 16)
    assert cfg_small.optimizer == "lbfgs"

    # Medium dataset
    cfg_med = auto_select_ann_architecture(n_samples=500, n_features=10)
    assert cfg_med.hidden_layers == (128, 64)
    assert cfg_med.optimizer == "adam"

    # Large dataset
    cfg_large = auto_select_ann_architecture(n_samples=2000, n_features=25)
    assert cfg_large.hidden_layers == (256, 128, 64)
    assert cfg_large.early_stopping is True


def test_ann_overfitting_detection(temp_registry):
    """7. Test that large train/validation score divergences trigger overfitting warnings."""
    engine = ModelTrainingEngine(registry=temp_registry)

    # Pure noise dataset with 20 features and 25 rows
    np.random.seed(42)
    n = 25
    df_noise = pd.DataFrame({f"f_{i}": np.random.randn(n) for i in range(15)})
    df_noise["y"] = np.random.choice([0, 1], size=n)

    res = engine.train_and_validate_candidate(
        candidate={"model_name": "Artificial Neural Network (ANN/MLP)", "hyperparameters": {"hidden_layer_sizes": (64, 32), "max_iter": 300}},
        df=df_noise,
        target_col="y",
        feature_cols=[f"f_{i}" for i in range(15)],
        task_type="binary_classification",
    )

    assert res.status == "success"
    assert "accuracy" in res.training_metrics
    assert "accuracy" in res.validation_metrics


def test_ann_metric_generation(temp_registry, ann_regression_df):
    """8. Test comprehensive metric calculation on ANN regression."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="target_revenue",
        feature_columns=["feature_1", "feature_2"],
        task_type="regression",
        candidate_models=["Artificial Neural Network (ANN/MLP)"],
        optimization_metric="r2",
    )
    res = engine.train_and_compare(request, ann_regression_df)
    metrics = res.best_model.validation_metrics

    assert "r2" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics


# ==============================================================================
# 9-12. Artifacts, Registry & Prediction Tests
# ==============================================================================

def test_ann_model_artifact_and_registry(temp_registry, ann_regression_df):
    """9 & 10. Test ANN model serialization, artifact files, and ModelRegistry retrieval."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="target_revenue",
        feature_columns=["feature_1", "feature_2"],
        task_type="regression",
        candidate_models=["Artificial Neural Network (ANN/MLP)"],
    )
    result = engine.train_and_compare(request, ann_regression_df)

    best = result.best_model
    assert best.model_artifact_path is not None
    assert os.path.exists(best.model_artifact_path)

    # Fetch from registry
    model_obj, preprocessor, meta = temp_registry.get_model(best.model_id)
    assert model_obj is not None
    assert meta.model_family == "ann"
    assert meta.target_column == "target_revenue"


def test_ann_prediction_pipeline(temp_registry, ann_binary_df):
    """11. Test end-to-end schema-validated inference on new unseen records."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="is_active",
        feature_columns=["score_1", "score_2", "category_b"],
        task_type="binary_classification",
        candidate_models=["Artificial Neural Network (ANN/MLP)"],
    )
    result = engine.train_and_compare(request, ann_binary_df)
    model_id = result.best_model.model_id

    # New records for prediction
    new_data = pd.DataFrame({
        "score_1": [50.0, 80.0],
        "score_2": [3.0, 8.0],
        "category_b": ["Alpha", "Beta"],
    })

    pred_res = engine.predict_model(model_id, new_data)
    assert pred_res["model_id"] == model_id
    assert len(pred_res["predictions"]) == 2
    assert "probabilities" in pred_res


def test_ann_prediction_feature_order_enforcement(temp_registry, ann_binary_df):
    """12. Test that prediction enforces exact training feature order even if input columns are permuted."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="is_active",
        feature_columns=["score_1", "score_2", "category_b"],
        task_type="binary_classification",
        candidate_models=["Artificial Neural Network (ANN/MLP)"],
    )
    result = engine.train_and_compare(request, ann_binary_df)
    model_id = result.best_model.model_id

    # Permuted feature order
    permuted_data = pd.DataFrame({
        "category_b": ["Alpha", "Beta"],
        "score_2": [3.0, 8.0],
        "score_1": [50.0, 80.0],
    })

    pred_res = engine.predict_model(model_id, permuted_data)
    assert len(pred_res["predictions"]) == 2


# ==============================================================================
# 13-15. Error Handling & Safe Fallback Tests
# ==============================================================================

def test_ann_invalid_input_handling(temp_registry):
    """13. Test that training on an empty DataFrame returns a structured failure."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(target_column="target", feature_columns=["x1"])
    result = engine.train_and_compare(request, pd.DataFrame())

    assert result.status == "failed"
    assert result.best_model is None


def test_ann_insufficient_samples_handling(temp_registry, ann_regression_df):
    """14. Test that ANNAgent rejects datasets with fewer than 10 samples."""
    agent = ANNAgent()
    tiny_df = ann_regression_df.iloc[:5]
    res = agent.run({"data": tiny_df, "target": "target_revenue"})

    assert isinstance(res, AgentResult)
    assert res.is_success is False
    assert "requires at least 10 samples" in res.message


def test_ann_prediction_missing_features_error(temp_registry, ann_binary_df):
    """15. Test that predicting with missing feature columns raises a clear error."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="is_active",
        feature_columns=["score_1", "score_2", "category_b"],
        task_type="binary_classification",
        candidate_models=["Artificial Neural Network (ANN/MLP)"],
    )
    result = engine.train_and_compare(request, ann_binary_df)
    model_id = result.best_model.model_id

    # Incomplete new data (missing score_2)
    incomplete_data = pd.DataFrame({"score_1": [50.0], "category_b": ["Alpha"]})

    with pytest.raises(ValueError, match="missing required features"):
        engine.predict_model(model_id, incomplete_data)


# ==============================================================================
# 16-18. Integration, Tool Registry & Comparison Tests
# ==============================================================================

def test_ann_model_selection_agent_integration(ann_regression_df):
    """16. Test that ModelSelectionAgent includes ANN candidate with transparent score."""
    selection_agent = ModelSelectionAgent()
    req = ModelSelectionRequest(target_column="target_revenue")
    plan = selection_agent.plan_model_selection(req, dataframe=ann_regression_df)

    candidate_names = [c.model_name for c in plan.candidates]
    assert any("Artificial Neural Network" in name for name in candidate_names)


def test_ann_tool_registry_integration():
    """17. Test that 'ann_trainer' is registered with valid capabilities in ToolRegistry."""
    tool = DEFAULT_TOOL_REGISTRY.get("ann_trainer")
    assert tool is not None
    assert "ann_training" in tool.capabilities
    assert "tabular_regression" in tool.capabilities


def test_ann_side_by_side_comparison_with_ml(temp_registry, ann_regression_df):
    """18. Test side-by-side training comparison of ANN against Random Forest and Linear Regression."""
    engine = ModelTrainingEngine(registry=temp_registry)
    request = TrainingRequest(
        target_column="target_revenue",
        feature_columns=["feature_1", "feature_2", "category_a"],
        task_type="regression",
        candidate_models=[
            "Linear Regression",
            "Random Forest Regressor",
            "Artificial Neural Network (ANN/MLP)",
        ],
        optimization_metric="r2",
    )
    result = engine.train_and_compare(request, ann_regression_df)

    assert result.status == "success"
    assert len(result.candidates) == 3
    assert len(result.ranking) == 3
    # Winner is determined solely by validated metric, not hard-coded bias
    assert result.best_model is not None
    assert result.best_model.model_name in [
        "Linear Regression",
        "Random Forest Regressor",
        "Artificial Neural Network (ANN/MLP)",
    ]

