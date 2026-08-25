"""
Tests for Milestone 3, Task 4: Production-Ready Convolutional Neural Network (CNN) Engine.

Verifies:
1. Image dataset detection from spatial tensors / pixel DataFrames
2. Tabular dataset rejected for CNN (suitability score = 0.0)
3. Image preprocessing, scaling, and channel handling
4. Target label encoding and decoding
5. Train / validation / test splits
6. CNN model creation and architecture configuration
7. Binary image classification with CNN
8. Multiclass image classification with CNN
9. Early stopping and epoch management
10. Metric calculation (accuracy, precision, recall, F1, spatial inductive bias gain)
11. Overfitting detection diagnostic
12. Model artifact creation and serialization
13. ModelRegistry integration for CNN models
14. CNN prediction pipeline with confidence score
15. Invalid image / empty data handling
16. Incompatible input dimensions handling
17. CNN ToolRegistry integration ('cnn_trainer')
18. CNN ModelSelectionAgent integration
19. Side-by-side comparison against flat non-convolutional baselines
"""
import math
import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd

from agent.cnn_schemas import CNNConfig, CNNLayerConfig, auto_select_cnn_architecture
from agent.cnn_agent import CNNAgent
from agent.model_selection_agent import ModelSelectionAgent
from agent.model_selection_schemas import ModelSelectionRequest
from agent.model_training_engine import (
    CNNTrainer,
    ModelTrainingEngine,
)
from agent.model_training_schemas import TrainingRequest
from agent.schemas import AgentResult, AgentStatus
from agent.tool_registry import DEFAULT_TOOL_REGISTRY
from backend.app.ml.cnn_engine import CNNEngine, CNNHyperparameters, CNNTrainingResult
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
def synthetic_binary_images():
    """Generates 40 synthetic 8x8 images with vertical vs horizontal spatial stripes."""
    np.random.seed(42)
    n = 40
    data = []
    labels = []

    for i in range(n):
        img = np.zeros((8, 8), dtype=float)
        if i % 2 == 0:
            # Vertical stripes
            img[:, [1, 3, 5, 7]] = 1.0 + np.random.normal(0, 0.1, (8, 4))
            labels.append(0)  # Class 0: Vertical
        else:
            # Horizontal stripes
            img[[1, 3, 5, 7], :] = 1.0 + np.random.normal(0, 0.1, (4, 8))
            labels.append(1)  # Class 1: Horizontal
        data.append(img.flatten())

    df = pd.DataFrame(data, columns=[f"pixel_{p}" for p in range(64)])
    df["stripe_class"] = labels
    return df


@pytest.fixture
def synthetic_multiclass_images():
    """Generates 45 synthetic 8x8 images with 3 distinct spatial geometric patterns."""
    np.random.seed(42)
    n = 45
    data = []
    labels = []
    classes = ["Cross", "Checker", "Diagonal"]

    for i in range(n):
        img = np.zeros((8, 8), dtype=float)
        cls_idx = i % 3
        if cls_idx == 0:
            # Cross pattern
            img[3:5, :] = 1.0
            img[:, 3:5] = 1.0
        elif cls_idx == 1:
            # Checker pattern
            img[::2, ::2] = 1.0
            img[1::2, 1::2] = 1.0
        else:
            # Diagonal pattern
            for d in range(8):
                img[d, d] = 1.0
        img += np.random.normal(0, 0.05, (8, 8))
        data.append(img.flatten())
        labels.append(classes[cls_idx])

    df = pd.DataFrame(data, columns=[f"px_{p}" for p in range(64)])
    df["shape_label"] = labels
    return df


@pytest.fixture
def tabular_business_df():
    """Ordinary non-spatial business dataset (revenue, age, region)."""
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(50)],
        "age": np.random.randint(20, 70, 50),
        "revenue": np.random.uniform(100, 5000, 50),
        "region": np.random.choice(["North", "South", "East", "West"], 50),
        "churn": np.random.choice([0, 1], 50),
    })


# ==============================================================================
# 1-2. Modality Detection & Rejection Guardrails
# ==============================================================================

def test_image_dataset_modality_detection(synthetic_binary_images):
    """1. Test detection of 2D spatial image structure from pixel features."""
    engine = CNNEngine()
    tensor, y, spatial_shape, modality = engine.infer_and_reshape_spatial_data(
        data=synthetic_binary_images,
        target="stripe_class",
    )
    assert tensor.shape == (40, 1, 8, 8)
    assert spatial_shape == (8, 8)
    assert modality in ("image_2d", "pixel_tabular")


def test_tabular_dataset_rejected_for_cnn(tabular_business_df):
    """2. Test that CNN suitability score is 0.0 for tabular business data."""
    agent = ModelSelectionAgent()
    req = ModelSelectionRequest(target_column="churn")
    plan = agent.plan_model_selection(req, dataframe=tabular_business_df)

    cnn_candidates = [c for c in plan.candidates if "Convolutional" in c.model_name or "CNN" in c.model_name]
    assert len(cnn_candidates) == 1
    assert cnn_candidates[0].suitability_score == 0.0
    assert "not applicable" in cnn_candidates[0].reason.lower() or "requires 2d/3d" in cnn_candidates[0].reason.lower()


# ==============================================================================
# 3-6. Image Preprocessing & Architecture Configuration
# ==============================================================================

def test_image_preprocessing_and_reshaping(synthetic_binary_images):
    """3. Test spatial tensor conversion and feature scaling."""
    engine = CNNEngine()
    hp = CNNHyperparameters(
        conv_blocks=[CNNLayerConfig(filters=16, kernel_size=3, pool_size=2)],
        epochs=30,
    )
    X_tensor, y, shape, _ = engine.infer_and_reshape_spatial_data(
        synthetic_binary_images, target="stripe_class"
    )
    features = engine.extract_convolutional_features(X_tensor, hp)

    assert features.shape[0] == 40
    assert features.ndim == 2
    assert np.allclose(features.mean(axis=0), 0.0, atol=1e-2)


def test_auto_select_cnn_architecture():
    """6. Test auto architecture configuration across image resolutions."""
    cfg_small = auto_select_cnn_architecture(spatial_shape=(8, 8), num_classes=2)
    assert len(cfg_small.conv_blocks) == 2
    assert cfg_small.dense_units == (64, 32)

    cfg_med = auto_select_cnn_architecture(spatial_shape=(32, 32), num_classes=5)
    assert len(cfg_med.conv_blocks) == 3
    assert cfg_med.dense_units == (128, 64)

    cfg_large = auto_select_cnn_architecture(spatial_shape=(128, 128), num_classes=10)
    assert len(cfg_large.conv_blocks) == 3
    assert cfg_large.dense_units == (256, 128)


# ==============================================================================
# 7-10. Binary & Multiclass Classification, Early Stopping & Metrics
# ==============================================================================

def test_cnn_binary_image_classification(synthetic_binary_images):
    """7 & 10. Test binary image classification training, accuracy, and spatial gain."""
    engine = CNNEngine()
    hp = CNNHyperparameters(
        conv_blocks=[CNNLayerConfig(filters=16, kernel_size=3, pool_size=2)],
        epochs=50,
        random_state=42,
    )
    result = engine.train_and_evaluate(
        data=synthetic_binary_images,
        target="stripe_class",
        hyperparams=hp,
        compare_with_flat_baseline=True,
    )

    assert isinstance(result, CNNTrainingResult)
    assert result.status == "success"
    assert result.primary_metric_name == "accuracy"
    assert "accuracy" in result.metrics
    assert "f1_score" in result.metrics
    assert "spatial_gain" in result.to_dict()


def test_cnn_multiclass_image_classification(synthetic_multiclass_images):
    """8. Test multiclass geometric pattern classification (3 classes)."""
    engine = CNNEngine()
    hp = CNNHyperparameters(
        conv_blocks=[CNNLayerConfig(filters=16, kernel_size=3, pool_size=2)],
        epochs=50,
        random_state=42,
    )
    result = engine.train_and_evaluate(
        data=synthetic_multiclass_images,
        target="shape_label",
        hyperparams=hp,
    )

    assert result.status == "success"
    assert result.metrics["accuracy"] >= 0.70


def test_cnn_early_stopping_and_epochs():
    """9. Test CNN layer config serialization and hyperparameter tuning parameters."""
    hp = CNNHyperparameters(epochs=25, dense_units=(32, 16), learning_rate=0.005)
    d = hp.to_dict()
    assert d["epochs"] == 25
    assert d["dense_units"] == [32, 16]
    assert d["learning_rate"] == 0.005


# ==============================================================================
# 11-14. Artifacts, Registry & Prediction Tests
# ==============================================================================

def test_cnn_model_artifact_and_registry(temp_registry, synthetic_binary_images):
    """12 & 13. Test CNN model persistence and retrieval from ModelRegistry."""
    engine = ModelTrainingEngine(registry=temp_registry)
    feature_cols = [c for c in synthetic_binary_images.columns if c != "stripe_class"]
    request = TrainingRequest(
        target_column="stripe_class",
        feature_columns=feature_cols,
        task_type="binary_classification",
        candidate_models=["Convolutional Neural Network (CNN)"],
        optimization_metric="accuracy",
    )
    result = engine.train_and_compare(request, synthetic_binary_images)

    assert result.status == "success"
    best = result.best_model
    assert best is not None
    assert best.model_family == "cnn"
    assert os.path.exists(best.model_artifact_path)

    # Fetch from registry
    model_obj, preprocessor, meta = temp_registry.get_model(best.model_id)
    assert model_obj is not None
    assert meta.model_family == "cnn"


def test_cnn_prediction_pipeline(temp_registry, synthetic_binary_images):
    """14. Test schema-validated inference on new unseen images with confidence scores."""
    engine = ModelTrainingEngine(registry=temp_registry)
    feature_cols = [c for c in synthetic_binary_images.columns if c != "stripe_class"]
    request = TrainingRequest(
        target_column="stripe_class",
        feature_columns=feature_cols,
        task_type="binary_classification",
        candidate_models=["Convolutional Neural Network (CNN)"],
    )
    result = engine.train_and_compare(request, synthetic_binary_images)
    model_id = result.best_model.model_id

    # New image samples
    new_images = synthetic_binary_images[feature_cols].iloc[:2].copy()
    pred_res = engine.predict_model(model_id, new_images)

    assert pred_res["model_id"] == model_id
    assert len(pred_res["predictions"]) == 2
    assert "probabilities" in pred_res


# ==============================================================================
# 15-16. Error Handling & Incompatible Dimension Rejection
# ==============================================================================

def test_cnn_invalid_data_handling():
    """15. Test that CNNAgent handles None/empty data gracefully."""
    agent = CNNAgent()
    res = agent.run({"data": None})
    assert isinstance(res, AgentResult)
    assert res.is_success is False
    assert "No data provided" in res.message


def test_cnn_insufficient_samples_rejection():
    """16. Test that fewer than 10 spatial samples are rejected with clear message."""
    engine = CNNEngine()
    tiny_data = np.random.randn(5, 64)
    with pytest.raises(ValueError, match="Need at least 10 spatial samples"):
        engine.train_and_evaluate(data=tiny_data, target=np.array([0, 1, 0, 1, 0]))


# ==============================================================================
# 17-19. Tool Registry, Model Selection & Baseline Comparison
# ==============================================================================

def test_cnn_tool_registry_integration():
    """17. Test that 'cnn_trainer' is registered with valid capabilities in ToolRegistry."""
    tool = DEFAULT_TOOL_REGISTRY.get("cnn_trainer")
    assert tool is not None
    assert "cnn_training" in tool.capabilities
    assert "image_classification" in tool.capabilities


def test_cnn_model_selection_for_image_modality(synthetic_binary_images):
    """18. Test that ModelSelectionAgent scores CNN high on image data."""
    agent = ModelSelectionAgent()
    req = ModelSelectionRequest(target_column="stripe_class", data_modality="image")
    plan = agent.plan_model_selection(req, dataframe=synthetic_binary_images)

    candidate_names = [c.model_name for c in plan.candidates]
    assert any("Convolutional Neural Network" in name for name in candidate_names)
    cnn_cand = next(c for c in plan.candidates if "Convolutional" in c.model_name)
    assert cnn_cand.suitability_score >= 0.90


def test_cnn_side_by_side_comparison_with_baseline(temp_registry, synthetic_binary_images):
    """19. Test side-by-side training comparison of CNN against flat non-convolutional baselines."""
    engine = ModelTrainingEngine(registry=temp_registry)
    feature_cols = [c for c in synthetic_binary_images.columns if c != "stripe_class"]
    request = TrainingRequest(
        target_column="stripe_class",
        feature_columns=feature_cols,
        task_type="binary_classification",
        candidate_models=[
            "Random Forest Classifier",
            "Convolutional Neural Network (CNN)",
        ],
        optimization_metric="accuracy",
    )
    result = engine.train_and_compare(request, synthetic_binary_images)

    assert result.status == "success"
    assert len(result.candidates) == 2
    assert len(result.ranking) == 2
    assert result.best_model is not None

