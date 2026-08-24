"""Comprehensive test suite for Phase 6: Convolutional Neural Network (CNN) Engine."""
import numpy as np
import pandas as pd
import pytest

from agent.schemas import AgentResult, AgentStatus, ClaimType
from agent.cnn_agent import CNNAgent
from agent.planner import PlannerAgent
from backend.app.ml.cnn_engine import (
    CNNEngine,
    CNNHyperparameters,
    CNNLayerConfig,
    CNNTrainingResult,
)
from backend.app.ml.model_selection import ProblemType


# ==============================================================================
# Fixtures: Synthetic Spatial, Image, and Signal Data
# ==============================================================================

@pytest.fixture
def synthetic_image_df():
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
    df["pattern_type"] = labels
    return df


@pytest.fixture
def synthetic_sensor_signals():
    """Generates 30 synthetic 1D frequency signals for spectrogram testing."""
    np.random.seed(42)
    n = 30
    time_pts = np.linspace(0, 1.0, 128)
    signals = []
    labels = []

    for i in range(n):
        freq = 10.0 if i % 2 == 0 else 35.0
        sig = np.sin(2 * np.pi * freq * time_pts) + np.random.normal(0, 0.2, 128)
        signals.append(sig)
        labels.append(0 if freq == 10.0 else 1)

    return np.array(signals, dtype=np.float32), np.array(labels, dtype=int)


# ==============================================================================
# 1. Hyperparameters & Preprocessing Tests
# ==============================================================================

def test_cnn_hyperparameters_serialization():
    """Verify CNNHyperparameters and layer configuration."""
    blocks = [CNNLayerConfig(filters=16, kernel_size=3, pool_size=2)]
    hp = CNNHyperparameters(conv_blocks=blocks, dense_units=(64, 32), epochs=50)
    d = hp.to_dict()

    assert len(d["conv_blocks"]) == 1
    assert d["conv_blocks"][0]["filters"] == 16
    assert d["dense_units"] == [64, 32]
    assert d["epochs"] == 50


def test_spatial_data_ingestion_and_reshaping(synthetic_image_df):
    """Verify ingestion of flattened pixel DataFrames into 4D tensors (N, 1, H, W)."""
    engine = CNNEngine()
    tensor, y, spatial_shape, modality = engine.infer_and_reshape_spatial_data(
        data=synthetic_image_df,
        target="pattern_type",
    )

    assert tensor.shape == (40, 1, 8, 8)
    assert spatial_shape == (8, 8)
    assert len(y) == 40
    assert modality in ("image_2d", "pixel_tabular")


def test_signal_to_spectrogram_transformation(synthetic_sensor_signals):
    """Verify conversion of 1D signals to 2D time-frequency spectrogram heatmaps."""
    signals, labels = synthetic_sensor_signals
    engine = CNNEngine()
    spec_tensor, (freq_bins, time_bins) = engine.signal_to_spectrogram(signals, fs=100.0, nperseg=32)

    assert spec_tensor.shape[0] == 30
    assert spec_tensor.shape[1] == 1
    assert freq_bins > 0
    assert time_bins > 0


# ==============================================================================
# 2. CNN Training, Feature Extraction & Baseline Comparison Tests
# ==============================================================================

def test_cnn_train_and_evaluate_image_classification(synthetic_image_df):
    """Verify end-to-end CNN training, evaluation metrics, and baseline comparison."""
    engine = CNNEngine()
    hp = CNNHyperparameters(
        conv_blocks=[CNNLayerConfig(filters=8, kernel_size=3, pool_size=2)],
        dense_units=(32, 16),
        epochs=80,
        random_state=42,
    )
    result = engine.train_and_evaluate(
        data=synthetic_image_df,
        target="pattern_type",
        spatial_shape=(8, 8),
        hyperparams=hp,
        compare_with_flat_baseline=True,
    )

    assert isinstance(result, CNNTrainingResult)
    assert result.status == "success"
    assert result.problem_type == ProblemType.BINARY_CLASSIFICATION
    assert "accuracy" in result.metrics
    assert "f1_score" in result.metrics
    assert result.primary_metric_value >= 0.70

    # Verify architecture summary
    assert "Conv2D(8@3x3)" in result.architecture_summary
    assert "MaxPool(2x2)" in result.architecture_summary
    assert "Flatten()" in result.architecture_summary

    # Verify baseline comparison
    assert "flat_baseline_model" in result.comparison_with_flat_baseline
    assert "cnn_score" in result.comparison_with_flat_baseline


def test_cnn_train_on_signal_spectrograms(synthetic_sensor_signals):
    """Verify CNN training on 2D spectrogram signal representations."""
    signals, labels = synthetic_sensor_signals
    engine = CNNEngine()
    spec_tensor, spec_shape = engine.signal_to_spectrogram(signals, fs=100.0, nperseg=32)

    hp = CNNHyperparameters(
        conv_blocks=[CNNLayerConfig(filters=8, kernel_size=3, pool_size=2)],
        dense_units=(32, 16),
        epochs=80,
        random_state=42,
    )
    result = engine.train_and_evaluate(
        data=spec_tensor,
        target=labels,
        spatial_shape=spec_shape,
        hyperparams=hp,
        compare_with_flat_baseline=True,
    )

    assert isinstance(result, CNNTrainingResult)
    assert result.status == "success"
    assert result.primary_metric_value >= 0.60


# ==============================================================================
# 3. CNNAgent & Planner Integration Tests
# ==============================================================================

def test_cnn_agent_execution_image_df(synthetic_image_df):
    """Verify CNNAgent execution and validated AgentResult output."""
    agent = CNNAgent()
    result = agent.run({
        "data": synthetic_image_df,
        "target": "pattern_type",
        "spatial_shape": (8, 8),
        "epochs": 60,
    })

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.agent == "CNN Agent"
    assert "architecture_summary" in result.output
    assert "comparison_with_flat_baseline" in result.output
    assert len(result.evidence) >= 1
    assert result.evidence[0].method == "cnn_convolutional_training"


def test_planner_agent_cnn_action(synthetic_image_df):
    """Verify PlannerAgent routing for 'cnn' action."""
    planner = PlannerAgent(data=synthetic_image_df)
    result = planner.run_agent({
        "action": "cnn",
        "target": "pattern_type",
        "spatial_shape": (8, 8),
        "epochs": 60,
    })

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.validation is not None
    assert result.validation.passed is True
    assert "architecture_summary" in result.output

