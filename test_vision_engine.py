"""Tests for Multi-Modal Computer Vision Engine & Spatial Feature Extractor.

Verifies:
1. ComputerVisionFeatureEngine feature extraction on 2D and 3D image arrays
2. Spatial gradient filtering (Sobel, Laplacian) and grid pooling
3. Color moments, brightness, contrast, and blur detection
4. Batch transformation into structured tabular DataFrames (extract_batch_to_dataframe)
5. Seamless training with MLModelComparisonEngine and ANNEngine on extracted image features
"""
import numpy as np
import pandas as pd
import pytest

from backend.app.core.vision_engine import (
    ComputerVisionFeatureEngine,
    ImageFeatureVector,
    VisionExtractionReport,
    global_vision_engine,
)
from backend.app.ml.model_selection import MLModelComparisonEngine, ModelComparisonReport
from backend.app.ml.ann_engine import ANNEngine


@pytest.fixture
def synthetic_image_batch():
    """Create synthetic 2D/3D images belonging to two distinct visual classes (Circles vs Stripes)."""
    np.random.seed(42)
    images = []
    labels = []

    # Class 0: Vertical Stripes
    for i in range(20):
        img = np.zeros((32, 32, 3), dtype=np.float32)
        img[:, ::4, :] = 1.0  # Vertical white stripes
        img += np.random.normal(0, 0.05, img.shape)
        images.append(np.clip(img, 0.0, 1.0))
        labels.append("stripes")

    # Class 1: Solid Background with Center Square
    for i in range(20):
        img = np.full((32, 32, 3), 0.2, dtype=np.float32)
        img[10:22, 10:22, :] = 0.9  # Bright center square
        img += np.random.normal(0, 0.05, img.shape)
        images.append(np.clip(img, 0.0, 1.0))
        labels.append("square")

    return images, labels


def test_single_image_feature_extraction():
    """Verify single image extraction for quality, color, and spatial gradients."""
    engine = ComputerVisionFeatureEngine(target_size=(16, 16))
    img = np.random.uniform(0.0, 1.0, (28, 28, 3)).astype(np.float32)

    vec: ImageFeatureVector = engine.extract_features_from_array(img, image_id="test_01", label="sample")

    assert isinstance(vec, ImageFeatureVector)
    assert vec.image_id == "test_01"
    assert vec.width == 28
    assert vec.height == 28
    assert vec.channels == 3
    assert 0.0 <= vec.brightness <= 1.0
    assert vec.contrast > 0.0
    assert "r_mean" in vec.color_moments
    assert len(vec.spatial_features) > 100


def test_batch_extraction_to_dataframe(synthetic_image_batch):
    """Verify batch processing of image collections into a structured DataFrame."""
    images, labels = synthetic_image_batch
    engine = ComputerVisionFeatureEngine(target_size=(16, 16))

    report: VisionExtractionReport = engine.extract_batch_to_dataframe(images, labels=labels)

    assert isinstance(report, VisionExtractionReport)
    assert report.total_images == 40
    assert report.labels_found == ["square", "stripes"]
    assert isinstance(report.feature_dataframe, pd.DataFrame)
    assert len(report.feature_dataframe) == 40
    assert "label" in report.feature_dataframe.columns
    assert "blur_score" in report.feature_dataframe.columns
    assert "brightness" in report.feature_dataframe.columns
    assert "vfeat_0" in report.feature_dataframe.columns


def test_end_to_end_ml_training_on_image_features(synthetic_image_batch):
    """Verify extracted image feature DataFrame can be directly trained by AutoML."""
    images, labels = synthetic_image_batch
    engine = ComputerVisionFeatureEngine(target_size=(16, 16))
    report: VisionExtractionReport = engine.extract_batch_to_dataframe(images, labels=labels)

    feature_df = report.feature_dataframe.drop(columns=["image_id"])

    # Run AutoML Model Comparison on extracted visual features
    ml_engine = MLModelComparisonEngine()
    comp_report: ModelComparisonReport = ml_engine.compare_models(
        df=feature_df,
        target_column="label",
    )

    assert comp_report.best_model is not None
    assert comp_report.problem_type == "classification"
    assert comp_report.best_score > 0.85  # Strong visual discrimination on stripes vs square


def test_end_to_end_ann_training_on_image_features(synthetic_image_batch):
    """Verify extracted image features can be trained with ANNEngine MLP."""
    images, labels = synthetic_image_batch
    engine = ComputerVisionFeatureEngine(target_size=(16, 16))
    report: VisionExtractionReport = engine.extract_batch_to_dataframe(images, labels=labels)

    feature_df = report.feature_dataframe.drop(columns=["image_id"])

    ann_engine = ANNEngine()
    ann_result = ann_engine.train(
        df=feature_df,
        target_column="label",
        hidden_layer_sizes=[32, 16],
        max_iter=50,
    )

    assert ann_result.problem_type == "classification"
    assert ann_result.evaluation_metric in ("accuracy", "f1")
    assert ann_result.evaluation_score > 0.80
