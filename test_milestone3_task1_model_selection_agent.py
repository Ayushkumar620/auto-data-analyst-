"""
Tests for Milestone 3, Task 1: Intelligent Model Selection Agent.

Verifies:
1. Tabular regression candidate generation & scoring
2. Tabular binary classification candidate generation & scoring
3. Multiclass classification candidate generation & scoring
4. Imbalanced classification metric prioritization (F1/PR-AUC over accuracy)
5. Clustering task detection & candidate generation (K-Means, DBSCAN)
6. Time-series detection & candidate generation
7. Image / CNN eligibility detection
8. CNN rejected for tabular data (suitability = 0.0 with explicit rationale)
9. ANN candidate selection (high suitability on large data, down-weighted on small data)
10. Target missing handling & error reporting
11. Insufficient samples handling (< 10 rows)
12. Data leakage detection (target derivatives, IDs, future features)
13. Evaluation metric selection rules
14. Model suitability scoring explainability
15. LLM unavailable deterministic fallback
"""
import pytest
import numpy as np
import pandas as pd

from agent.dataset_knowledge import DatasetKnowledge
from agent.intent import IntentType, UserIntent
from agent.model_selection_agent import ModelSelectionAgent
from agent.model_selection_schemas import (
    DataModality,
    MLTaskType,
    ModelCandidate,
    ModelSelectionRequest,
    ModelSelectionResult,
)
from agent.schemas import AgentResult, AgentStatus


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def regression_df():
    """Synthetic tabular regression dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feature_1": np.random.uniform(10, 50, n),
        "feature_2": np.random.uniform(1, 10, n),
        "feature_3": np.random.normal(0, 1, n),
        "revenue": np.random.uniform(100, 1000, n),
    })


@pytest.fixture
def binary_classification_df():
    """Synthetic binary classification dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "age": np.random.randint(18, 70, n),
        "monthly_spend": np.random.uniform(50, 500, n),
        "support_tickets": np.random.randint(0, 10, n),
        "is_churned": np.random.choice([0, 1], size=n, p=[0.5, 0.5]),
    })


@pytest.fixture
def imbalanced_classification_df():
    """Synthetic heavily imbalanced binary classification dataset (90/10)."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "tx_amount": np.random.uniform(10, 1000, n),
        "tx_count": np.random.randint(1, 20, n),
        "is_fraud": np.random.choice([0, 1], size=n, p=[0.90, 0.10]),
    })


@pytest.fixture
def multiclass_df():
    """Synthetic multiclass classification dataset."""
    np.random.seed(42)
    n = 90
    return pd.DataFrame({
        "x1": np.random.normal(0, 1, n),
        "x2": np.random.normal(5, 2, n),
        "risk_tier": np.random.choice(["Low", "Medium", "High"], size=n),
    })


@pytest.fixture
def pixel_image_df():
    """Synthetic 28x28 pixel grid dataframe (784 pixels)."""
    np.random.seed(42)
    n = 20
    data = {f"pixel_{i}": np.random.uniform(0, 255, n) for i in range(784)}
    data["digit_label"] = np.random.randint(0, 10, n)
    return pd.DataFrame(data)


# ==============================================================================
# 1-4. Candidate Generation & Metric Prioritization Tests
# ==============================================================================

def test_tabular_regression_candidate_generation(regression_df):
    """1. Test candidate model generation and ranking for tabular regression."""
    agent = ModelSelectionAgent()
    request = ModelSelectionRequest(
        target_column="revenue",
        feature_columns=["feature_1", "feature_2", "feature_3"],
    )
    result = agent.plan_model_selection(request, dataframe=regression_df)

    assert isinstance(result, ModelSelectionResult)
    assert result.task_type == MLTaskType.REGRESSION.value
    assert result.data_modality == DataModality.TABULAR.value
    assert result.evaluation_metric == "r2"

    candidate_names = [c.model_name for c in result.candidates]
    assert "Linear Regression" in candidate_names
    assert "Random Forest Regressor" in candidate_names
    assert "Gradient Boosting Regressor" in candidate_names

    # Top candidate should be tree ensemble
    assert result.candidates[0].suitability_score >= 0.85


def test_tabular_binary_classification(binary_classification_df):
    """2. Test candidate model generation for balanced binary classification."""
    agent = ModelSelectionAgent()
    request = ModelSelectionRequest(
        target_column="is_churned",
        feature_columns=["age", "monthly_spend", "support_tickets"],
    )
    result = agent.plan_model_selection(request, dataframe=binary_classification_df)

    assert result.task_type == MLTaskType.BINARY_CLASSIFICATION.value
    assert result.evaluation_metric == "roc_auc"
    candidate_names = [c.model_name for c in result.candidates]
    assert "Logistic Regression" in candidate_names
    assert "Random Forest Classifier" in candidate_names
    assert "Gradient Boosting Classifier" in candidate_names


def test_multiclass_classification(multiclass_df):
    """3. Test multiclass classification task detection and candidate models."""
    agent = ModelSelectionAgent()
    request = ModelSelectionRequest(
        target_column="risk_tier",
        feature_columns=["x1", "x2"],
    )
    result = agent.plan_model_selection(request, dataframe=multiclass_df)

    assert result.task_type == MLTaskType.MULTICLASS_CLASSIFICATION.value
    assert result.evaluation_metric == "accuracy"
    assert len(result.candidates) >= 3


def test_imbalanced_classification_metric_prioritization(imbalanced_classification_df):
    """4. Test that imbalanced binary classification prioritizes F1 over ROC-AUC/Accuracy."""
    agent = ModelSelectionAgent()
    request = ModelSelectionRequest(
        target_column="is_fraud",
        feature_columns=["tx_amount", "tx_count"],
    )
    result = agent.plan_model_selection(request, dataframe=imbalanced_classification_df)

    assert result.task_type == MLTaskType.BINARY_CLASSIFICATION.value
    # Crucial invariant: imbalanced data must prioritize F1
    assert result.evaluation_metric == "f1"
    assert "f1" in result.secondary_metrics


# ==============================================================================
# 5-8. Modality, Clustering, Time-Series & CNN Eligibility Tests
# ==============================================================================

def test_clustering_task_detection(regression_df):
    """5. Test clustering candidate generation when no target is provided."""
    agent = ModelSelectionAgent()
    intent = UserIntent(
        intent_type=IntentType.DATASET_ANALYSIS,
        objective="Group customers into clusters without target labels.",
        required_capabilities=["clustering"],
    )
    request = ModelSelectionRequest(user_intent=intent)
    result = agent.plan_model_selection(request, dataframe=regression_df)

    assert result.task_type == MLTaskType.CLUSTERING.value
    candidate_names = [c.model_name for c in result.candidates]
    assert "K-Means Clustering" in candidate_names
    assert "DBSCAN Clustering" in candidate_names
    assert result.evaluation_metric == "silhouette"


def test_time_series_detection(regression_df):
    """6. Test time-series detection and forecasting model candidate generation."""
    agent = ModelSelectionAgent()
    ts_df = regression_df.copy()
    ts_df["date"] = pd.date_range("2024-01-01", periods=len(ts_df), freq="D")

    intent = UserIntent(
        intent_type=IntentType.FORECASTING,
        objective="Forecast next month's revenue.",
        metrics=["revenue"],
        required_capabilities=["forecasting"],
    )
    request = ModelSelectionRequest(
        target_column="revenue",
        user_intent=intent,
    )
    result = agent.plan_model_selection(request, dataframe=ts_df)

    assert result.data_modality == DataModality.TIME_SERIES.value
    assert result.task_type == MLTaskType.TIME_SERIES_FORECASTING.value
    assert "ARIMA / Exponential Smoothing" in [c.model_name for c in result.candidates]


def test_image_cnn_eligibility(pixel_image_df):
    """7. Test that image pixel grids are recognized as Image Modality with high CNN suitability."""
    agent = ModelSelectionAgent()
    request = ModelSelectionRequest(target_column="digit_label")
    result = agent.plan_model_selection(request, dataframe=pixel_image_df)

    assert result.data_modality == DataModality.IMAGE.value
    cnn_candidate = next(c for c in result.candidates if "Convolutional Neural Network" in c.model_name)
    assert cnn_candidate.suitability_score >= 0.90


def test_cnn_rejected_for_tabular_data(regression_df):
    """8. Invariant: CNN must have suitability score = 0.0 for ordinary tabular data."""
    agent = ModelSelectionAgent()
    request = ModelSelectionRequest(target_column="revenue")
    result = agent.plan_model_selection(request, dataframe=regression_df)

    cnn_candidate = next(c for c in result.candidates if "Convolutional Neural Network" in c.model_name)
    assert cnn_candidate.suitability_score == 0.0
    assert "not applicable for 1D tabular" in cnn_candidate.reason or "spatial" in cnn_candidate.reason


# ==============================================================================
# 9-12. Deep Learning, Error Handling & Data Leakage Tests
# ==============================================================================

def test_ann_candidate_selection(regression_df):
    """9. Test ANN suitability scaling across small and large sample sizes."""
    agent = ModelSelectionAgent()

    # Small dataset (15 samples)
    small_df = regression_df.iloc[:15]
    res_small = agent.plan_model_selection(ModelSelectionRequest(target_column="revenue"), dataframe=small_df)
    ann_small = next(c for c in res_small.candidates if "ANN" in c.model_name)
    assert ann_small.suitability_score <= 0.50  # Penalized for overfitting risk on small data

    # Large dataset (300 samples)
    large_df = pd.concat([regression_df] * 3, ignore_index=True)
    res_large = agent.plan_model_selection(ModelSelectionRequest(target_column="revenue"), dataframe=large_df)
    ann_large = next(c for c in res_large.candidates if "ANN" in c.model_name)
    assert ann_large.suitability_score >= 0.70  # Higher suitability with ample training rows


def test_target_missing_warning(regression_df):
    """10. Test that specifying a non-existent target produces a warning."""
    agent = ModelSelectionAgent()
    request = ModelSelectionRequest(target_column="non_existent_column_xyz")
    result = agent.plan_model_selection(request, dataframe=regression_df)

    assert any("does not exist in dataset" in w for w in result.warnings)


def test_insufficient_samples_handling(regression_df):
    """11. Test that benchmarking with fewer than 10 samples returns a clear error."""
    agent = ModelSelectionAgent()
    tiny_df = regression_df.iloc[:5]  # Only 5 rows
    result = agent.run({"data": tiny_df, "target": "revenue"})

    assert isinstance(result, AgentResult)
    assert result.is_success is False
    assert "requires at least 10 samples" in result.message


def test_data_leakage_detection():
    """12. Test detection of target derivatives, identifiers, and post-outcome features."""
    agent = ModelSelectionAgent()
    leak_df = pd.DataFrame({
        "customer_id": [f"ID_{i}" for i in range(50)],  # ID column
        "churn": [0, 1] * 25,
        "churn_reason_code": [0, 1] * 25,  # Target-derived column
        "post_cancellation_survey": [1, 2] * 25,  # Future event token
        "tenure_months": np.random.randint(1, 60, 50),
    })

    leakages = agent.detect_data_leakage(leak_df, "churn", list(leak_df.columns))
    assert len(leakages) >= 3
    assert any("derived from target" in l for l in leakages)
    assert any("identifier" in l for l in leakages)
    assert any("future or post-outcome" in l for l in leakages)


# ==============================================================================
# 13-15. Metrics, Explainability & LLM Fallback Tests
# ==============================================================================

def test_evaluation_metric_selection_rules():
    """13. Test metric selection across all supported ML task types."""
    agent = ModelSelectionAgent()

    reg_metric, _ = agent.select_evaluation_metrics(MLTaskType.REGRESSION)
    assert reg_metric == "r2"

    clf_metric, _ = agent.select_evaluation_metrics(
        MLTaskType.BINARY_CLASSIFICATION,
        pd.Series([0] * 50 + [1] * 50)
    )
    assert clf_metric == "roc_auc"

    imb_metric, _ = agent.select_evaluation_metrics(
        MLTaskType.BINARY_CLASSIFICATION,
        pd.Series([0] * 90 + [1] * 10)
    )
    assert imb_metric == "f1"

    clus_metric, _ = agent.select_evaluation_metrics(MLTaskType.CLUSTERING)
    assert clus_metric == "silhouette"


def test_model_suitability_explainability(regression_df):
    """14. Test that suitability scores come with clear reasons and evidence."""
    agent = ModelSelectionAgent()
    result = agent.plan_model_selection(ModelSelectionRequest(target_column="revenue"), dataframe=regression_df)

    assert len(result.selection_reason) > 20
    assert len(result.evidence) >= 1
    assert result.evidence[0].source == "ModelSelectionAgent"
    for candidate in result.candidates:
        assert len(candidate.reason) > 10


def test_llm_unavailable_deterministic_fallback(binary_classification_df):
    """15. Test high-level natural language candidate selection without LLM."""
    agent = ModelSelectionAgent(llm_provider=None)
    result = agent.select_candidates(
        "Build the best model to predict is_churned",
        dataframe=binary_classification_df,
    )

    assert isinstance(result, ModelSelectionResult)
    assert result.task_type == MLTaskType.BINARY_CLASSIFICATION.value
    assert result.selected_model in ("Random Forest Classifier", "Gradient Boosting Classifier", "Logistic Regression")
    assert len(result.candidates) >= 3

