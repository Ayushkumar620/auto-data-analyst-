"""
Universal Dataset-Agnostic Analytical Pipeline Test Suite.

Verifies end-to-end autonomous data analysis across arbitrary datasets,
modalities, tasks, and natural language user commands without hardcoded assumptions.
"""
import pytest
import numpy as np
import pandas as pd

from agent.canonical_data_layer import CanonicalDataLayer, CanonicalDataset, SemanticProfile
from agent.predictor import DataPredictor
from agent.autonomous_forecast_engine import AutonomousForecastEngine
from agent.forecasting_schemas import ForecastRequest
from agent.command_parser import CommandParser


# ==============================================================================
# Modality Fixtures (Arbitrary Names & Structures)
# ==============================================================================

@pytest.fixture
def regression_tabular_df():
    """Arbitrary continuous regression dataset."""
    rng = np.random.RandomState(42)
    n = 40
    x1 = rng.uniform(10, 100, n)
    x2 = rng.uniform(1, 10, n)
    # y is non-linear combination where Random Forest / Gradient Boosting excels
    y = 5.0 + 2.5 * x1 + 0.8 * (x2 ** 2) + rng.normal(0, 2.0, n)
    return pd.DataFrame({"feat_alpha": x1, "feat_beta": x2, "continuous_target": y})


@pytest.fixture
def classification_tabular_df():
    """Arbitrary binary/multiclass classification dataset."""
    rng = np.random.RandomState(42)
    n = 50
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    labels = np.where(x1 + x2 > 10.0, "HighRisk", "LowRisk")
    return pd.DataFrame({"metric_p": x1, "metric_q": x2, "risk_category": labels})


@pytest.fixture
def time_series_arbitrary_df():
    """Arbitrary longitudinal time-series dataset."""
    dates = pd.date_range("2021-01-01", periods=30, freq="ME")
    vals = 250.0 + 8.5 * np.arange(30) + 15.0 * np.sin(np.arange(30) * np.pi / 6.0)
    return pd.DataFrame({"recorded_epoch": dates, "sensor_output": vals, "notes": [None] * 28 + ["a", "b"]})


@pytest.fixture
def mixed_dirty_tabular_df():
    """Arbitrary dataset with currency strings, negative numbers, missing notes, and categorical codes."""
    n = 25
    return pd.DataFrame({
        "tx_id": [f"ID_{i}" for i in range(n)],  # Identifier
        "amount_str": [f"${1000 + i * 50:,.2f}" for i in range(n)],  # Formatted currency string
        "loss_val": [f"({i * 10})" if i % 3 == 0 else f"{i * 5}" for i in range(n)],  # Negative parentheses notation
        "channel": np.random.RandomState(42).choice(["Direct", "Partner", "Web"], size=n),
        "commentary": [None] * 22 + ["check", "valid", "done"],  # Sparse column
        "target_score": 50.0 + 3.0 * np.arange(n) + np.random.RandomState(42).normal(0, 1.0, n),
    })


# ==============================================================================
# 1. Canonical Data Layer & Semantic Profiling Tests
# ==============================================================================

def test_canonical_semantic_profiler_arbitrary_data(mixed_dirty_tabular_df):
    """Verify semantic profiler accurately categorizes columns without name assumptions."""
    dataset = CanonicalDataLayer.ingest(mixed_dirty_tabular_df)

    assert isinstance(dataset, CanonicalDataset)
    assert dataset.original_rows == 25
    assert "tx_id" in dataset.profile.identifier_columns
    assert "channel" in dataset.profile.categorical_columns
    assert "target_score" in dataset.profile.numeric_columns
    assert "amount_str" in dataset.profile.numeric_columns
    assert len(dataset.profile.target_candidates) > 0


def test_non_destructive_feature_cleaning_preserves_target_rows(mixed_dirty_tabular_df):
    """Verify sparse commentary column does not drop rows from valid target."""
    X, y, audit = CanonicalDataLayer.prepare_tabular_prediction_data(
        mixed_dirty_tabular_df,
        target_column="target_score",
        minimum_required_rows=10,
    )

    assert X is not None
    assert y is not None
    assert len(X) == 25
    assert len(y) == 25
    assert audit.valid_rows == 25
    assert audit.rows_removed == 0
    # Identifier tx_id should be excluded from X
    assert "tx_id" not in X.columns


# ==============================================================================
# 2. Data-Driven Model Benchmarking & Multi-Candidate Evaluation
# ==============================================================================

def test_supervised_regression_model_benchmarking(regression_tabular_df):
    """Verify DataPredictor benchmarks regression candidates and returns leaderboard & metrics."""
    dp = DataPredictor(regression_tabular_df)
    res = dp.predict(target="continuous_target")

    assert "error" not in res
    assert res["target"] == "continuous_target"
    assert res["metric"]["type"] == "regression"
    assert res["metric"]["r2_score"] is not None
    assert len(res["leaderboard"]) >= 3
    assert res["train_size"] + res["test_size"] == 40


def test_supervised_classification_model_benchmarking(classification_tabular_df):
    """Verify DataPredictor benchmarks classification candidates and returns accuracy/F1."""
    dp = DataPredictor(classification_tabular_df)
    res = dp.predict(target="risk_category")

    assert "error" not in res
    assert res["target"] == "risk_category"
    assert res["metric"]["type"] == "classification"
    assert "accuracy" in res["metric"]
    assert "f1_score" in res["metric"]
    assert len(res["leaderboard"]) >= 2


# ==============================================================================
# 3. Autonomous Time-Series Forecasting & Single Source of Truth
# ==============================================================================

def test_time_series_forecasting_single_source_of_truth(time_series_arbitrary_df):
    """Verify AutonomousForecastEngine is single source of truth across DataPredictor and direct calls."""
    engine = AutonomousForecastEngine()
    req = ForecastRequest(
        dataset=time_series_arbitrary_df,
        target_column="sensor_output",
        time_column="recorded_epoch",
        forecast_horizon=5,
    )
    canon_res = engine.run_forecast(req)

    dp = DataPredictor(time_series_arbitrary_df)
    dp_res = dp.forecast(target="sensor_output", periods=5)

    canon_preds = [round(float(p.prediction), 4) for p in canon_res.predictions]
    assert canon_preds == dp_res["forecast_values"]
    assert dp_res["history_points"] == 30
    assert dp_res["slope"] > 0
    assert dp_res["trend"] == "upward"


# ==============================================================================
# 4. User Intent & Command Interpretation
# ==============================================================================

def test_user_intent_command_routing_forecast(time_series_arbitrary_df):
    """Verify user command 'forecast next periods' routes dynamically to forecasting."""
    parser = CommandParser(time_series_arbitrary_df)
    res = parser.parse("forecast sensor_output for 4 periods")

    assert res["type"] == "forecast"
    assert "result" in res
    assert "forecast_values" in res["result"]
    assert len(res["result"]["forecast_values"]) == 4


def test_user_intent_command_routing_predict(regression_tabular_df):
    """Verify user command 'predict continuous target' routes to supervised ML."""
    parser = CommandParser(regression_tabular_df)
    res = parser.parse("predict continuous_target")

    assert res["type"] == "predict"
    assert "result" in res
    assert "metric" in res["result"]
    assert res["result"]["target"] == "continuous_target"


def test_user_intent_command_routing_summary(mixed_dirty_tabular_df):
    """Verify user command 'summary' routes to descriptive profiling."""
    parser = CommandParser(mixed_dirty_tabular_df)
    res = parser.parse("summary")

    assert res["type"] == "summary"
    assert "reports" in res


# ==============================================================================
# 5. Robust Error Diagnostics & Edge Case Guardrails
# ==============================================================================

def test_insufficient_rows_structured_diagnostics():
    """Verify datasets with insufficient rows return rich structured diagnostic metadata."""
    df_tiny = pd.DataFrame({"var_a": [1, 2, 3], "var_b": [10, 20, 30]})
    dp = DataPredictor(df_tiny)

    pred_res = dp.predict()
    assert "error" in pred_res
    assert pred_res["original_rows"] == 3
    assert pred_res["valid_rows"] == 3
    assert pred_res["minimum_required_rows"] == 10
    assert len(pred_res["removal_reasons"]) > 0

    fc_res = dp.forecast()
    assert "error" in fc_res
    assert fc_res["minimum_required_rows"] == 5
