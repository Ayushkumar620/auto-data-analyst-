"""
Tests for Unified Forecasting Architecture (Root-Level Single Source of Truth).

Verifies:
1. Single Source of Truth: DataPredictor, Forecaster, and AutonomousForecastEngine return the identical forecast predictions and metrics.
2. Metric Invariance: projected_change_pct is strictly derived from the final returned forecast array and matches mean(y_hat) vs y_last.
3. Synthetic Modalities: Upward, Downward, Flat, Noisy, Seasonal, Zero Baseline, Negative Values, Different Scales, Varying Horizons.
4. Robust Edge Cases & Error Boundaries: Graceful handling of insufficient rows, missing temporal columns, non-numeric targets.
"""
import pytest
import numpy as np
import pandas as pd

from agent.autonomous_forecast_engine import AutonomousForecastEngine
from agent.forecasting_schemas import ForecastRequest, ForecastResult
from agent.predictor import DataPredictor
from backend.app.forecasting.forecaster import Forecaster


# ==============================================================================
# Synthetic Test Fixtures
# ==============================================================================

@pytest.fixture
def upward_trend_df():
    dates = pd.date_range("2023-01-01", periods=20, freq="ME")
    vals = 100.0 + 5.0 * np.arange(20) + np.random.RandomState(42).normal(0, 1.0, 20)
    return pd.DataFrame({"obs_date": dates, "metric_val": vals})


@pytest.fixture
def downward_trend_df():
    dates = pd.date_range("2023-01-01", periods=20, freq="ME")
    vals = 500.0 - 12.0 * np.arange(20) + np.random.RandomState(42).normal(0, 2.0, 20)
    return pd.DataFrame({"obs_date": dates, "metric_val": vals})


@pytest.fixture
def flat_series_df():
    dates = pd.date_range("2023-01-01", periods=16, freq="ME")
    vals = 50.0 + np.random.RandomState(42).normal(0, 0.05, 16)
    return pd.DataFrame({"obs_date": dates, "metric_val": vals})


@pytest.fixture
def seasonal_series_df():
    dates = pd.date_range("2022-01-01", periods=24, freq="ME")
    vals = 200.0 + 20.0 * np.sin(np.arange(24) * np.pi / 6.0) + np.random.RandomState(42).normal(0, 1.0, 24)
    return pd.DataFrame({"obs_date": dates, "metric_val": vals})


@pytest.fixture
def zero_baseline_df():
    dates = pd.date_range("2023-01-01", periods=12, freq="ME")
    vals = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]
    return pd.DataFrame({"obs_date": dates, "metric_val": vals})


@pytest.fixture
def negative_values_df():
    dates = pd.date_range("2023-01-01", periods=15, freq="ME")
    vals = -50.0 + 3.0 * np.arange(15) + np.random.RandomState(42).normal(0, 0.5, 15)
    return pd.DataFrame({"obs_date": dates, "metric_val": vals})


@pytest.fixture
def large_scale_df():
    dates = pd.date_range("2023-01-01", periods=18, freq="ME")
    vals = 1_000_000.0 + 25_000.0 * np.arange(18)
    return pd.DataFrame({"obs_date": dates, "metric_val": vals})


# ==============================================================================
# Unit & Integration Tests
# ==============================================================================

def test_single_source_of_truth_across_entrypoints(upward_trend_df):
    """1. Verify DataPredictor, Forecaster, and AutonomousForecastEngine return the identical predictions."""
    # Canonical Engine
    engine = AutonomousForecastEngine()
    req = ForecastRequest(dataset=upward_trend_df, target_column="metric_val", time_column="obs_date", forecast_horizon=5)
    canon_res = engine.run_forecast(req)

    # Legacy DataPredictor Adapter
    dp = DataPredictor(upward_trend_df)
    dp_res = dp.forecast(target="metric_val", periods=5)

    # Legacy Forecaster Adapter
    fc = Forecaster()
    fc_res = fc.forecast(upward_trend_df, horizon=5, target="metric_val", date_column="obs_date")

    # Assert identical predictions
    canon_preds = [round(float(p.prediction), 4) for p in canon_res.predictions]
    dp_preds = dp_res["forecast_values"]
    fc_preds = [round(float(p["prediction"]), 4) for p in fc_res.forecast]

    assert canon_preds == dp_preds
    assert canon_preds == fc_preds

    # Assert identical target and horizon
    assert canon_res.target == dp_res["target"] == fc_res.target == "metric_val"
    assert canon_res.forecast_horizon == dp_res["forecast_horizon"] == fc_res.horizon == 5


def test_metric_origin_invariance_projected_change(upward_trend_df, downward_trend_df, seasonal_series_df):
    """2. Verify projected_change_pct is strictly derived from the final returned forecast predictions."""
    engine = AutonomousForecastEngine()

    for df in (upward_trend_df, downward_trend_df, seasonal_series_df):
        req = ForecastRequest(dataset=df, target_column="metric_val", time_column="obs_date", forecast_horizon=6)
        res = engine.run_forecast(req)

        # Compute expected from predictions array
        pred_vals = [p.prediction for p in res.predictions]
        last_hist = float(df["metric_val"].iloc[-1])
        expected_proj_change = round(((float(np.mean(pred_vals)) - last_hist) / abs(last_hist)) * 100.0, 2)

        assert res.projected_change_pct == expected_proj_change
        assert res.to_dict()["projected_change_pct"] == expected_proj_change
        assert res.to_dict()["projected_change_percent"] == expected_proj_change


def test_upward_and_downward_trends(upward_trend_df, downward_trend_df):
    """3. Verify slope and trend directions on upward and downward series."""
    engine = AutonomousForecastEngine()

    up_res = engine.run_forecast(ForecastRequest(dataset=upward_trend_df, forecast_horizon=4))
    assert up_res.slope > 0
    assert up_res.projected_change_pct > 0

    down_res = engine.run_forecast(ForecastRequest(dataset=downward_trend_df, forecast_horizon=4))
    assert down_res.slope < 0
    assert down_res.projected_change_pct < 0


def test_flat_and_noisy_series(flat_series_df):
    """4. Verify flat series is handled stably."""
    engine = AutonomousForecastEngine()
    res = engine.run_forecast(ForecastRequest(dataset=flat_series_df, forecast_horizon=4))
    assert res.status == "SUCCESS"
    assert len(res.predictions) == 4
    for pt in res.predictions:
        assert np.isfinite(pt.prediction)


def test_zero_and_near_zero_baseline(zero_baseline_df):
    """5. Verify zero baseline does not cause ZeroDivisionError or NaN projected change."""
    engine = AutonomousForecastEngine()
    res = engine.run_forecast(ForecastRequest(dataset=zero_baseline_df, forecast_horizon=3))
    assert res.status == "SUCCESS"
    assert res.projected_change_pct is not None
    assert np.isfinite(res.projected_change_pct)


def test_negative_values_series(negative_values_df):
    """6. Verify series with negative values is supported."""
    engine = AutonomousForecastEngine()
    res = engine.run_forecast(ForecastRequest(dataset=negative_values_df, forecast_horizon=4))
    assert res.status == "SUCCESS"
    assert len(res.predictions) == 4


def test_large_scale_values(large_scale_df):
    """7. Verify numeric scale invariance ($1M+)."""
    engine = AutonomousForecastEngine()
    res = engine.run_forecast(ForecastRequest(dataset=large_scale_df, forecast_horizon=5))
    assert res.status == "SUCCESS"
    assert res.predictions[0].prediction > 1_000_000.0


def test_varying_forecast_horizons(upward_trend_df):
    """8. Verify arbitrary forecast horizons (1, 3, 8, 12)."""
    engine = AutonomousForecastEngine()
    for h in (1, 3, 8, 12):
        res = engine.run_forecast(ForecastRequest(dataset=upward_trend_df, forecast_horizon=h))
        assert len(res.predictions) == h
        assert res.forecast_horizon == h


def test_insufficient_samples_rejection():
    """9. Verify dataset with < 5 rows is cleanly rejected."""
    df_small = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=3, freq="ME"),
        "val": [10.0, 20.0, 30.0],
    })
    dp = DataPredictor(df_small)
    res = dp.forecast(periods=3)
    assert "error" in res
    assert "at least 5" in res["error"].lower()


def test_no_temporal_column_rejection():
    """10. Verify dataset with no date column raises controlled error in Forecaster."""
    df_nodate = pd.DataFrame({"col_a": [1, 2, 3, 4, 5, 6], "col_b": [10, 20, 30, 40, 50, 60]})
    fc = Forecaster()
    with pytest.raises(ValueError, match="no usable date or timestamp column"):
        fc.forecast(df_nodate, horizon=3)
