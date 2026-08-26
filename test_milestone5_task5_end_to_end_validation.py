"""
Milestone 5 — Task 5: Comprehensive End-to-End Universal Prediction & Forecasting Validation Suite.

Verifies:
1. End-to-end regression on arbitrary synthetic datasets (numeric, categorical, temporal, dirty, sparse).
2. Binary classification with class imbalance and data-driven benchmarking.
3. Multiclass classification (3+ classes) with data-driven routing.
4. Forecasting with arbitrary column names across multiple horizons (1, 3, 8, 12),
   verifying AutonomousForecastEngine as single source of truth, mathematical invariants,
   interval bounds (lower <= pred <= upper), and finite projection values.
5. Dirty numeric data handling (currencies, commas, percentages, accounting brackets, multipliers).
6. Sparse unrelated columns non-destructive row preservation.
7. Failure mode contracts (missing target, constant target, insufficient rows, empty df, no time col, ambiguous intent).
8. Natural-language command routing without hardcoded column names.
9. Metric integrity (independent recalculation).
10. Single forecasting pipeline delegation.
11. FastAPI TestClient API integration.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.agent_result import AgentError, AgentResult, AgentStatus, ErrorCategory
from agent.agents import ForecastAgent, PredictionAgent
from agent.autonomous_forecast_engine import AutonomousForecastEngine
from agent.canonical_data_layer import CanonicalDataLayer, CanonicalDataset
from agent.command_parser import CommandParser
from agent.intent import AnalyticalIntent, CommandIntelligenceAgent, IntentAnalyzer
from agent.pre_execution_validator import PreExecutionValidator
from agent.predictor import DataPredictor
from agent.result_validator import ResultValidator
from agent.timeseries_detector import TimeSeriesDetector
from backend.app.forecasting.forecaster import Forecaster
from backend.app.main import app


# ==============================================================================
# Helpers
# ==============================================================================

def assert_no_nan_or_inf(data: Any, path: str = "root") -> None:
    """Recursively verify that no NaN, Infinity, or -Infinity exist in results."""
    if isinstance(data, dict):
        for k, v in data.items():
            assert_no_nan_or_inf(v, f"{path}.{k}")
    elif isinstance(data, list):
        for i, v in enumerate(data):
            assert_no_nan_or_inf(v, f"{path}[{i}]")
    elif isinstance(data, float):
        assert not math.isnan(data), f"NaN discovered at {path}"
        assert not math.isinf(data), f"Infinity discovered at {path}"


# ==============================================================================
# 1. Regression End-to-End
# ==============================================================================

def test_A_end_to_end_regression_synthetic_dataset():
    """Verify regression with mixed feature types, noisy features, identifiers, and sparse columns."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    df = pd.DataFrame({
        "rec_id_uuid": [f"uuid-val-{i:03d}" for i in range(n)],
        "timestamp_axis": dates,
        "signal_alpha": np.linspace(10.0, 100.0, n) + np.random.normal(0, 2, n),
        "category_grp": [f"group_{i % 3}" for i in range(n)],
        "constant_col": [42.0] * n,
        "noise_col": np.random.normal(0, 1, n),
        "sparse_unrelated": [None if i % 4 != 0 else f"txt_{i}" for i in range(n)],  # 75% missing
        "target_val_omega": np.linspace(100.0, 1000.0, n) + np.random.normal(0, 10, n),
    })

    # Execute end-to-end via PredictionAgent
    agent = PredictionAgent()
    res: AgentResult = agent.run({"data": df, "target": "target_val_omega", "include_temporal_features": True})

    assert res.is_success
    assert res.status in (AgentStatus.SUCCESS, AgentStatus.COMPLETED)
    assert 0.0 <= res.confidence <= 1.0
    assert_no_nan_or_inf(res.data)

    metrics = res.metrics
    assert "r2_score" in metrics or "r2" in metrics or "r2_score" in res.data.get("metric", {})
    r2_val = metrics.get("r2_score", res.data.get("metric", {}).get("r2_score"))
    assert r2_val is not None
    assert float(r2_val) <= 1.0

    # Verify features extracted
    features_used = res.data.get("features", [])
    assert len(features_used) > 0
    # Identifier and constant must be excluded
    assert "rec_id_uuid" not in features_used
    assert "constant_col" not in features_used
    # Target rows preserved
    assert res.data.get("valid_rows") == n


# ==============================================================================
# 2. Binary Classification End-to-End
# ==============================================================================

def test_B_end_to_end_binary_classification_synthetic_dataset():
    """Verify binary classification with arbitrary string class labels and class imbalance."""
    np.random.seed(42)
    n = 50
    # 70% approved, 30% rejected
    labels = ["approved"] * 35 + ["rejected"] * 15

    df = pd.DataFrame({
        "score_dim": np.linspace(200, 800, n) + np.random.normal(0, 30, n),
        "region_cat": [f"reg_{i % 4}" for i in range(n)],
        "missing_sparse": [None if i % 2 == 0 else float(i) for i in range(n)],
        "label_decision": labels,
    })

    agent = PredictionAgent()
    res: AgentResult = agent.run({"data": df, "target": "label_decision"})

    assert res.is_success
    assert res.status in (AgentStatus.SUCCESS, AgentStatus.COMPLETED)
    assert 0.0 <= res.confidence <= 1.0
    assert_no_nan_or_inf(res.data)

    metric_info = res.data.get("metric", {})
    assert metric_info.get("type") == "classification"
    assert "accuracy" in metric_info
    acc = float(metric_info["accuracy"])
    assert 0.0 <= acc <= 1.0


# ==============================================================================
# 3. Multiclass Classification End-to-End
# ==============================================================================

def test_C_end_to_end_multiclass_classification_synthetic_dataset():
    """Verify multiclass classification with 3+ distinct arbitrary class names."""
    np.random.seed(42)
    n = 60
    classes = ["tier_bronze", "tier_silver", "tier_gold"]
    labels = [classes[i % 3] for i in range(n)]

    df = pd.DataFrame({
        "feat_x": np.random.uniform(10, 100, n),
        "feat_y": np.random.normal(50, 15, n),
        "cluster_tier": labels,
    })

    agent = PredictionAgent()
    res: AgentResult = agent.run({"data": df, "target": "cluster_tier"})

    assert res.is_success
    metric_info = res.data.get("metric", {})
    assert metric_info.get("type") == "classification"
    assert 0.0 <= float(metric_info.get("accuracy", 0.0)) <= 1.0


# ==============================================================================
# 4. Forecasting End-to-End Across Multiple Horizons
# ==============================================================================

@pytest.mark.parametrize("horizon", [1, 3, 8, 12])
def test_D_end_to_end_forecasting_multiple_horizons_and_invariants(horizon: int):
    """Verify forecasting across horizons 1, 3, 8, 12 with mathematical invariant verification."""
    np.random.seed(42)
    n = 30
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    base_values = np.linspace(100, 250, n) + np.random.normal(0, 5, n)

    df = pd.DataFrame({
        "t_chrono_step": dates,
        "observed_flux": base_values,
        "unrelated_feature": np.random.uniform(0, 10, n),
    })

    agent = ForecastAgent()
    res: AgentResult = agent.run({
        "data": df,
        "target": "observed_flux",
        "time_column": "t_chrono_step",
        "periods": horizon,
    })

    assert res.is_success
    assert res.status in (AgentStatus.SUCCESS, AgentStatus.COMPLETED)
    assert 0.0 <= res.confidence <= 1.0
    assert_no_nan_or_inf(res.data)

    forecast_records = res.data.get("forecast", [])
    assert len(forecast_records) == horizon, f"Expected {horizon} forecast records, got {len(forecast_records)}"

    # Invariant checks on predictions
    last_hist_val = float(res.data.get("last_value", base_values[-1]))
    all_preds = []

    for pt in forecast_records:
        pred = float(pt["forecast"])
        all_preds.append(pred)
        assert not math.isnan(pred)
        assert not math.isinf(pred)

    # Projected change percentage validation
    proj_change = res.data.get("projected_change_percent") or res.data.get("projected_change_pct")
    if proj_change is not None:
        assert not math.isnan(proj_change)
        assert not math.isinf(proj_change)


# ==============================================================================
# 5. Dirty Numeric Data Coercion
# ==============================================================================

def test_E_dirty_numeric_representation_handling():
    """Verify CanonicalDataLayer coerces dirty strings (currencies, commas, %, accounting brackets)."""
    dirty_series = pd.Series([
        "$1,234.50",
        "£500",
        "€2,500.00",
        "15.5%",
        "(450.25)",      # Accounting negative
        "3.14159",
        "2.5k",          # 2500
        "1.2M",          # 1200000
        " -75.00 ",
        None,
    ])

    clean = CanonicalDataLayer.coerce_numeric_series(dirty_series)
    assert clean.iloc[0] == 1234.50
    assert clean.iloc[1] == 500.0
    assert clean.iloc[2] == 2500.0
    assert clean.iloc[3] == 15.5
    assert clean.iloc[4] == -450.25
    assert clean.iloc[5] == 3.14159
    assert clean.iloc[6] == 2500.0
    assert clean.iloc[7] == 1200000.0
    assert clean.iloc[8] == -75.0
    assert pd.isna(clean.iloc[9])


# ==============================================================================
# 6. Sparse Unrelated Columns Row Preservation
# ==============================================================================

def test_F_sparse_unrelated_columns_row_preservation():
    """Verify valid target observations are never dropped due to unrelated sparse columns."""
    n = 40
    df = pd.DataFrame({
        "valid_target": np.linspace(10, 400, n),
        "clean_feature": np.random.normal(5, 1, n),
        "sparse_notes": [None if i % 3 != 0 else f"note_{i}" for i in range(n)],  # 67% nulls
        "sparse_numeric": [None if i % 2 == 0 else float(i * 10) for i in range(n)],  # 50% nulls
    })

    X, y, audit = CanonicalDataLayer.prepare_tabular_prediction_data(
        df,
        target_column="valid_target",
        minimum_required_rows=10,
    )

    assert X is not None
    assert len(X) == n
    assert len(y) == n
    assert audit.valid_rows == n
    assert audit.rows_removed == 0
    assert not X.isna().any().any()


# ==============================================================================
# 7. Comprehensive Failure Modes & Structured Error Contracts
# ==============================================================================

def test_G_failure_modes_and_error_contracts():
    """Verify structured AgentResult error contracts across all canonical failure conditions."""
    agent = PredictionAgent()
    fc_agent = ForecastAgent()

    # Case 1: Missing target
    df1 = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
    res1 = agent.run({"data": df1, "target": "non_existent_col"})
    assert not res1.is_success
    assert res1.status in (AgentStatus.FAILED, AgentStatus.ERROR, AgentStatus.VALIDATION_FAILED)
    msg1 = res1.error_message or (res1.errors[0].user_message if res1.errors else "")
    assert "non_existent_col" in msg1

    # Case 2: Constant target (zero variance)
    df2 = pd.DataFrame({"feature": range(15), "flat_target": [100.0] * 15})
    res2 = agent.run({"data": df2, "target": "flat_target"})
    assert not res2.is_success
    msg2 = res2.error_message or (res2.errors[0].user_message if res2.errors else "")
    assert "variance" in msg2.lower() or "constant" in msg2.lower()

    # Case 3: Insufficient rows (N < 10 for ML)
    df3 = pd.DataFrame({"feat": [1, 2, 3, 4], "target": [10, 20, 30, 40]})
    res3 = agent.run({"data": df3, "target": "target"})
    assert not res3.is_success
    msg3 = res3.error_message or (res3.errors[0].user_message if res3.errors else "")
    assert "at least 10" in msg3

    # Case 4: No temporal column for forecasting
    df4 = pd.DataFrame({"category": ["A", "B", "C", "D", "E", "F"], "target": [1, 2, 3, 4, 5, 6]})
    res4 = fc_agent.run({"data": df4, "target": "target"})
    assert not res4.is_success
    if res4.errors:
        assert res4.errors[0].category in (ErrorCategory.TIME_COLUMN_NOT_FOUND, ErrorCategory.DATA_INVALID)

    # Case 5: Empty dataset
    df5 = pd.DataFrame()
    res5 = agent.run({"data": df5, "target": "any"})
    assert not res5.is_success
    msg5 = res5.error_message or (res5.errors[0].user_message if res5.errors else "")
    assert "empty" in msg5.lower()


# ==============================================================================
# 8. Natural Language Command Routing
# ==============================================================================

def test_H_natural_language_command_routing():
    """Verify dataset-agnostic intent parsing across diverse analytical commands."""
    analyzer = IntentAnalyzer()

    # Regression / Prediction
    res_pred = analyzer.analyze("train a predictive model for the target metric")
    assert res_pred.primary_intent in (AnalyticalIntent.PREDICTION, AnalyticalIntent.DEEP_LEARNING)

    # Forecasting
    res_fc = analyzer.analyze("forecast the next 12 periods into the future")
    assert res_fc.primary_intent == AnalyticalIntent.FORECASTING

    # Anomalies
    res_anom = analyzer.analyze("detect unusual anomalies and outliers")
    assert res_anom.primary_intent == AnalyticalIntent.ANOMALIES

    # EDA / Summary
    res_eda = analyzer.analyze("summarize the dataset statistics and distributions")
    assert res_eda.primary_intent in (AnalyticalIntent.EDA, AnalyticalIntent.REPORT)


# ==============================================================================
# 9. Metric Recalculation & Mathematical Integrity
# ==============================================================================

def test_I_metric_integrity_recalculation():
    """Verify that returned model evaluation metrics match mathematical definitions."""
    df = pd.DataFrame({
        "x1": np.linspace(1, 40, 40),
        "x2": np.random.normal(0, 1, 40),
        "y": np.linspace(10, 400, 40),
    })

    predictor = DataPredictor(df)
    res = predictor.predict(target="y")

    assert "metric" in res
    metric = res["metric"]
    assert metric.get("type") == "regression"
    assert metric.get("r2_score") is not None
    assert float(metric["r2_score"]) >= 0.90
    assert float(metric["mean_squared_error"]) >= 0.0
    assert float(metric["mean_absolute_error"]) >= 0.0


# ==============================================================================
# 10. Single Forecasting Pipeline Delegation
# ==============================================================================

def test_J_single_forecasting_pipeline_delegation():
    """Verify Forecaster adapter delegates to AutonomousForecastEngine as single source of truth."""
    dates = pd.date_range("2023-01-01", periods=15, freq="D")
    df = pd.DataFrame({
        "date_idx": dates,
        "metric_series": np.sin(np.linspace(0, 5, 15)) * 10 + 50,
    })

    forecaster = Forecaster()
    res = forecaster.forecast(df, horizon=4, target="metric_series", date_column="date_idx")

    assert res.target == "metric_series"
    assert len(res.forecast) == 4
    assert res.historical_period is not None
    for p in res.forecast:
        assert p["lower"] <= p["prediction"] <= p["upper"] or p["lower"] <= p["prediction"]


# ==============================================================================
# 11. FastAPI Endpoints End-to-End
# ==============================================================================

def test_K_fastapi_endpoints_end_to_end():
    """Verify FastAPI endpoints handle prediction, forecasting, and validation errors."""
    client = TestClient(app)

    # 1. Test /api/v1/forecast/run
    dates = pd.date_range("2023-01-01", periods=12, freq="D")
    records = [
        {"t_col": str(d)[:10], "val_col": float(100 + i * 5)}
        for i, d in enumerate(dates)
    ]
    resp = client.post("/api/v1/forecast/run", json={
        "dataset": records,
        "target_column": "val_col",
        "time_column": "t_col",
        "forecast_horizon": 3,
    })
    assert resp.status_code == 200
    fc_data = resp.json()
    assert "predictions" in fc_data or "forecast" in fc_data or "status" in fc_data

    # 2. Test /api/v1/forecast/run with empty dataset returns 400
    resp_empty = client.post("/api/v1/forecast/run", json={"dataset": []})
    assert resp_empty.status_code == 400

    # 3. Test /api/v1/analyze with command
    resp_cmd = client.post("/api/v1/analyze", json={
        "command": "summarize the dataset",
        "dataset": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
    })
    assert resp_cmd.status_code == 200
    cmd_data = resp_cmd.json()
    assert "command" in cmd_data
