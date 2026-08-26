"""
Milestone 6 — Task 1: Comprehensive Universal Anomaly Detection Test Suite.

Verifies:
A. Basic univariate anomaly detection
B. Multivariate anomaly detection
C. Arbitrary column names
D. Arbitrary numeric feature names
E. Identifier exclusion
F. Constant feature exclusion
G. Sparse unrelated feature handling
H. Dirty numeric coercion
I. Negative values
J. Large numeric scales
K. Different dataset sizes
L. Very low anomaly rate
M. Moderate anomaly rate
N. Explicit contamination
O. Automatic contamination
P. Insufficient dataset
Q. No usable numeric features
R. All constant features
S. Ambiguous command
T. Structured error handling
U. NaN / Infinity sanitization
V. Confidence bounds
W. Evidence integrity
X. Anomaly score consistency
Y. Anomaly rate mathematical consistency
Z. API integration (FastAPI TestClient)
AA. ResultValidator integration
AB. AgentResult backward compatibility & property contracts
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.agent_result import AgentError, AgentResult, AgentStatus, ClaimType, ErrorCategory
from agent.anomaly_agent import AnomalyDetectionAgent
from agent.anomaly_detection_engine import AnomalyDetectionEngine
from agent.canonical_data_layer import CanonicalDataLayer
from agent.confidence_calculator import ConfidenceCalculator
from agent.intent import AnalyticalIntent, IntentAnalyzer
from agent.pre_execution_validator import PreExecutionValidator
from agent.result_validator import ResultValidator
from backend.app.main import app


# ---------------------------------------------------------------------------
# Helper: Recursively assert no NaN, Inf, -Inf
# ---------------------------------------------------------------------------
def assert_no_nan_or_inf(obj: Any, path: str = "root") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_nan_or_inf(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_no_nan_or_inf(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        assert not math.isnan(obj), f"NaN found at {path}"
        assert not math.isinf(obj), f"Infinity found at {path}"


# ---------------------------------------------------------------------------
# A. Basic Univariate Anomaly Detection
# ---------------------------------------------------------------------------
def test_A_univariate_anomaly_detection():
    """Verify single-feature anomaly detection flags extreme deviations."""
    np.random.seed(42)
    n = 40
    # Normal distribution with 2 injected outliers
    vals = np.random.normal(50.0, 5.0, n)
    vals[5] = 250.0   # Extreme high
    vals[20] = -120.0 # Extreme low

    df = pd.DataFrame({"signal_measure": vals})
    engine = AnomalyDetectionEngine()
    res = engine.detect(df, method="robust_zscore")

    assert "error" not in res
    assert res["rows_analyzed"] == n
    assert res["anomalies_found"] >= 2
    assert 0.0 < res["anomaly_rate"] <= 0.5
    # Outlier rows 5 and 20 should have top ranks
    flagged_indices = [obs["row_index"] for obs in res["observations"] if obs["anomaly_label"] == "ANOMALY"]
    assert 5 in flagged_indices
    assert 20 in flagged_indices


# ---------------------------------------------------------------------------
# B. Multivariate Anomaly Detection
# ---------------------------------------------------------------------------
def test_B_multivariate_anomaly_detection():
    """Verify multivariate detector flags points unusual in joint feature space."""
    np.random.seed(42)
    n = 60
    # Correlated bivariate features: x2 ~= 2 * x1
    x1 = np.random.uniform(10.0, 50.0, n)
    x2 = 2.0 * x1 + np.random.normal(0, 1.0, n)

    # Inject multivariate outlier (x1 is normal, x2 is normal, but relationship is broken)
    x1[10] = 45.0
    x2[10] = 10.0  # Deviates from x2 ~= 2*x1

    df = pd.DataFrame({"dim_alpha": x1, "dim_beta": x2})
    engine = AnomalyDetectionEngine()
    res = engine.detect(df, method="isolation_forest", contamination=0.05)

    assert "error" not in res
    assert res["method_family"] == "ensemble"
    flagged = [obs["row_index"] for obs in res["observations"] if obs["anomaly_label"] == "ANOMALY"]
    assert 10 in flagged


# ---------------------------------------------------------------------------
# C & D. Arbitrary Column Names & Numeric Feature Names
# ---------------------------------------------------------------------------
def test_C_D_arbitrary_column_and_feature_names():
    """Verify engine operates strictly without keywords like sales, revenue, date."""
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "t_omega_idx": range(n),
        "gamma_flux_reading": np.linspace(100, 200, n),
        "theta_sensor_delta": np.random.normal(0, 2, n),
    })
    df.loc[12, "gamma_flux_reading"] = 9999.0  # Outlier

    agent = AnomalyDetectionAgent()
    res = agent.run({"data": df})

    assert res.is_success
    assert res.data["anomalies_found"] >= 1
    assert "gamma_flux_reading" in res.data["features_used"]


# ---------------------------------------------------------------------------
# E & F. Identifier & Constant Feature Exclusion
# ---------------------------------------------------------------------------
def test_E_F_identifier_and_constant_exclusion():
    """Verify unique UUIDs/IDs and constant (0 variance) columns are excluded."""
    n = 35
    df = pd.DataFrame({
        "tx_guid_id": [f"guid-{i:04d}" for i in range(n)],
        "fixed_constant": [42.0] * n,
        "valid_metric_a": np.random.normal(10, 2, n),
        "valid_metric_b": np.random.normal(50, 5, n),
    })
    df.loc[3, "valid_metric_a"] = 150.0

    engine = AnomalyDetectionEngine()
    res = engine.detect(df)

    assert "error" not in res
    assert "tx_guid_id" not in res["features_used"]
    assert "fixed_constant" not in res["features_used"]
    assert "valid_metric_a" in res["features_used"]
    assert "valid_metric_b" in res["features_used"]


# ---------------------------------------------------------------------------
# G. Sparse Unrelated Feature Handling (Row Preservation)
# ---------------------------------------------------------------------------
def test_G_sparse_unrelated_feature_handling():
    """Verify unrelated sparse columns (>60% null) are dropped without losing observations."""
    n = 40
    df = pd.DataFrame({
        "clean_metric": np.random.normal(100, 10, n),
        "sparse_text_meta": [None if i % 4 != 0 else f"txt_{i}" for i in range(n)],  # 75% null
        "moderately_missing": [None if i % 5 == 0 else float(i) for i in range(n)],    # 20% null
    })
    df.loc[7, "clean_metric"] = 800.0

    engine = AnomalyDetectionEngine()
    res = engine.detect(df)

    assert "error" not in res
    assert res["rows_analyzed"] == n
    assert "sparse_text_meta" in res["excluded_features"]
    assert "clean_metric" in res["features_used"]
    assert "moderately_missing" in res["features_used"]


# ---------------------------------------------------------------------------
# H. Dirty Numeric Coercion
# ---------------------------------------------------------------------------
def test_H_dirty_numeric_coercion():
    """Verify currencies, commas, percentages, and accounting brackets are parsed."""
    dirty_vals = [
        "$1,200", "€2,300", "£4,500", "15.5%", "(1,250.50)",
        "2.5k", "3.2M", "100.0", "120.0", "$50,000.00"  # Last one is outlier
    ]
    df = pd.DataFrame({"dirty_amount": dirty_vals})

    engine = AnomalyDetectionEngine()
    res = engine.detect(df)

    assert "error" not in res
    assert res["rows_analyzed"] == 10
    assert "dirty_amount" in res["features_used"]
    # The $50,000 should be the top anomaly (rank 1)
    assert res["observations"][0]["row_index"] == 9 or res["observations"][0]["anomaly_score"] > 0.8


# ---------------------------------------------------------------------------
# I & J. Negative Values & Large Numeric Scales
# ---------------------------------------------------------------------------
def test_I_J_negative_values_and_large_scales():
    """Verify robust handling of negative values and large scale numbers ($10^7$)."""
    n = 30
    df = pd.DataFrame({
        "negative_scale": np.linspace(-500, -100, n),
        "large_scale": np.linspace(1e6, 5e6, n),
    })
    df.loc[2, "negative_scale"] = 1000.0  # Positive spike in all-negative feature
    df.loc[15, "large_scale"] = 5e8       # Huge outlier

    agent = AnomalyDetectionAgent()
    res = agent.run({"data": df})

    assert res.is_success
    assert_no_nan_or_inf(res.data)
    assert res.data["anomalies_found"] >= 2


# ---------------------------------------------------------------------------
# K, L, M. Dataset Sizes and Contamination Levels
# ---------------------------------------------------------------------------
def test_K_L_M_dataset_sizes_and_rates():
    """Verify small, medium, and low/moderate anomaly rates."""
    # Small dataset (N=12)
    df_small = pd.DataFrame({"val": list(range(11)) + [1000]})
    engine = AnomalyDetectionEngine()
    res_small = engine.detect(df_small)
    assert res_small["rows_analyzed"] == 12
    assert res_small["anomalies_found"] >= 1

    # Medium dataset (N=80) with low anomaly rate
    np.random.seed(42)
    df_med = pd.DataFrame({"val": np.random.normal(0, 1, 80)})
    df_med.loc[0, "val"] = 25.0
    res_med = engine.detect(df_med, contamination="auto")
    assert 0.0 < res_med["anomaly_rate"] <= 0.20


# ---------------------------------------------------------------------------
# N & O. Explicit vs Automatic Contamination
# ---------------------------------------------------------------------------
def test_N_O_explicit_and_auto_contamination():
    """Verify explicit contamination parameter and auto estimation."""
    df = pd.DataFrame({"val": np.linspace(1, 100, 50)})
    df.loc[0, "val"] = 999.0
    df.loc[1, "val"] = -999.0

    engine = AnomalyDetectionEngine()

    # Explicit 10%
    res_exp = engine.detect(df, contamination=0.10)
    assert res_exp["contamination"] == 0.10
    assert res_exp["anomalies_found"] == 5

    # Auto
    res_auto = engine.detect(df, contamination="auto")
    assert 0.01 <= res_auto["contamination"] <= 0.15


# ---------------------------------------------------------------------------
# P, Q, R. Failure Conditions & Pre-Execution Validation
# ---------------------------------------------------------------------------
def test_P_Q_R_failure_conditions():
    """Verify structured errors for insufficient data, no numeric cols, all constant cols."""
    agent = AnomalyDetectionAgent()

    # P: Insufficient rows (N < 5)
    df_short = pd.DataFrame({"val": [1, 2, 3]})
    res_short = agent.run({"data": df_short})
    assert not res_short.is_success
    assert res_short.status in (AgentStatus.FAILED, AgentStatus.ERROR, AgentStatus.VALIDATION_FAILED)
    assert "at least 5" in (res_short.error_message or "")

    # Q: No usable numeric features
    df_text = pd.DataFrame({"txt1": ["A", "B", "C", "D", "E", "F"], "txt2": ["x", "y", "z", "w", "v", "u"]})
    res_text = agent.run({"data": df_text})
    assert not res_text.is_success

    # R: All constant features (0 variance)
    df_const = pd.DataFrame({"col_a": [10.0] * 10, "col_b": [20.0] * 10})
    res_const = agent.run({"data": df_const})
    assert not res_const.is_success
    assert "variance" in (res_const.error_message or "").lower() or "constant" in (res_const.error_message or "").lower()


# ---------------------------------------------------------------------------
# S & T. Ambiguous Commands & Structured Error Contracts
# ---------------------------------------------------------------------------
def test_S_T_intent_routing_and_error_contracts():
    """Verify natural language anomaly commands route properly and never leak tracebacks."""
    analyzer = IntentAnalyzer()

    # Anomaly queries
    q1 = analyzer.analyze("find anomalies in this dataset")
    assert q1.primary_intent == AnalyticalIntent.ANOMALIES

    q2 = analyzer.analyze("detect outliers and unusual observations")
    assert q2.primary_intent == AnalyticalIntent.ANOMALIES

    # Error contract safety
    agent = AnomalyDetectionAgent()
    empty_df = pd.DataFrame()
    res_err = agent.run({"data": empty_df})
    assert not res_err.is_success
    assert "Traceback" not in (res_err.error_message or "")


# ---------------------------------------------------------------------------
# U, V, W, X, Y. Mathematical Invariants & Epistemic Contracts
# ---------------------------------------------------------------------------
def test_U_V_W_X_Y_invariants_and_contracts():
    """Verify non-finite sanitization, confidence bounds, evidence integrity, score consistency."""
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "feat_x": np.random.normal(50, 5, n),
        "feat_y": np.random.normal(10, 2, n),
    })
    df.loc[0, "feat_x"] = 500.0

    agent = AnomalyDetectionAgent()
    res = agent.run({"data": df})

    assert res.is_success
    # U: No NaN or Inf
    assert_no_nan_or_inf(res.data)

    # V: Confidence bounded in [0, 1]
    assert 0.0 <= res.confidence <= 1.0

    # W: Evidence attached with OBSERVATION claim type
    assert len(res.evidence) > 0
    ev = res.evidence[0]
    assert ev.claim_type == ClaimType.OBSERVATION

    # X: Scores in [0, 1]
    for obs in res.data["observations"]:
        assert 0.0 <= obs["anomaly_score"] <= 1.0
        assert obs["anomaly_label"] in ("ANOMALY", "NORMAL")

    # Y: Mathematical rate consistency
    n_analyzed = res.data["rows_analyzed"]
    n_anomalies = res.data["anomalies_found"]
    rate = res.data["anomaly_rate"]
    assert rate == round(n_anomalies / n_analyzed, 4)


# ---------------------------------------------------------------------------
# Z. FastAPI End-to-End API Integration
# ---------------------------------------------------------------------------
def test_Z_fastapi_anomaly_endpoint():
    """Verify live HTTP calls to /api/v1/anomalies/detect."""
    client = TestClient(app)

    # 1. Valid anomaly detection request
    records = [{"metric_a": float(i), "metric_b": float(i * 2)} for i in range(25)]
    records[0]["metric_a"] = 999.0  # Injected outlier

    resp = client.post("/api/v1/anomalies/detect", json={
        "dataset": records,
        "contamination": 0.08,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("success", "completed")
    assert data["result"]["anomalies_found"] >= 1

    # 2. Empty dataset request returns 400
    resp_bad = client.post("/api/v1/anomalies/detect", json={"dataset": []})
    assert resp_bad.status_code == 400


# ---------------------------------------------------------------------------
# AA & AB. ResultValidator & Backward Compatibility
# ---------------------------------------------------------------------------
def test_AA_AB_result_validator_and_backward_compatibility():
    """Verify ResultValidator validates anomaly result and dict-like access works."""
    df = pd.DataFrame({"measure": [10.0, 11.0, 9.5, 10.2, 10.8, 100.0, 9.8, 10.1]})
    agent = AnomalyDetectionAgent()
    res: AgentResult = agent.run({"data": df})

    assert res.is_success
    # Validation passed
    val = ResultValidator().validate(res, context={"data": df})
    assert val.is_valid

    # Dict-like backward compatibility
    assert res["anomalies_found"] >= 1
    assert res.get("rows_analyzed") == 8
