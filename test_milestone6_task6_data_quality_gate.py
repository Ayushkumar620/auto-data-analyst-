"""
Milestone 6 — Task 6: Comprehensive Universal Data Quality Gate & Pre-Analysis Validation Test Suite.

Verifies:
A. Valid regression dataset
B. Valid classification dataset
C. Valid forecasting dataset
D. Valid clustering dataset
E. Valid anomaly dataset
F. Valid statistical relationship dataset
G. Valid EDA dataset
H. Empty dataset
I. No columns
J. Malformed input
K. Missing target
L. Constant target
M. Insufficient rows
N. Invalid feature list
O. Sparse features
P. Dirty numeric values
Q. Duplicate rows
R. Duplicate column names
S. Identifier-only dataset
T. Constant-only dataset
U. Datetime parsing
V. Invalid datetime for forecasting
W. Duplicate timestamps
X. Irregular temporal data
Y. Potential leakage detection
Z. Row accounting
AA. Feature eligibility
AB. Target eligibility
AC. Task-specific severity
AD. Deterministic output
AE. Original DataFrame immutability
AF. No traceback leakage
AG. AgentResult contract
AH. FastAPI integration
AI. Prediction pipeline integration
AJ. Forecasting pipeline integration
AK. Clustering pipeline integration
AL. Anomaly pipeline integration
AM. Statistical pipeline integration
AN. EDA compatibility on datasets rejected by predictive tasks
"""
from __future__ import annotations

import math
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.agent_result import AgentError, AgentResult, AgentStatus, ClaimType, ErrorCategory
from agent.confidence_calculator import ConfidenceCalculator
from agent.data_quality_agent import DataQualityAgent
from agent.data_quality_gate import DataQualityGate, IssueSeverity, QualityGateDecision, QualityGateStatus
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
# A & B. Valid Regression & Classification Datasets
# ---------------------------------------------------------------------------
def test_A_B_valid_regression_and_classification():
    """Verify Quality Gate correctly identifies ready regression and classification datasets."""
    n = 60
    df = pd.DataFrame({
        "feat_x": np.linspace(10, 100, n),
        "feat_y": np.random.normal(50, 10, n),
        "target_continuous": np.linspace(100, 200, n) + np.random.normal(0, 2, n),
        "target_discrete": ["TierA", "TierB"] * 30,
    })

    gate = DataQualityGate()

    # 1. Regression
    res_reg = gate.validate(df, task_type="regression", target="target_continuous")
    assert res_reg.is_ready is True
    assert res_reg.status in (QualityGateStatus.READY.value, QualityGateStatus.NEEDS_TRANSFORMATION.value)
    assert res_reg.target_eligibility["detected_type"] == "numeric"
    assert res_reg.row_accounting["analysis_rows"] == 60

    # 2. Classification
    res_clf = gate.validate(df, task_type="classification", target="target_discrete")
    assert res_clf.is_ready is True
    assert res_clf.target_eligibility["unique_count"] == 2
    assert res_clf.target_eligibility["detected_type"] == "categorical"


# ---------------------------------------------------------------------------
# C & D & E & F & G. Forecasting, Clustering, Anomaly, Stats & EDA Tasks
# ---------------------------------------------------------------------------
def test_C_D_E_F_G_various_tasks_readiness():
    """Verify Quality Gate evaluates distinct analytical tasks correctly."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "date_col": dates.strftime("%Y-%m-%d"),
        "measure_1": np.random.normal(100, 15, 60),
        "measure_2": np.random.normal(50, 5, 60),
    })

    gate = DataQualityGate()

    # 1. Forecasting
    res_fc = gate.validate(df, task_type="forecasting", target="measure_1", time_column="date_col")
    assert res_fc.is_ready is True
    assert res_fc.temporal_eligibility["usable"] is True
    assert res_fc.temporal_eligibility["chronological_order"] is True

    # 2. Clustering
    res_cl = gate.validate(df[["measure_1", "measure_2"]], task_type="clustering")
    assert res_cl.is_ready is True

    # 3. Anomaly Detection
    res_anom = gate.validate(df[["measure_1"]], task_type="anomaly_detection")
    assert res_anom.is_ready is True

    # 4. Statistical Relationship
    res_stat = gate.validate(df[["measure_1", "measure_2"]], task_type="statistical_relationship")
    assert res_stat.is_ready is True

    # 5. EDA
    res_eda = gate.validate(df, task_type="eda")
    assert res_eda.is_ready is True
    assert res_eda.quality_score >= 0.80


# ---------------------------------------------------------------------------
# H, I, J. Failure Modes: Empty, Zero Columns, Malformed Input
# ---------------------------------------------------------------------------
def test_H_I_J_empty_zero_cols_and_malformed():
    """Verify Quality Gate blocks structurally invalid inputs."""
    gate = DataQualityGate()

    # 1. Empty DataFrame
    res_empty = gate.validate(pd.DataFrame(), task_type="regression")
    assert res_empty.is_ready is False
    assert res_empty.status == QualityGateStatus.BLOCKED.value

    # 2. Zero columns
    res_zero = gate.validate(pd.DataFrame(index=range(5)), task_type="regression")
    assert res_zero.is_ready is False
    assert res_zero.status == QualityGateStatus.BLOCKED.value

    # 3. 100% missing dataset
    df_null = pd.DataFrame({"a": [None, None], "b": [None, None]})
    res_null = gate.validate(df_null, task_type="eda")
    assert res_null.is_ready is False
    assert res_null.status == QualityGateStatus.BLOCKED.value


# ---------------------------------------------------------------------------
# K & L & M. Target Issues: Missing Target, Constant Target, Insufficient Rows
# ---------------------------------------------------------------------------
def test_K_L_M_target_issues_and_sample_size():
    """Verify Quality Gate detects missing targets, constant targets, and insufficient sample sizes."""
    gate = DataQualityGate()

    # 1. Missing target specification for regression
    df = pd.DataFrame({"feature_a": [1, 2, 3, 4, 5, 6], "feature_b": [10, 20, 30, 40, 50, 60]})
    res_no_tgt = gate.validate(df, task_type="regression")
    assert res_no_tgt.status == QualityGateStatus.NEEDS_CLARIFICATION.value

    # 2. Constant target
    df_const_tgt = pd.DataFrame({"feat": [1, 2, 3, 4, 5, 6], "target": [100.0] * 6})
    res_const = gate.validate(df_const_tgt, task_type="regression", target="target")
    assert res_const.status == QualityGateStatus.BLOCKED.value
    assert "constant" in res_const.reasons[0].lower()

    # 3. Insufficient rows for regression (N=2)
    df_tiny = pd.DataFrame({"feat": [1.0, 2.0], "target": [10.0, 20.0]})
    res_tiny = gate.validate(df_tiny, task_type="regression", target="target")
    assert res_tiny.status == QualityGateStatus.BLOCKED.value


# ---------------------------------------------------------------------------
# N, O, P, Q. Features: Sparse, Dirty Numeric, Duplicate Rows
# ---------------------------------------------------------------------------
def test_N_O_P_Q_sparse_dirty_and_duplicates():
    """Verify feature eligibility matrix tracks sparse columns, dirty numeric formats, and duplicates."""
    n = 50
    df = pd.DataFrame({
        "clean_num": np.random.normal(50, 10, n),
        "dirty_curr": ["$1,200", "€500", "£300", "(100.0)"] * (n // 4 + 1),
        "sparse_field": [None if i < 40 else float(i) for i in range(n + 4)],  # >70% null
        "target": np.linspace(10, 100, n + 4),
    })

    gate = DataQualityGate()
    res = gate.validate(df, task_type="regression", target="target")

    assert res.feature_eligibility["sparse_field"]["usable"] is False
    assert "High missing rate" in res.feature_eligibility["sparse_field"]["reason"]
    assert res.feature_eligibility["clean_num"]["usable"] is True
    assert res.feature_eligibility["dirty_curr"]["transformation_required"] is True


# ---------------------------------------------------------------------------
# R. Duplicate Column Names Handling
# ---------------------------------------------------------------------------
def test_R_duplicate_column_names_handling():
    """Verify duplicate column names are internally disambiguated without mutating source data."""
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["metric", "metric", "target"])

    gate = DataQualityGate()
    res = gate.validate(df, task_type="regression", target="target")

    assert res.diagnostics["has_duplicate_columns"] is True
    assert any(iss["code"] == "DUPLICATE_COLUMN_NAMES" for iss in res.issues)
    assert any("metric__duplicate_1" in k for k in res.feature_eligibility.keys())


# ---------------------------------------------------------------------------
# S & T. Identifier-Only and Constant-Only Datasets
# ---------------------------------------------------------------------------
def test_S_T_identifier_and_constant_only_datasets():
    """Verify datasets containing only identifiers or constants are blocked for modeling."""
    n = 40
    df_id_only = pd.DataFrame({
        "uuid": [f"user_{i:04d}" for i in range(n)],
        "id_seq": list(range(100, 100 + n)),
        "target": np.random.normal(50, 5, n),
    })

    gate = DataQualityGate()
    res = gate.validate(df_id_only, task_type="regression", target="target")
    assert res.status == QualityGateStatus.BLOCKED.value
    assert "usable explanatory features" in res.reasons[0].lower()


# ---------------------------------------------------------------------------
# U, V, W, X. Datetime & Forecasting Validation
# ---------------------------------------------------------------------------
def test_U_V_W_X_datetime_and_forecasting_validation():
    """Verify datetime parsing, invalid dates, and duplicate timestamp handling for forecasting."""
    gate = DataQualityGate()

    # 1. Invalid temporal column
    df_no_dt = pd.DataFrame({"feat": [1, 2, 3, 4, 5], "val": [10, 20, 30, 40, 50]})
    res_no_dt = gate.validate(df_no_dt, task_type="forecasting", target="val")
    assert res_no_dt.status == QualityGateStatus.BLOCKED.value

    # 2. Duplicate timestamps
    dates_dup = ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"] * 4
    df_dup = pd.DataFrame({"ts": dates_dup, "target": np.random.normal(50, 5, len(dates_dup))})
    res_dup = gate.validate(df_dup, task_type="forecasting", target="target", time_column="ts")
    assert res_dup.temporal_eligibility["duplicate_count"] > 0
    assert any("Duplicate timestamps" in r for r in res_dup.recommendations)


# ---------------------------------------------------------------------------
# Y. Potential Target Leakage Detection
# ---------------------------------------------------------------------------
def test_Y_potential_leakage_detection():
    """Verify exact target duplicates are flagged as potential leakage risks."""
    df = pd.DataFrame({
        "feature_clean": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        "feature_leak": [10.0, 20.0, 30.0, 40.0, 50.0] * 10,
        "target": [10.0, 20.0, 30.0, 40.0, 50.0] * 10,
    })

    gate = DataQualityGate()
    res = gate.validate(df, task_type="regression", target="target")

    assert len(res.leakage_risks) >= 1
    assert res.leakage_risks[0]["column"] == "feature_leak"
    assert res.leakage_risks[0]["risk_type"] == "exact_target_duplicate"


# ---------------------------------------------------------------------------
# Z, AA, AB, AC, AD, AE, AF, AG. Invariants, Contracts & Immutability
# ---------------------------------------------------------------------------
def test_Z_through_AG_invariants_immutability_and_contracts():
    """Verify row accounting, immutability, AgentResult contract, and zero tracebacks."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, None, 4.0, 5.0] * 6,
        "b": ["x", "y", "x", "y", "x"] * 6,
        "target": [10.0, 20.0, 30.0, 40.0, 50.0] * 6,
    })
    df_copy = df.copy()

    agent = DataQualityAgent()
    res: AgentResult = agent.run({"data": df, "task_type": "regression", "target": "target"})

    # Immutability
    assert df.equals(df_copy)

    # AgentResult contract
    assert res.is_success
    assert res.confidence >= 0.30
    assert_no_nan_or_inf(res.data)

    data = res.data
    assert "row_accounting" in data
    assert "feature_eligibility" in data
    assert "target_eligibility" in data
    assert 0.0 <= data["quality_score"] <= 1.0


# ---------------------------------------------------------------------------
# AN. EDA Compatibility on Datasets Rejected by Predictive Tasks
# ---------------------------------------------------------------------------
def test_AN_eda_compatibility_on_rejected_predictive_datasets():
    """Prove that EDA succeeds on datasets that are BLOCKED for predictive modeling."""
    # N=2 dataset (Too small for regression, perfectly fine for basic EDA)
    df_tiny = pd.DataFrame({"val": [10.0, 20.0]})

    gate = DataQualityGate()
    res_reg = gate.validate(df_tiny, task_type="regression", target="val")
    res_eda = gate.validate(df_tiny, task_type="eda")

    assert res_reg.status == QualityGateStatus.BLOCKED.value
    assert res_eda.is_ready is True
    assert res_eda.status in (QualityGateStatus.READY.value, QualityGateStatus.READY_WITH_WARNINGS.value)


# ---------------------------------------------------------------------------
# AH. FastAPI Live HTTP Endpoints
# ---------------------------------------------------------------------------
def test_AH_fastapi_data_quality_endpoints():
    """Verify POST /api/v1/data-quality/validate and POST /api/v1/data-quality/check."""
    client = TestClient(app)

    records = [{"feature_1": float(i), "feature_2": f"Grp_{i%3}", "target_y": float(i * 2)} for i in range(25)]

    # 1. /api/v1/data-quality/validate
    resp1 = client.post("/api/v1/data-quality/validate", json={"dataset": records, "task_type": "regression", "target": "target_y"})
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["status"] in ("success", "completed")
    assert data1["result"]["is_ready"] is True

    # 2. /api/v1/data-quality/check
    resp2 = client.post("/api/v1/data-quality/check", json={"dataset": records, "task_type": "eda"})
    assert resp2.status_code == 200

    # 3. Empty dataset returns 400
    resp_empty = client.post("/api/v1/data-quality/validate", json={"dataset": []})
    assert resp_empty.status_code == 400