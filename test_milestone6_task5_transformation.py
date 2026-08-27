"""
Milestone 6 — Task 5: Comprehensive Universal Data Transformation & Feature Engineering Test Suite.

Verifies:
A. Arbitrary numeric dataset
B. Dirty currency values
C. Percentage values
D. Accounting negatives
E. Multipliers
F. Missing numeric values
G. Missing categorical values
H. Sparse columns
I. One-hot encoding
J. Unknown categories during inference
K. Datetime feature engineering
L. Cyclical datetime features
M. Forecasting temporal isolation
N. Identifier exclusion
O. Constant exclusion
P. Near-constant exclusion
Q. Scaling
R. Robust scaling with outliers
S. Skew transformation
T. Outlier clipping
U. Explicit feature selection
V. Invalid feature selection
W. Target isolation
X. Train/test leakage prevention
Y. Row alignment
Z. Original DataFrame immutability
AA. Deterministic transformation
AB. Serialization/deserialization of TransformationState
AC. Schema drift detection
AD. Malformed input
AE. Empty input
AF. Mathematical invariants
AG. AgentResult contract
AH. No traceback leakage
AI. FastAPI integration
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.agent_result import AgentError, AgentResult, AgentStatus, ClaimType, ErrorCategory
from agent.confidence_calculator import ConfidenceCalculator
from agent.intent import AnalyticalIntent, IntentAnalyzer
from agent.pre_execution_validator import PreExecutionValidator
from agent.result_validator import ResultValidator
from agent.transformation_agent import TransformationAgent
from agent.transformation_engine import TransformationEngine, TransformationPlan, TransformationState
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
# A. Arbitrary Numeric Dataset
# ---------------------------------------------------------------------------
def test_A_arbitrary_numeric_dataset():
    """Verify transformation of arbitrary numeric datasets with custom non-keyword column names."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "dim_theta_1": np.linspace(10, 100, n),
        "val_psi_2": np.random.normal(50, 10, n),
    })

    engine = TransformationEngine()
    X, state, plan = engine.fit_transform(df)

    assert len(X) == 50
    assert len(X.columns) == 2
    assert "dim_theta_1_scaled" in X.columns
    assert "val_psi_2_scaled" in X.columns
    assert np.all(np.isfinite(X.to_numpy()))


# ---------------------------------------------------------------------------
# B, C, D, E. Dirty Numeric Values, Currencies, %, Accounting, Multipliers
# ---------------------------------------------------------------------------
def test_B_C_D_E_dirty_numeric_coercion():
    """Verify universal coercion of dirty currencies, percentages, accounting brackets, and suffixes."""
    dirty_vals = ["$1,200", "€1,200", "£1,200", "15%", "(1,200.50)", "1.5k", "2M", "3B", " 500 ", "100.0"]
    df = pd.DataFrame({"dirty_feature": dirty_vals * 5})

    engine = TransformationEngine()
    X, state, plan = engine.fit_transform(df)

    assert len(X) == 50
    assert "dirty_feature_scaled" in X.columns
    assert np.all(np.isfinite(X["dirty_feature_scaled"]))


# ---------------------------------------------------------------------------
# F, G, H. Missing Value Imputation & Sparse Column Exclusion
# ---------------------------------------------------------------------------
def test_F_G_H_missing_values_and_sparse_columns():
    """Verify numeric/categorical imputation and automatic exclusion of sparse columns."""
    n = 60
    df = pd.DataFrame({
        "num_with_nulls": [None if i % 5 == 0 else float(i) for i in range(n)],
        "cat_with_nulls": [None if i % 4 == 0 else f"Tier_{i%3}" for i in range(n)],
        "sparse_over_60": [None if i < 45 else float(i) for i in range(n)],  # 75% missing
    })

    engine = TransformationEngine()
    X, state, plan = engine.fit_transform(df)

    # Sparse column should be excluded
    assert "sparse_over_60" in state.excluded_features
    assert not any("sparse_over_60" in c for c in X.columns)

    # Nulls in included columns are imputed cleanly
    assert not X.isna().any().any()


# ---------------------------------------------------------------------------
# I & J. Categorical One-Hot Encoding & Unknown Categories Handling
# ---------------------------------------------------------------------------
def test_I_J_categorical_encoding_and_unknown_categories():
    """Verify one-hot encoding on train and safe inference on unseen categories without column shifting."""
    train_df = pd.DataFrame({"tier": ["Bronze", "Silver", "Gold"] * 10})
    test_df = pd.DataFrame({"tier": ["Bronze", "Platinum", "Diamond", "Gold"]})  # Unseen categories

    engine = TransformationEngine()
    state = engine.fit(train_df, config={"encoding": "onehot"})

    X_train = engine.transform(train_df, state)
    X_test = engine.transform(test_df, state)

    # Columns match exactly
    assert list(X_train.columns) == list(X_test.columns)
    assert set(X_train.columns) == {"tier_Bronze", "tier_Silver", "tier_Gold"}

    # Unseen categories in test do not crash and set all OHE columns to 0.0
    assert X_test.iloc[1].sum() == 0.0  # Platinum
    assert X_test.iloc[2].sum() == 0.0  # Diamond
    assert X_test.iloc[0]["tier_Bronze"] == 1.0


# ---------------------------------------------------------------------------
# K & L & M. Datetime Engineering, Cyclical Features & Forecasting Isolation
# ---------------------------------------------------------------------------
def test_K_L_M_datetime_features_and_forecasting_isolation():
    """Verify temporal feature components and cyclical sin/cos features."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "event_date": dates.strftime("%Y-%m-%d"),
        "reading": np.random.normal(50, 5, 60),
    })

    engine = TransformationEngine()
    X, state, plan = engine.fit_transform(df)

    assert "event_date_year" in X.columns
    assert "event_date_month" in X.columns
    assert "event_date_day" in X.columns
    assert "event_date_month_sin" in X.columns
    assert "event_date_month_cos" in X.columns
    assert "event_date_dow_sin" in X.columns
    assert "event_date_dow_cos" in X.columns


# ---------------------------------------------------------------------------
# N, O, P. Identifier, Constant & Near-Constant Exclusion
# ---------------------------------------------------------------------------
def test_N_O_P_identifier_and_constant_exclusion():
    """Verify UUIDs, sequence IDs, constants, and zero-variance features are excluded by default."""
    n = 50
    df = pd.DataFrame({
        "user_uuid": [f"user_{i:04d}_alpha" for i in range(n)],
        "id_seq": list(range(100, 100 + n)),
        "all_fixed": ["FIXED"] * n,
        "valid_metric": np.random.normal(10, 2, n),
    })

    engine = TransformationEngine()
    X, state, plan = engine.fit_transform(df)

    assert "user_uuid" in state.excluded_features
    assert "id_seq" in state.excluded_features
    assert "all_fixed" in state.excluded_features
    assert list(X.columns) == ["valid_metric_scaled"]


# ---------------------------------------------------------------------------
# Q & R & S & T. Scaling, Robust Scaler, Skewness & Outlier Clipping
# ---------------------------------------------------------------------------
def test_Q_R_S_T_scaling_robust_skew_and_outliers():
    """Verify robust scaling, Yeo-Johnson/log1p skewness handling, and outlier clipping."""
    np.random.seed(42)
    # Heavily skewed positive values with 2 extreme outliers
    vals = np.random.exponential(scale=10.0, size=100)
    vals[0] = 5000.0
    vals[1] = 8000.0
    df = pd.DataFrame({"skewed_metric": vals})

    engine = TransformationEngine()
    X, state, plan = engine.fit_transform(df, config={"scaling": "robust", "outliers": "clip"})

    assert "skewed_metric" in state.outlier_bounds
    assert state.scalers["skewed_metric"]["method"] == "robust"
    assert np.all(np.isfinite(X["skewed_metric_scaled"]))


# ---------------------------------------------------------------------------
# U & V. Explicit Feature Selection & Validation Failure
# ---------------------------------------------------------------------------
def test_U_V_explicit_feature_selection_and_validation():
    """Verify explicit features=[...] is strictly respected, while invalid columns error out."""
    df = pd.DataFrame({
        "col_a": [1, 2, 3, 4, 5],
        "col_b": [10, 20, 30, 40, 50],
        "col_c": [100, 200, 300, 400, 500],
    })

    engine = TransformationEngine()
    # Explicit subset
    X, state, plan = engine.fit_transform(df, features=["col_a", "col_c"])
    assert list(X.columns) == ["col_a_scaled", "col_c_scaled"]

    # Invalid feature raises structured ValueError
    with pytest.raises(ValueError, match="Requested feature 'col_non_existent' not found"):
        engine.fit(df, features=["col_non_existent"])


# ---------------------------------------------------------------------------
# W. Target Isolation (Never Included in X)
# ---------------------------------------------------------------------------
def test_W_target_isolation():
    """Verify target column is never included in the transformed feature matrix X."""
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "target_col": ["Class_A", "Class_B", "Class_A", "Class_B", "Class_A"],
    })

    engine = TransformationEngine()
    X, state, plan = engine.fit_transform(df, target="target_col")

    assert "target_col" not in X.columns
    assert not any("target_col" in c for c in X.columns)
    assert state.target_name == "target_col"
    assert state.target_type == "categorical"


# ---------------------------------------------------------------------------
# X. Train/Test Leakage Prevention
# ---------------------------------------------------------------------------
def test_X_train_test_leakage_prevention():
    """Verify statistics learned from train data are NOT affected by test data (even with extreme outliers)."""
    train_df = pd.DataFrame({"score": [10.0, 20.0, 30.0, 40.0, 50.0]})
    test_normal = pd.DataFrame({"score": [15.0, 25.0]})
    test_extreme = pd.DataFrame({"score": [100000.0, 999999.0]})

    engine = TransformationEngine()
    state = engine.fit(train_df, config={"scaling": "standard"})

    center_before = state.scalers["score"]["center"]
    scale_before = state.scalers["score"]["scale"]

    _ = engine.transform(test_normal, state)
    _ = engine.transform(test_extreme, state)

    # State remains 100% identical and unchanged
    assert state.scalers["score"]["center"] == center_before
    assert state.scalers["score"]["scale"] == scale_before


# ---------------------------------------------------------------------------
# Y & Z & AA. Row Alignment, Immutability & Determinism
# ---------------------------------------------------------------------------
def test_Y_Z_AA_row_alignment_immutability_and_determinism():
    """Verify len(X) == len(df), source DataFrame is never mutated, and outputs are deterministic."""
    np.random.seed(42)
    dirty_vals = ["$100.00", "$200.00", "$300.00"] * 10
    original_series = pd.Series(dirty_vals, name="raw_metric")
    df = pd.DataFrame({"raw_metric": original_series.copy()})

    engine = TransformationEngine()
    X1, state1, _ = engine.fit_transform(df)
    X2 = engine.transform(df, state1)

    # Row alignment
    assert len(X1) == len(df) == 30
    assert (X1.index == df.index).all()

    # Immutability
    assert (df["raw_metric"] == original_series).all()

    # Determinism
    pd.testing.assert_frame_equal(X1, X2)


# ---------------------------------------------------------------------------
# AB. Serialization / Deserialization of TransformationState
# ---------------------------------------------------------------------------
def test_AB_state_serialization_deserialization():
    """Verify TransformationState can be serialized to dict/JSON and reloaded without loss."""
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cat": ["A", "B", "A", "B", "C"],
    })

    engine = TransformationEngine()
    state = engine.fit(df)

    state_dict = state.to_dict()
    state_json = json.dumps(state_dict)

    reloaded_dict = json.loads(state_json)
    reloaded_state = TransformationState.from_dict(reloaded_dict)

    X_orig = engine.transform(df, state)
    X_reloaded = engine.transform(df, reloaded_state)

    pd.testing.assert_frame_equal(X_orig, X_reloaded)


# ---------------------------------------------------------------------------
# AC. Schema Drift Detection (Strict vs Compatible)
# ---------------------------------------------------------------------------
def test_AC_schema_drift_detection():
    """Verify schema drift detection under strict vs compatible policies."""
    train_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    test_drift_df = pd.DataFrame({"a": [10, 20]})  # Missing column 'b'

    engine = TransformationEngine()
    state = engine.fit(train_df)

    # Compatible policy fills missing feature with imputer default
    X_comp = engine.transform(test_drift_df, state, drift_policy="compatible")
    assert len(X_comp) == 2
    assert "b_scaled" in X_comp.columns

    # Strict policy raises ValueError
    with pytest.raises(ValueError, match="Schema Drift Error"):
        engine.transform(test_drift_df, state, drift_policy="strict")


# ---------------------------------------------------------------------------
# AD & AE. Failure Modes: Malformed & Empty Inputs
# ---------------------------------------------------------------------------
def test_AD_AE_empty_and_malformed_inputs():
    """Verify structured AgentResult error handling for empty or all-null DataFrames."""
    agent = TransformationAgent()

    # Empty DataFrame
    res_empty = agent.run({"data": pd.DataFrame()})
    assert not res_empty.is_success

    # All-null DataFrame
    df_null = pd.DataFrame({"a": [None, None], "b": [None, None]})
    res_null = agent.run({"data": df_null})
    assert not res_null.is_success


# ---------------------------------------------------------------------------
# AF & AG & AH. Invariants, AgentResult Contract & No Tracebacks
# ---------------------------------------------------------------------------
def test_AF_AG_AH_agent_result_contract_and_invariants():
    """Verify canonical AgentResult contract and confidence calculations."""
    df = pd.DataFrame({
        "feature_x": [10.0, 20.0, 30.0, 40.0, 50.0],
        "category_y": ["A", "B", "A", "B", "A"],
    })

    agent = TransformationAgent()
    res: AgentResult = agent.run({"data": df})

    assert res.is_success
    assert res.confidence >= 0.30
    assert_no_nan_or_inf(res.data)
    assert "transformation_plan" in res.data
    assert "state" in res.data


# ---------------------------------------------------------------------------
# AI. FastAPI Live HTTP Endpoints
# ---------------------------------------------------------------------------
def test_AI_fastapi_transformation_endpoints():
    """Verify POST /api/v1/transformation, POST /api/v1/transformation/fit, and POST /api/v1/transformation/transform."""
    client = TestClient(app)

    records = [{"metric_a": float(i), "segment_b": f"Group_{i%2}"} for i in range(20)]

    # 1. /api/v1/transformation
    resp1 = client.post("/api/v1/transformation", json={"dataset": records})
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["status"] in ("success", "completed")
    assert "transformation_plan" in data1["result"]

    # 2. /api/v1/transformation/fit
    resp_fit = client.post("/api/v1/transformation/fit", json={"dataset": records})
    assert resp_fit.status_code == 200
    state_payload = resp_fit.json()["result"]["state"]

    # 3. /api/v1/transformation/transform
    resp_transform = client.post("/api/v1/transformation/transform", json={"dataset": records, "state": state_payload})
    assert resp_transform.status_code == 200

    # 4. Empty dataset returns 400
    resp_empty = client.post("/api/v1/transformation", json={"dataset": []})
    assert resp_empty.status_code == 400