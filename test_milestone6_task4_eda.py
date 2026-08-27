"""
Milestone 6 — Task 4: Comprehensive Universal EDA, Data Profiling & Data Quality Intelligence Test Suite.

Verifies:
A. Basic arbitrary dataset profiling
B. Arbitrary column names
C. Numeric statistics
D. Categorical profiling
E. Datetime profiling
F. Dirty currency values
G. Percentage values
H. Accounting negatives
I. Unit multipliers
J. Missing values
K. Sparse columns
L. Duplicate rows
M. Constant columns
N. Near-constant columns
O. Identifier detection
P. High-cardinality text
Q. Outlier detection
R. Skew detection
S. Invalid datetime values
T. Empty dataset
U. Malformed dataset
V. Mathematical invariants
W. Deterministic output
X. AgentResult contract
Y. No traceback leakage
Z. FastAPI endpoint integration
AA. DataFrame non-mutation & raw values preservation
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
from agent.eda_agent import EDAAgent
from agent.eda_engine import EDAEngine
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
# A & B. Basic Numeric Dataset & Arbitrary Column Names
# ---------------------------------------------------------------------------
def test_A_B_basic_numeric_and_arbitrary_names():
    """Verify EDA engine profiles arbitrary numeric datasets with custom non-keyword column names."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "dim_alpha_9": np.linspace(10, 100, n),
        "val_zeta_8": np.random.normal(50, 15, n),
    })

    engine = EDAEngine()
    res = engine.analyze(df)

    assert "error" not in res
    assert res["summary"]["row_count"] == 60
    assert res["summary"]["column_count"] == 2
    assert res["summary"]["original_rows"] == 60
    assert res["summary"]["original_columns"] == 2
    assert "dim_alpha_9" in res["summary"]["numeric_columns"]
    assert "val_zeta_8" in res["summary"]["numeric_columns"]

    stats_a = res["statistics"]["numeric"]["dim_alpha_9"]
    assert stats_a["min"] == 10.0
    assert stats_a["max"] == 100.0
    assert stats_a["mean"] == 55.0
    assert stats_a["median"] == 55.0
    assert stats_a["q1"] <= stats_a["median"] <= stats_a["q3"]


# ---------------------------------------------------------------------------
# C & D. Numeric Statistics & Categorical Profiling
# ---------------------------------------------------------------------------
def test_C_D_numeric_and_categorical_profiling():
    """Verify distinct profiling of numeric vs categorical features."""
    df = pd.DataFrame({
        "category_code": ["TierA", "TierB", "TierA", "TierC"] * 15,
        "amount_usd": np.random.exponential(100.0, 60),
    })

    engine = EDAEngine()
    res = engine.profile(df)

    assert "error" not in res
    assert "category_code" in res["summary"]["categorical_columns"]
    assert "amount_usd" in res["summary"]["numeric_columns"]

    cat_stats = res["statistics"]["categorical"]["category_code"]
    assert cat_stats["unique_count"] == 3
    assert len(cat_stats["top_categories"]) == 3
    assert cat_stats["dominant_category_percentage"] == 50.0  # TierA is 30/60 = 50%
    assert cat_stats["entropy"] > 0.0
    assert "top_values" in cat_stats
    assert "top_value_counts" in cat_stats


# ---------------------------------------------------------------------------
# E. Datetime Profiling
# ---------------------------------------------------------------------------
def test_E_datetime_profiling():
    """Verify datetime parsing, span calculation, and frequency detection."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "event_timestamp": dates.strftime("%Y-%m-%d"),
        "reading": np.random.normal(20, 2, 100),
    })

    engine = EDAEngine()
    res = engine.analyze(df)

    assert "error" not in res
    assert "event_timestamp" in res["summary"]["datetime_columns"]

    dt_stats = res["statistics"]["datetime"]["event_timestamp"]
    assert dt_stats["unique_timestamps"] == 100
    assert dt_stats["duplicate_timestamps"] == 0
    assert dt_stats["date_span_days"] == 99.0
    assert dt_stats["inferred_frequency"] in ("D", "irregular")
    assert "min_date" in dt_stats
    assert "max_date" in dt_stats


# ---------------------------------------------------------------------------
# F, G, H, I. Dirty Numeric Coercion (Currencies, %, Parens, Unit Multipliers)
# ---------------------------------------------------------------------------
def test_F_G_H_I_dirty_numeric_coercion_types():
    """Verify universal coercion of dirty currencies, percentages, accounting brackets, and unit multipliers."""
    dirty_vals = ["$1,200", "€1,200", "£1,200", "15%", "(1,200.50)", "1.5k", "2M", "3B", " 500 ", "100.0"]
    df = pd.DataFrame({"dirty_measure": dirty_vals * 5})

    engine = EDAEngine()
    res = engine.profile(df)

    assert "error" not in res
    assert "dirty_measure" in res["summary"]["numeric_columns"]

    dirty_summary = res["dirty_data_analysis"]
    assert dirty_summary["total_dirty_columns"] >= 1
    assert "dirty_measure" in dirty_summary["columns"]

    dc = dirty_summary["columns"]["dirty_measure"]
    assert dc["coercion_success_rate"] >= 0.90
    assert dc["values_cleaned_count"] > 0
    assert dc["transformation_applied"] is not None


# ---------------------------------------------------------------------------
# J & K. Missing Values & Sparse Columns Detection
# ---------------------------------------------------------------------------
def test_J_K_missing_data_and_sparse_columns():
    """Verify missing data severity bands without destructive row dropping."""
    n = 100
    df = pd.DataFrame({
        "clean_col": [1.0] * n,
        "low_missing": [None if i < 5 else float(i) for i in range(n)],     # 5%
        "mid_missing": [None if i < 25 else float(i) for i in range(n)],    # 25%
        "high_missing": [None if i < 50 else float(i) for i in range(n)],   # 50%
        "sparse_col": [None if i < 80 else float(i) for i in range(n)],     # 80%
        "extreme_col": [None if i < 95 else float(i) for i in range(n)],    # 95%
    })

    engine = EDAEngine()
    res = engine.analyze(df)

    assert "error" not in res
    missing_analysis = res["missing_analysis"]
    bands = missing_analysis["columns_by_severity"]

    assert "clean_col" in bands["0%"]
    assert "low_missing" in bands[">0-10%"]
    assert "mid_missing" in bands[">10-30%"]
    assert "high_missing" in bands[">30-60%"]
    assert "sparse_col" in bands[">60-90%"]
    assert "extreme_col" in bands[">90%"]
    assert "sparse_col" in res["summary"]["sparse_columns"]


# ---------------------------------------------------------------------------
# L. Duplicate Rows Analysis
# ---------------------------------------------------------------------------
def test_L_duplicate_rows_analysis():
    """Verify exact duplicate row tracking and duplicate percentage calculation."""
    df = pd.DataFrame({
        "a": [1, 1, 1, 2, 3],
        "b": ["x", "x", "x", "y", "z"],
    })

    engine = EDAEngine()
    res = engine.analyze(df)

    assert "error" not in res
    dup_analysis = res["duplicate_analysis"]
    assert dup_analysis["exact_duplicate_rows"] == 2
    assert dup_analysis["duplicate_percentage"] == 40.0
    assert dup_analysis["has_duplicates"] is True
    assert dup_analysis["unique_rows_count"] == 3


# ---------------------------------------------------------------------------
# M & N. Constant & Near-Constant Columns Detection
# ---------------------------------------------------------------------------
def test_M_N_constant_and_near_constant_columns():
    """Verify zero-variance constant and near-constant columns are detected."""
    df = pd.DataFrame({
        "const_num": [42.0] * 50,
        "const_str": ["FIXED"] * 50,
        "near_const": ["MAIN"] * 48 + ["RARE", "RARE2"],  # 96% dominant
        "normal_feat": np.random.normal(0, 1, 50),
    })

    engine = EDAEngine()
    res = engine.analyze(df)

    assert "error" not in res
    assert "const_num" in res["summary"]["constant_columns"]
    assert "const_str" in res["summary"]["constant_columns"]
    assert res["columns"]["near_const"]["is_near_constant"] is True
    assert res["columns"]["near_const"]["near_constant"] is True


# ---------------------------------------------------------------------------
# O. Statistical Identifier Detection
# ---------------------------------------------------------------------------
def test_O_identifier_detection():
    """Verify 100% unique sequence and string identifiers are recognized as keys."""
    n = 60
    df = pd.DataFrame({
        "id_seq": list(range(1001, 1001 + n)),
        "uuid_str": [f"user_{i:04d}_alpha" for i in range(n)],
        "metric_val": np.random.normal(50, 10, n),
    })

    engine = EDAEngine()
    res = engine.profile(df)

    assert "error" not in res
    assert "id_seq" in res["summary"]["identifier_columns"]
    assert "uuid_str" in res["summary"]["identifier_columns"]
    assert res["columns"]["id_seq"]["semantic_role"] == "key"
    assert res["columns"]["uuid_str"]["semantic_role"] == "key"


# ---------------------------------------------------------------------------
# P. High-Cardinality Text Fields
# ---------------------------------------------------------------------------
def test_P_high_cardinality_text_fields():
    """Verify free-form multi-word text is categorized as text rather than identifier."""
    n = 60
    texts = [f"Customer reported issue with billing system regarding transaction #{i} on portal" for i in range(n)]
    df = pd.DataFrame({
        "feedback_text": texts,
        "rating": np.random.choice([1, 2, 3, 4, 5], n),
    })

    engine = EDAEngine()
    res = engine.analyze(df)

    assert "error" not in res
    assert "feedback_text" in res["summary"]["text_columns"]
    assert res["columns"]["feedback_text"]["semantic_role"] == "attribute"
    assert "text_stats" in res["columns"]["feedback_text"]
    txt_st = res["columns"]["feedback_text"]["text_stats"]
    assert txt_st["average_length"] > 40
    assert txt_st["min_length"] > 0


# ---------------------------------------------------------------------------
# Q & R. Outlier & Skew Detection
# ---------------------------------------------------------------------------
def test_Q_R_outlier_and_skew_detection():
    """Verify non-causal outlier detection using Tukey IQR and skewness diagnostics."""
    np.random.seed(42)
    # Normal distribution with 3 extreme outliers
    v = np.random.normal(50, 5, 100)
    v[0] = 500.0   # Extreme high
    v[1] = -400.0  # Extreme low
    v[2] = 450.0   # Extreme high
    df = pd.DataFrame({"sensor_reading": v})

    engine = EDAEngine()
    res = engine.profile(df)

    assert "error" not in res
    num_st = res["statistics"]["numeric"]["sensor_reading"]
    assert num_st["outliers"]["count"] >= 3
    assert num_st["outlier_count"] >= 3
    assert num_st["outliers"]["method"] == "tukey_iqr_1.5"
    assert num_st["skewness"] != 0.0
    assert "robust_zscore_outliers" in num_st
    assert "histogram" in num_st


# ---------------------------------------------------------------------------
# S. Ambiguous / Invalid Datetime Values
# ---------------------------------------------------------------------------
def test_S_ambiguous_datetime_values():
    """Verify numeric integer quantities are not incorrectly parsed as datetimes."""
    df = pd.DataFrame({
        "item_quantity": [1, 2, 5, 10, 20, 50, 100] * 10,
        "price_usd": [10.5, 20.0, 5.0, 99.9, 150.0, 12.0, 8.0] * 10,
    })

    engine = EDAEngine()
    res = engine.analyze(df)

    assert "error" not in res
    assert "item_quantity" in res["summary"]["numeric_columns"]
    assert "item_quantity" not in res["summary"]["datetime_columns"]


# ---------------------------------------------------------------------------
# T & U. Failure Modes: Empty & Malformed Datasets
# ---------------------------------------------------------------------------
def test_T_U_failure_modes_and_validation():
    """Verify structured AgentError returns for empty, zero-column, or all-null DataFrames."""
    agent = EDAAgent()

    # 1. Empty DataFrame
    res_empty = agent.run({"data": pd.DataFrame()})
    assert not res_empty.is_success
    assert res_empty.status in (AgentStatus.ERROR, AgentStatus.VALIDATION_FAILED)

    # 2. All-null DataFrame
    df_null = pd.DataFrame({"a": [None, None], "b": [None, None]})
    res_null = agent.run({"data": df_null})
    assert not res_null.is_success
    assert "null" in (res_null.error_message or "").lower() or "missing" in (res_null.error_message or "").lower()


# ---------------------------------------------------------------------------
# V & W & X & Y. Invariants, Determinism, AgentResult & Non-Causal Language
# ---------------------------------------------------------------------------
def test_V_W_X_Y_invariants_determinism_and_agent_result():
    """Verify mathematical bounds [0, 1], zero NaN/Inf leakage, and deterministic outputs."""
    np.random.seed(42)
    df = pd.DataFrame({
        "metric_x": np.random.normal(100, 20, 50),
        "grp_y": ["Alpha", "Beta"] * 25,
    })

    agent = EDAAgent()
    res1: AgentResult = agent.run({"data": df})
    res2: AgentResult = agent.run({"data": df})

    assert res1.is_success
    assert res1.confidence >= 0.30
    assert_no_nan_or_inf(res1.data)

    # Quality score bounds
    dq = res1.data["data_quality"]
    assert 0.0 <= dq["quality_score"] <= 1.0
    for comp_name, comp_val in dq["components"].items():
        assert 0.0 <= comp_val <= 1.0, f"Component {comp_name} out of bounds: {comp_val}"

    # Column quality scores
    for col_name, col_data in res1.data["columns"].items():
        assert 0.0 <= col_data["quality_score"] <= 1.0

    # Determinism
    assert res1.data["summary"] == res2.data["summary"]

    # Non-causal wording
    for rec in res1.data.get("recommendations", []):
        for forbidden in ("causes", "drives", "because of", "results in"):
            assert forbidden not in rec.lower()


# ---------------------------------------------------------------------------
# AA. DataFrame Non-Mutation & Raw Values Preservation
# ---------------------------------------------------------------------------
def test_AA_dataframe_non_mutation():
    """Verify input DataFrame is not mutated in-place and raw string formatting is preserved."""
    dirty_vals = ["$100.00", "$200.00", "$300.00"] * 10
    original_series = pd.Series(dirty_vals, name="raw_revenue")
    df = pd.DataFrame({"raw_revenue": original_series.copy()})

    engine = EDAEngine()
    res = engine.analyze(df)

    assert "error" not in res
    # Ensure source DataFrame was NOT overwritten in-place
    assert (df["raw_revenue"] == original_series).all()
    assert df["raw_revenue"].dtype == object
    assert pd.api.types.is_string_dtype(df["raw_revenue"]) or df["raw_revenue"].dtype == object
    assert df["raw_revenue"].iloc[0] == "$100.00"


# ---------------------------------------------------------------------------
# Z. FastAPI Live HTTP Endpoints
# ---------------------------------------------------------------------------
def test_Z_fastapi_eda_endpoints():
    """Verify POST /api/v1/eda/profile, POST /api/v1/eda, POST /api/v1/eda/run, and POST /api/v1/data/profile via TestClient."""
    client = TestClient(app)

    records = [{"feature_a": float(i), "feature_b": f"Group_{i%3}"} for i in range(25)]

    # 1. /api/v1/eda/profile
    resp1 = client.post("/api/v1/eda/profile", json={"dataset": records})
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["status"] in ("success", "completed")
    assert data1["result"]["summary"]["row_count"] == 25

    # 2. /api/v1/eda
    resp2 = client.post("/api/v1/eda", json={"dataset": records})
    assert resp2.status_code == 200

    # 3. /api/v1/eda/run
    resp_run = client.post("/api/v1/eda/run", json={"dataset": records})
    assert resp_run.status_code == 200

    # 4. /api/v1/data/profile
    resp3 = client.post("/api/v1/data/profile", json={"dataset": records})
    assert resp3.status_code == 200

    # 5. Empty dataset returns 400
    resp_empty = client.post("/api/v1/eda/profile", json={"dataset": []})
    assert resp_empty.status_code == 400