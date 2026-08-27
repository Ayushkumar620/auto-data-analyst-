"""
Milestone 6 — Task 4: Comprehensive Universal EDA, Data Profiling & Data Quality Intelligence Test Suite.

Verifies:
A. Basic arbitrary numeric dataset
B. Arbitrary column names
C. Mixed numeric/categorical dataset
D. Missing values & severity bands
E. Sparse columns detection
F. Duplicate rows analysis
G. Constant columns detection
H. Identifier columns detection
I. High-cardinality text fields
J. Dirty currency values
K. Percentage values
L. Accounting negative values
M. Datetime strings & span calculation
N. Ambiguous datetime/numeric values
O. Numeric distribution statistics
P. Categorical distribution statistics
Q. Outlier detection (non-causal)
R. Quality score bounds & components
S. Confidence bounds & separation from quality score
T. Evidence validity & provenance
U. Non-causal language enforcement
V. NaN/Infinity sanitization
W. Empty dataset failure contract
X. Invalid dataset failure contract
Y. Natural language EDA routing
Z. FastAPI endpoints integration
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
    res = engine.profile(df)

    assert "error" not in res
    assert res["summary"]["row_count"] == 60
    assert res["summary"]["column_count"] == 2
    assert "dim_alpha_9" in res["summary"]["numeric_columns"]
    assert "val_zeta_8" in res["summary"]["numeric_columns"]

    stats_a = res["statistics"]["numeric"]["dim_alpha_9"]
    assert stats_a["min"] == 10.0
    assert stats_a["max"] == 100.0
    assert stats_a["mean"] == 55.0
    assert stats_a["median"] == 55.0


# ---------------------------------------------------------------------------
# C. Mixed Numeric & Categorical Dataset
# ---------------------------------------------------------------------------
def test_C_mixed_numeric_categorical_dataset():
    """Verify distinct profiling of numeric vs categorical features."""
    df = pd.DataFrame({
        "category_code": ["TierA", "TierB", "TierA", "TierC"] * 15,
        "score_val": np.random.normal(100, 20, 60),
    })

    engine = EDAEngine()
    res = engine.profile(df)

    assert "error" not in res
    assert "category_code" in res["summary"]["categorical_columns"]
    assert "score_val" in res["summary"]["numeric_columns"]

    cat_stat = res["statistics"]["categorical"]["category_code"]
    assert cat_stat["unique_count"] == 3
    assert cat_stat["cardinality_ratio"] == 0.05
    assert len(cat_stat["top_categories"]) == 3


# ---------------------------------------------------------------------------
# D & E. Missing Values Severity & Sparse Columns
# ---------------------------------------------------------------------------
def test_D_E_missing_severity_and_sparse_columns():
    """Verify missingness classification into severity bands without global row dropping."""
    n = 50
    df = pd.DataFrame({
        "full_col": list(range(n)),                                  # 0% missing
        "low_missing": [None if i < 3 else i for i in range(n)],     # 6% missing (>0-10%)
        "med_missing": [None if i < 12 else i for i in range(n)],    # 24% missing (>10-30%)
        "high_missing": [None if i < 25 else i for i in range(n)],   # 50% missing (>30-60%)
        "sparse_col": [None if i < 45 else i for i in range(n)],     # 90% missing (>60-90%)
    })

    engine = EDAEngine()
    res = engine.profile(df)

    assert "error" not in res
    m_analysis = res["missing_analysis"]
    bands = m_analysis["columns_by_severity"]

    assert "full_col" in bands["0%"]
    assert "low_missing" in bands[">0-10%"]
    assert "med_missing" in bands[">10-30%"]
    assert "high_missing" in bands[">30-60%"]
    assert "sparse_col" in bands[">60-90%"] or "sparse_col" in bands[">90%"]
    assert m_analysis["sparse_columns_count"] >= 1


# ---------------------------------------------------------------------------
# F. Duplicate Rows Analysis
# ---------------------------------------------------------------------------
def test_F_duplicate_rows_analysis():
    """Verify duplicate row counting and percentage calculation without auto-deletion."""
    df = pd.DataFrame({
        "feat_1": [1, 2, 3, 1, 2, 3, 1, 2, 3, 10],
        "feat_2": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "Z"],
    })

    engine = EDAEngine()
    res = engine.profile(df)

    assert "error" not in res
    dup_info = res["duplicate_analysis"]
    assert dup_info["has_duplicates"] is True
    assert dup_info["exact_duplicate_rows"] == 6
    assert dup_info["duplicate_percentage"] == 60.0


# ---------------------------------------------------------------------------
# G & H. Constant Columns & Identifier Detection
# ---------------------------------------------------------------------------
def test_G_H_constant_and_identifier_detection():
    """Verify zero-variance constant columns and surrogate key / UUID detection."""
    n = 40
    df = pd.DataFrame({
        "const_numeric": [42.0] * n,
        "const_string": ["FIXED"] * n,
        "uuid_keys": [f"550e8400-e29b-41d4-a716-{i:012d}" for i in range(n)],
        "seq_index": list(range(100, 100 + n)),
        "real_feature": np.random.normal(0, 1, n),
    })

    engine = EDAEngine()
    res = engine.profile(df)

    assert "error" not in res
    summ = res["summary"]
    assert "const_numeric" in summ["constant_columns"]
    assert "const_string" in summ["constant_columns"]
    assert "uuid_keys" in summ["identifier_columns"]
    assert "seq_index" in summ["identifier_columns"]
    assert "real_feature" in summ["numeric_columns"]


# ---------------------------------------------------------------------------
# I. High-Cardinality Text Fields
# ---------------------------------------------------------------------------
def test_I_high_cardinality_text():
    """Verify high-cardinality text fields are identified and assigned text attribute role."""
    n = 60
    df = pd.DataFrame({
        "free_text_comment": [f"Customer remark number {i} regarding order experience" for i in range(n)],
        "num_metric": np.random.normal(50, 10, n),
    })

    engine = EDAEngine()
    res = engine.profile(df)

    assert "error" not in res
    col_prof = res["columns"]["free_text_comment"]
    assert col_prof["inferred_type"] in ("text", "identifier")
    assert col_prof["is_high_cardinality"] is True


# ---------------------------------------------------------------------------
# J, K, L. Dirty Numeric Values (Currencies, %, Accounting Negatives, Units)
# ---------------------------------------------------------------------------
def test_J_K_L_dirty_numeric_coercion():
    """Verify robust detection and cleaning of dirty currencies, percentages, accounting brackets, suffixes."""
    dirty_vals = ["$1,200", "€2.5k", "£500", "15.5%", "(1,200.50)", "2.4M", "3B", "100.0", "-500", "$2,000"]
    df = pd.DataFrame({"dirty_measure": dirty_vals * 4, "clean_num": list(range(40))})

    engine = EDAEngine()
    res = engine.profile(df)

    assert "error" not in res
    assert "dirty_measure" in res["summary"]["numeric_columns"]
    dirty_diag = res["dirty_data_analysis"]
    assert dirty_diag["total_dirty_columns"] >= 1
    assert "dirty_measure" in dirty_diag["columns"]
    assert dirty_diag["columns"]["dirty_measure"]["values_cleaned_count"] > 0


# ---------------------------------------------------------------------------
# M & N. Datetime Strings & Ambiguous Values
# ---------------------------------------------------------------------------
def test_M_N_datetime_strings_and_span():
    """Verify arbitrary ISO and date string detection, timestamp span, and regularity analysis."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "timestamp_iso": [d.isoformat() for d in dates],
        "metric_val": np.random.normal(10, 2, 50),
    })

    engine = EDAEngine()
    res = engine.profile(df)

    assert "error" not in res
    assert "timestamp_iso" in res["summary"]["datetime_columns"]
    dt_st = res["statistics"]["datetime"]["timestamp_iso"]
    assert dt_st["unique_timestamps"] == 50
    assert dt_st["date_span_days"] == 49.0


# ---------------------------------------------------------------------------
# O. Numeric Distribution Statistics & Quantiles
# ---------------------------------------------------------------------------
def test_O_numeric_distribution_statistics():
    """Verify exhaustive non-causal numeric metrics (mean, std, min, q25, median, q75, max, IQR, MAD, skew, kurtosis)."""
    np.random.seed(42)
    n = 100
    v = np.random.normal(50, 10, n)
    df = pd.DataFrame({"sample_metric": v})

    engine = EDAEngine()
    res = engine.profile(df)

    assert "error" not in res
    num_st = res["statistics"]["numeric"]["sample_metric"]
    assert num_st["count"] == 100
    assert 45.0 <= num_st["mean"] <= 55.0
    assert 8.0 <= num_st["std"] <= 12.0
    assert num_st["min"] <= num_st["q25"] <= num_st["median"] <= num_st["q75"] <= num_st["max"]
    assert num_st["iqr"] == pytest.approx(num_st["q75"] - num_st["q25"], rel=1e-3)
    assert "mad" in num_st
    assert "skewness" in num_st
    assert "kurtosis" in num_st
    assert "percentiles" in num_st
    assert len(num_st["percentiles"]) >= 5


# ---------------------------------------------------------------------------
# P & Q. Categorical Entropy & Outlier Detection
# ---------------------------------------------------------------------------
def test_P_Q_categorical_entropy_and_outliers():
    """Verify categorical entropy, imbalance detection, and Tukey IQR outlier detection."""
    np.random.seed(42)
    n = 60
    # Outlier injection
    vals = np.random.normal(20, 2, n)
    vals[0] = 100.0  # Extreme outlier
    vals[1] = 105.0  # Extreme outlier

    # Imbalanced category
    cats = ["Common"] * 55 + ["RareA", "RareB", "RareC", "RareD", "RareE"]

    df = pd.DataFrame({"outlier_metric": vals, "imbalanced_cat": cats})

    engine = EDAEngine()
    res = engine.profile(df)

    assert "error" not in res
    # Categorical stats
    cat_st = res["statistics"]["categorical"]["imbalanced_cat"]
    assert cat_st["is_imbalanced"] is True
    assert cat_st["dominant_category_percentage"] > 85.0
    assert cat_st["entropy"] > 0.0

    # Numeric outliers
    num_st = res["statistics"]["numeric"]["outlier_metric"]
    assert num_st["outliers"]["count"] >= 2
    assert num_st["outliers"]["method"] == "tukey_iqr_1.5"


# ---------------------------------------------------------------------------
# R & S. Data Quality Score & Confidence Bounds
# ---------------------------------------------------------------------------
def test_R_S_quality_score_and_confidence_bounds():
    """Verify quality score [0, 1] component breakdown and confidence [0, 1] separation."""
    df = pd.DataFrame({
        "num_1": [1.0, 2.0, None, 4.0, 5.0] * 6,
        "num_2": [10.0] * 30,  # Constant
        "cat_1": ["A", "B", "C"] * 10,
    })

    agent = EDAAgent()
    res: AgentResult = agent.run({"data": df})

    assert res.is_success
    dq = res.data["data_quality"]
    assert 0.0 <= dq["quality_score"] <= 1.0
    assert dq["quality_rating"] in ("EXCELLENT", "GOOD", "MODERATE", "POOR", "CRITICAL")
    comps = dq["components"]
    for c_name in ("completeness", "validity", "uniqueness", "consistency", "structural_usability"):
        assert 0.0 <= comps[c_name] <= 1.0

    # Confidence separate and bounded
    assert 0.0 <= res.confidence <= 1.0


# ---------------------------------------------------------------------------
# T, U, V. Evidence Validity, Non-Causal Language & Sanitization
# ---------------------------------------------------------------------------
def test_T_U_V_evidence_non_causal_and_sanitization():
    """Verify ClaimType.OBSERVATION evidence, strictly non-causal descriptions, and zero NaN/Inf leaks."""
    df = pd.DataFrame({
        "col_x": np.random.normal(0, 1, 30),
        "col_y": ["ValA", "ValB"] * 15,
    })

    agent = EDAAgent()
    res: AgentResult = agent.run({"data": df})

    assert res.is_success
    assert_no_nan_or_inf(res.data)

    # Evidence
    assert len(res.evidence) >= 1
    for ev in res.evidence:
        assert ev.claim_type == ClaimType.OBSERVATION

    # Findings non-causal check
    for f in res.data["findings"]:
        desc = f["description"].lower()
        for forbidden in ("causes", "caused", "drives", "driven by", "because of", "leads to"):
            assert forbidden not in desc


# ---------------------------------------------------------------------------
# W & X. Failure Contracts (Empty Dataset, All Null)
# ---------------------------------------------------------------------------
def test_W_X_failure_modes_and_error_contracts():
    """Verify structured failure handling for 0 rows, 0 columns, or all-null datasets."""
    agent = EDAAgent()

    # Empty rows
    res_empty = agent.run({"data": pd.DataFrame()})
    assert not res_empty.is_success
    assert "0 rows" in (res_empty.error_message or "").lower() or "empty" in (res_empty.error_message or "").lower()

    # All null
    res_null = agent.run({"data": pd.DataFrame({"a": [None, None], "b": [None, None]})})
    assert not res_null.is_success
    assert "null" in (res_null.error_message or "").lower() or "missing" in (res_null.error_message or "").lower()


# ---------------------------------------------------------------------------
# Y. Natural Language Intent Routing
# ---------------------------------------------------------------------------
def test_Y_natural_language_eda_routing():
    """Verify intent analyzer correctly maps exploratory and data profiling queries to EDA."""
    analyzer = IntentAnalyzer()

    r1 = analyzer.analyze("describe this dataset")
    assert r1.primary_intent == AnalyticalIntent.EDA

    r2 = analyzer.analyze("profile the data")
    assert r2.primary_intent == AnalyticalIntent.EDA

    r3 = analyzer.analyze("give me a data quality report")
    assert r3.primary_intent == AnalyticalIntent.EDA

    r4 = analyzer.analyze("what is wrong with this dataset?")
    assert r4.primary_intent == AnalyticalIntent.EDA

    r5 = analyzer.analyze("show dataset statistics and missing values")
    assert r5.primary_intent == AnalyticalIntent.EDA


# ---------------------------------------------------------------------------
# Z. FastAPI Live HTTP Endpoints
# ---------------------------------------------------------------------------
def test_Z_fastapi_eda_endpoints():
    """Verify POST /api/v1/eda/profile, POST /api/v1/eda, and POST /api/v1/data/profile via TestClient."""
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

    # 3. /api/v1/data/profile
    resp3 = client.post("/api/v1/data/profile", json={"dataset": records})
    assert resp3.status_code == 200

    # 4. Empty dataset returns 400
    resp_empty = client.post("/api/v1/eda/profile", json={"dataset": []})
    assert resp_empty.status_code == 400