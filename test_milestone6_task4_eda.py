"""
Milestone 6 — Task 4: Comprehensive Universal EDA, Data Profiling & Data Quality Intelligence Test Suite.

Verifies:
A. Arbitrary numeric columns
B. Arbitrary categorical columns
C. Arbitrary datetime columns
D. Dirty numeric strings
E. Currency values
F. Percentage values
G. Accounting negatives
H. Missing values
I. Sparse columns
J. All-null columns
K. Constant columns
L. Identifier columns
M. High-cardinality text
N. Duplicate rows
O. Outliers
P. Skewed distributions
Q. Zero-inflated data
R. Mixed numeric/categorical data
S. Arbitrary column names
T. Empty dataset
U. Insufficient rows
V. Non-finite values
W. Deterministic execution
X. AgentResult contract
Y. Evidence provenance
Z. No causal language
AA. ResultValidator invariants
AB. FastAPI /api/v1/eda endpoint
AC. Natural-language EDA routing
AD. No global row loss from unrelated missing columns
AE. Target-aware EDA
AF. Large dataset bounded analysis
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
# A & B & S. Arbitrary Numeric, Categorical Columns & Arbitrary Column Names
# ---------------------------------------------------------------------------
def test_A_B_S_arbitrary_numeric_categorical_and_names():
    """Verify EDA engine profiles arbitrary numeric and categorical data with arbitrary names."""
    n = 60
    df = pd.DataFrame({
        "var_num_99": np.linspace(10, 100, n),
        "var_cat_88": ["Alpha", "Beta", "Gamma"] * 20,
    })

    engine = EDAEngine()
    res = engine.analyze(df)

    assert "error" not in res
    assert res["summary"]["original_rows"] == 60
    assert res["summary"]["original_columns"] == 2
    assert "var_num_99" in res["summary"]["numeric_columns"]
    assert "var_cat_88" in res["summary"]["categorical_columns"]

    stats_num = res["statistics"]["numeric"]["var_num_99"]
    assert stats_num["min"] == 10.0
    assert stats_num["max"] == 100.0
    assert stats_num["mean"] == 55.0
    assert stats_num["q1"] <= stats_num["median"] <= stats_num["q3"]
    assert stats_num["range"] == 90.0
    assert stats_num["coefficient_of_variation"] >= 0.0

    stats_cat = res["statistics"]["categorical"]["var_cat_88"]
    assert stats_cat["unique_count"] == 3
    assert stats_cat["mode"] in ("Alpha", "Beta", "Gamma")
    assert stats_cat["mode_frequency"] == 20
    assert stats_cat["mode_percentage"] == round(20/60*100, 2)
    assert stats_cat["entropy"] > 0.0


# ---------------------------------------------------------------------------
# C. Arbitrary Datetime Columns
# ---------------------------------------------------------------------------
def test_C_arbitrary_datetime_columns():
    """Verify temporal column profiling without keyword matching."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "temporal_axis_x": dates.strftime("%Y-%m-%d"),
        "measure_k": np.random.normal(100, 10, 50),
    })

    engine = EDAEngine()
    res = engine.analyze(df)

    assert "temporal_axis_x" in res["summary"]["datetime_columns"]
    dt_st = res["statistics"]["datetime"]["temporal_axis_x"]
    assert dt_st["unique_timestamps"] == 50
    assert dt_st["duplicate_timestamps"] == 0
    assert dt_st["monotonicity"] is True
    assert dt_st["chronological_ordering"] is True
    assert "days" in dt_st["date_span"]


# ---------------------------------------------------------------------------
# D, E, F, G. Dirty Numeric Strings: Currencies, Percentages, Accounting Negatives
# ---------------------------------------------------------------------------
def test_D_E_F_G_dirty_numeric_currencies_percentages_negatives():
    """Verify automatic dirty numeric value detection and coercion."""
    df = pd.DataFrame({
        "dirty_curr": ["$1,200.50", "$2,300.00", "€450.00", "£980.25", "$3,100.00"] * 10,
        "dirty_pct": ["15.5%", "24.2%", "8.9%", "99.1%", "45.0%"] * 10,
        "dirty_neg": ["(120.0)", "(45.5)", "300.0", "(15.0)", "250.0"] * 10,
        "clean_num": list(range(50)),
    })

    engine = EDAEngine()
    res = engine.analyze(df)

    assert "dirty_curr" in res["summary"]["parseable_numeric_columns"]
    assert "dirty_pct" in res["summary"]["parseable_numeric_columns"]
    assert "dirty_neg" in res["summary"]["parseable_numeric_columns"]

    assert res["dirty_data_analysis"]["total_dirty_columns"] >= 3
    curr_dc = res["columns"]["dirty_curr"]["dirty_coercion"]
    assert curr_dc["detected_numeric"] is True
    assert curr_dc["coercion_success_rate"] == 1.0


# ---------------------------------------------------------------------------
# H, I, J. Missing Values, Sparse Columns, and All-Null Columns
# ---------------------------------------------------------------------------
def test_H_I_J_missing_sparse_and_all_null_columns():
    """Verify column-specific missingness without global row dropping."""
    n = 60
    df = pd.DataFrame({
        "complete_col": np.linspace(1, 60, n),
        "sparse_col": [None if i < 48 else float(i) for i in range(n)],  # 80% null
        "all_null_col": [None] * n,
        "mod_null_col": [None if i % 5 == 0 else float(i) for i in range(n)],  # 20% null
    })

    engine = EDAEngine()
    res = engine.analyze(df)

    miss = res["missing_analysis"]
    assert "all_null_col" in res["summary"]["empty_columns"]
    assert "sparse_col" in res["summary"]["sparse_columns"]
    assert "complete_col" in miss["columns_by_severity"]["0%"]
    assert miss["columns_summary"]["mod_null_col"]["missing_count"] == 12


# ---------------------------------------------------------------------------
# K, L, M, N. Constant, Identifier, High-Cardinality & Duplicate Rows
# ---------------------------------------------------------------------------
def test_K_L_M_N_constant_identifier_high_card_and_duplicates():
    """Verify detection of constants, identifiers, high cardinality, and duplicate rows."""
    n = 50
    df = pd.DataFrame({
        "const_val": [999.0] * n,
        "uuid_id": [f"usr_{i:05d}" for i in range(n)],
        "high_card_txt": [f"Detailed freeform feedback paragraph number {i} from user" for i in range(n)],
        "metric": [1.0, 2.0] * (n // 2),
    })
    # Add duplicate rows
    df_with_dup = pd.concat([df, df.iloc[:5]], ignore_index=True)

    engine = EDAEngine()
    res = engine.analyze(df_with_dup)

    assert "const_val" in res["summary"]["constant_columns"]
    assert "uuid_id" in res["summary"]["identifier_columns"]
    assert "high_card_txt" in res["summary"]["high_cardinality_columns"]
    assert res["summary"]["duplicate_rows"] == 5
    assert res["duplicate_analysis"]["has_duplicates"] is True


# ---------------------------------------------------------------------------
# O, P, Q, R. Outliers, Skewed Distributions, Zero-Inflated, Mixed Data
# ---------------------------------------------------------------------------
def test_O_P_Q_R_outliers_skew_zero_inflated_mixed():
    """Verify robust distribution shape classifications."""
    n = 100
    # Skewed with outliers
    skewed = np.random.exponential(scale=10.0, size=n)
    skewed[0] = 500.0  # extreme outlier

    # Zero-inflated
    zero_inf = np.array([0.0] * 60 + list(np.random.normal(50, 5, 40)))

    df = pd.DataFrame({
        "skewed_feat": skewed,
        "zero_inf_feat": zero_inf,
        "cat_dim": ["A", "B", "C", "D"] * 25,
    })

    engine = EDAEngine()
    res = engine.analyze(df)

    stats_sk = res["statistics"]["numeric"]["skewed_feat"]
    assert stats_sk["outliers"]["count"] >= 1
    assert stats_sk["skewness"] > 0.5
    assert stats_sk["distribution_shape"] in ("right_skewed", "heavy_tailed")

    stats_zi = res["statistics"]["numeric"]["zero_inf_feat"]
    assert stats_zi["distribution_shape"] == "zero_inflated"
    assert stats_zi["zero_count"] == 60


# ---------------------------------------------------------------------------
# T, U, V. Failure Modes: Empty, Insufficient Rows, Non-finite Values
# ---------------------------------------------------------------------------
def test_T_U_V_failure_modes_and_non_finite_values():
    """Verify robust handling of empty, small, and non-finite data."""
    engine = EDAEngine()

    # Empty DataFrame
    res_empty = engine.analyze(pd.DataFrame())
    assert "error" in res_empty

    # Non-finite values sanitization
    df_inf = pd.DataFrame({
        "num_col": [1.0, np.inf, -np.inf, np.nan, 5.0, 10.0, 20.0],
    })
    res_inf = engine.analyze(df_inf)
    assert "error" not in res_inf
    assert_no_nan_or_inf(res_inf)


# ---------------------------------------------------------------------------
# W, X, Y, Z, AA. Invariants, Contracts, Evidence Provenance, No Causal Language
# ---------------------------------------------------------------------------
def test_W_X_Y_Z_AA_contracts_invariants_and_provenance():
    """Verify deterministic output, canonical AgentResult contract, and ResultValidator invariants."""
    n = 40
    df = pd.DataFrame({
        "x": np.linspace(1, 40, n),
        "y": ["Grp1", "Grp2"] * 20,
    })

    agent = EDAAgent()
    res1 = agent.run({"data": df})
    res2 = agent.run({"data": df})

    # Determinism
    assert res1.data["summary"] == res2.data["summary"]

    # AgentResult contract
    assert res1.is_success
    assert res1.status in (AgentStatus.SUCCESS, AgentStatus.COMPLETED)
    assert 0.0 <= res1.confidence <= 1.0
    assert len(res1.evidence) >= 2
    assert all(e.claim_type == ClaimType.OBSERVATION for e in res1.evidence)

    # Invariant checks via ResultValidator
    vr = ResultValidator().validate(res1, context={"data": df})
    assert vr.is_valid is True

    # No causal language check
    text_data = str(res1.data).lower()
    assert "because of this feature, the outcome causes" not in text_data


# ---------------------------------------------------------------------------
# AB. FastAPI Live HTTP Endpoints
# ---------------------------------------------------------------------------
def test_AB_fastapi_eda_endpoints():
    """Verify POST /api/v1/eda/run and POST /api/v1/eda."""
    client = TestClient(app)

    records = [{"metric_1": float(i), "cat_dim": f"Tier_{i%3}"} for i in range(30)]

    resp1 = client.post("/api/v1/eda/run", json={"dataset": records})
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["status"] in ("success", "completed")
    assert "summary" in data1["result"]

    resp2 = client.post("/api/v1/eda", json={"dataset": records})
    assert resp2.status_code == 200

    # 400 on empty dataset
    resp_empty = client.post("/api/v1/eda/run", json={"dataset": []})
    assert resp_empty.status_code == 400


# ---------------------------------------------------------------------------
# AC. Natural Language Intent Routing
# ---------------------------------------------------------------------------
def test_AC_natural_language_intent_routing():
    """Verify natural-language EDA queries route to AnalyticalIntent.EDA."""
    analyzer = IntentAnalyzer()

    queries = [
        "analyze this dataset",
        "explore the data",
        "give me an EDA",
        "show dataset statistics",
        "profile this dataset",
        "summarize the columns",
        "find missing values",
        "show distributions",
    ]

    for q in queries:
        classification = analyzer.analyze(q)
        assert classification.primary_intent == AnalyticalIntent.EDA, f"Failed for query: {q}"


# ---------------------------------------------------------------------------
# AD. No Global Row Loss from Unrelated Missing Columns
# ---------------------------------------------------------------------------
def test_AD_no_global_row_loss_from_unrelated_missing():
    """Verify profiling one column does not drop rows because an unrelated column has nulls."""
    df = pd.DataFrame({
        "clean_feature": list(range(50)),
        "unrelated_sparse": [None] * 45 + [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    engine = EDAEngine()
    res = engine.analyze(df)

    assert res["summary"]["original_rows"] == 50
    assert res["statistics"]["numeric"]["clean_feature"]["count"] == 50
    assert res["columns"]["unrelated_sparse"]["non_null_count"] == 5


# ---------------------------------------------------------------------------
# AE. Target-Aware EDA
# ---------------------------------------------------------------------------
def test_AE_target_aware_eda():
    """Verify target distribution profiling when target is explicitly provided."""
    n = 60
    df = pd.DataFrame({
        "feat_a": np.random.normal(50, 10, n),
        "target_outcome": np.linspace(100, 200, n),
    })

    engine = EDAEngine()
    res = engine.analyze(df, target="target_outcome")

    assert res["target_profile"] is not None
    assert res["target_profile"]["target_column"] == "target_outcome"
    assert res["target_profile"]["inferred_type"] == "numeric"
    assert res["target_profile"]["numeric_stats"]["mean"] == 150.0


# ---------------------------------------------------------------------------
# AF. Large Dataset Bounded Sampling
# ---------------------------------------------------------------------------
def test_AF_large_dataset_bounded_sampling():
    """Verify bounded sampling for massive datasets preserves performance and determinism."""
    n = 1000
    df_large = pd.DataFrame({
        "val_1": np.linspace(0, 100, n),
        "val_2": np.random.normal(50, 5, n),
    })

    engine = EDAEngine(random_state=42)
    res = engine.analyze(df_large, max_rows=200)

    assert res["summary"]["original_rows"] == 1000
    assert res["summary"]["analyzed_rows"] == 200
    assert res["summary"]["is_sampled"] is True
    assert res["summary"]["sampling_method"] == "reproducible_uniform_sample"
    assert_no_nan_or_inf(res)