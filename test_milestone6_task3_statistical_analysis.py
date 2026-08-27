"""
Milestone 6 — Task 3: Comprehensive Universal Statistical Relationship & Dependency Analysis Test Suite.

Verifies:
A. Strong linear relationship
B. Strong monotonic non-linear relationship
C. Weak relationship
D. Negative relationship
E. Arbitrary column names
F. Different numeric scales
G. Outlier sensitivity
H. Missing values (pairwise non-destructive masking)
I. Sparse unrelated columns
J. Dirty numeric strings
K. Categorical <-> Numeric analysis (ANOVA, Kruskal-Wallis, Eta-squared)
L. Categorical <-> Categorical analysis (Chi-Square, Cramer's V)
M. Binary categorical relationship (Point-biserial)
N. Small sample validation (N < 3)
O. Constant-column rejection
P. Identifier exclusion
Q. Multiple-testing correction (Benjamini-Hochberg FDR)
R. Adjusted p-value validity
S. Effect-size validation
T. Deterministic repeated execution
U. Confidence bounds
V. Evidence provenance
W. Causal-language prevention
X. AgentResult contract
Y. Natural-language routing
Z. FastAPI integration
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
from agent.intent import AnalyticalIntent, IntentAnalyzer
from agent.pre_execution_validator import PreExecutionValidator
from agent.result_validator import ResultValidator
from agent.statistical_analysis_agent import StatisticalAnalysisAgent
from agent.statistical_analysis_engine import StatisticalAnalysisEngine
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
# A. Strong Linear Relationship
# ---------------------------------------------------------------------------
def test_A_strong_linear_relationship():
    """Verify engine detects strong positive linear correlation (r > 0.85, p < 0.001)."""
    np.random.seed(42)
    n = 60
    x = np.linspace(10, 100, n)
    y = 2.5 * x + np.random.normal(0, 5, n)
    df = pd.DataFrame({"signal_x": x, "signal_y": y})

    engine = StatisticalAnalysisEngine()
    res = engine.analyze(df)

    assert "error" not in res
    assert len(res["relationships"]) == 1
    rel = res["relationships"][0]
    assert rel["pearson"]["r"] > 0.85
    assert rel["p_value"] < 0.001
    assert rel["strength"] in ("strong", "very_strong")
    assert rel["direction"] == "positive"


# ---------------------------------------------------------------------------
# B. Strong Monotonic Non-Linear Relationship
# ---------------------------------------------------------------------------
def test_B_strong_monotonic_nonlinear_relationship():
    """Verify Spearman rank correlation captures monotonic non-linear dependencies (x^3 / exponential)."""
    np.random.seed(42)
    n = 50
    x = np.linspace(1, 10, n)
    y = np.exp(x / 2.0) + np.random.normal(0, 1, n)
    df = pd.DataFrame({"input_val": x, "nonlinear_output": y})

    engine = StatisticalAnalysisEngine()
    res = engine.analyze(df)

    assert "error" not in res
    rel = res["relationships"][0]
    assert rel["spearman"]["rho"] > 0.95
    assert rel["is_significant"]


# ---------------------------------------------------------------------------
# C & D. Weak & Negative Relationships
# ---------------------------------------------------------------------------
def test_C_D_weak_and_negative_relationships():
    """Verify detection of negative correlation (r < -0.80) and negligible independence."""
    np.random.seed(42)
    n = 70
    x = np.random.normal(50, 10, n)
    y_neg = -1.8 * x + np.random.normal(0, 4, n)
    y_noise = np.random.normal(100, 15, n)
    df = pd.DataFrame({"x_feat": x, "neg_feat": y_neg, "noise_feat": y_noise})

    engine = StatisticalAnalysisEngine()
    res = engine.analyze(df)

    assert "error" not in res
    ranked = res["ranked_relationships"]
    # Negative relationship should be top ranked by effect magnitude
    top_rel = ranked[0]
    assert top_rel["direction"] == "negative"
    assert top_rel["statistic"] < -0.80

    # Noise relationship should have negligible strength
    noise_rels = [r for r in ranked if "noise_feat" in (r["feature_x"], r["feature_y"])]
    for r in noise_rels:
        assert r["effect_size"] < 0.30


# ---------------------------------------------------------------------------
# E & F. Arbitrary Column Names & Different Scales
# ---------------------------------------------------------------------------
def test_E_F_arbitrary_names_and_scales():
    """Verify arbitrary naming without keywords and scale invariance ($10^7$ vs $10^{-4}$)."""
    np.random.seed(42)
    n = 50
    macro = np.linspace(1e6, 1e7, n)
    micro = macro * 1e-10 + np.random.normal(0, 1e-5, n)
    df = pd.DataFrame({"param_alpha_8": macro, "metric_zeta_9": micro})

    agent = StatisticalAnalysisAgent()
    res: AgentResult = agent.run({"data": df})

    assert res.is_success
    assert "param_alpha_8" in res.data["numeric_features"]
    assert "metric_zeta_9" in res.data["numeric_features"]
    assert res.data["relationships"][0]["effect_size"] > 0.80


# ---------------------------------------------------------------------------
# G. Outlier Sensitivity Detection
# ---------------------------------------------------------------------------
def test_G_outlier_sensitivity_detection():
    """Verify engine detects when extreme outliers distort Pearson vs Spearman."""
    np.random.seed(42)
    n = 60
    x = np.random.normal(0, 1, n)
    y = np.random.normal(0, 1, n)
    # Inject heavy leverage outlier that artificially inflates Pearson
    x[0] = 50.0
    y[0] = 50.0
    df = pd.DataFrame({"x_var": x, "y_var": y})

    engine = StatisticalAnalysisEngine()
    res = engine.analyze(df)

    assert "error" not in res
    rel = res["relationships"][0]
    assert rel["outlier_sensitivity"] is True
    assert rel["primary_method"] == "spearman"


# ---------------------------------------------------------------------------
# H & I. Missing Values & Sparse Unrelated Columns
# ---------------------------------------------------------------------------
def test_H_I_pairwise_missing_values_and_sparse_columns():
    """Verify pairwise-valid masking: 1 sparse column does NOT discard observations of other columns."""
    np.random.seed(42)
    n = 50
    x = np.linspace(1, 50, n)
    y = 2 * x + np.random.normal(0, 2, n)
    # Make x have 5 nulls, y have 5 nulls, and sparse have 85% nulls
    s_x = [None if i % 10 == 0 else x[i] for i in range(n)]
    s_y = [None if i % 8 == 0 else y[i] for i in range(n)]
    s_sparse = [None if i % 6 != 0 else f"cat_{i}" for i in range(n)]

    df = pd.DataFrame({"feat_x": s_x, "feat_y": s_y, "sparse_col": s_sparse})

    engine = StatisticalAnalysisEngine()
    res = engine.analyze(df)

    assert "error" not in res
    assert "sparse_col" in res["excluded_features"]
    rel = res["relationships"][0]
    # Pairwise valid rows should be around 40-42
    assert 38 <= rel["valid_rows"] <= 45
    assert rel["effect_size"] > 0.90


# ---------------------------------------------------------------------------
# J. Dirty Numeric Strings
# ---------------------------------------------------------------------------
def test_J_dirty_numeric_strings():
    """Verify currencies, percentages, commas, and negative accounting brackets are cleaned."""
    dirty_x = ["$1,200", "€2,300", "£4,500", "15.5%", "(1,250.50)", "2.5k", "3.2M", "100.0", "120.0", "$500"]
    dirty_y = ["$2,400", "€4,600", "£9,000", "31.0%", "(2,501.00)", "5.0k", "6.4M", "200.0", "240.0", "$1,000"]
    df = pd.DataFrame({"val_1": dirty_x * 3, "val_2": dirty_y * 3})

    engine = StatisticalAnalysisEngine()
    res = engine.analyze(df)

    assert "error" not in res
    assert len(res["relationships"]) == 1
    rel = res["relationships"][0]
    assert rel["effect_size"] > 0.95


# ---------------------------------------------------------------------------
# K & M. Categorical <-> Numeric Analysis (ANOVA, Point-Biserial, Eta2)
# ---------------------------------------------------------------------------
def test_K_M_categorical_numeric_analysis():
    """Verify Point-Biserial for binary and One-Way ANOVA / Eta-squared for multi-group."""
    np.random.seed(42)
    n = 60
    # Binary test
    df_bin = pd.DataFrame({
        "group_bin": ["Control"] * 30 + ["Treatment"] * 30,
        "metric_score": np.concatenate([np.random.normal(50, 5, 30), np.random.normal(80, 5, 30)]),
    })
    engine = StatisticalAnalysisEngine()
    res_bin = engine.analyze(df_bin)
    assert "error" not in res_bin
    rel_bin = res_bin["relationships"][0]
    assert rel_bin["pair_type"] == "numeric_categorical"
    assert rel_bin["primary_method"] == "point_biserial"
    assert rel_bin["effect_size"] > 0.50

    # Multi-group ANOVA test
    df_multi = pd.DataFrame({
        "tier_cat": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        "revenue_val": np.concatenate([np.random.normal(10, 2, 20), np.random.normal(30, 2, 20), np.random.normal(60, 3, 20)]),
    })
    res_multi = engine.analyze(df_multi)
    rel_multi = res_multi["relationships"][0]
    assert rel_multi["primary_method"] == "anova"
    assert rel_multi["anova"]["f_statistic"] > 50.0
    assert rel_multi["effect_size"] > 0.50  # High eta-squared


# ---------------------------------------------------------------------------
# L. Categorical <-> Categorical Analysis (Chi-Square, Cramer's V)
# ---------------------------------------------------------------------------
def test_L_categorical_categorical_analysis():
    """Verify Chi-square independence test and Cramer's V association metric."""
    df_cat = pd.DataFrame({
        "department": ["Eng"] * 30 + ["Sales"] * 30 + ["HR"] * 30,
        "location": ["Remote"] * 25 + ["Office"] * 5 + ["Office"] * 25 + ["Remote"] * 5 + ["Office"] * 15 + ["Remote"] * 15,
    })

    engine = StatisticalAnalysisEngine()
    res = engine.analyze(df_cat)

    assert "error" not in res
    rel = res["relationships"][0]
    assert rel["pair_type"] == "categorical_categorical"
    assert rel["primary_method"] == "chi_square"
    assert rel["chi_square"]["cramers_v"] > 0.30
    assert rel["is_significant"] is True


# ---------------------------------------------------------------------------
# N, O, P. Validation Failures (Small Sample, Constant, Identifier)
# ---------------------------------------------------------------------------
def test_N_O_P_validation_failure_modes():
    """Verify structured rejection for small samples, all constant, and identifier-only."""
    agent = StatisticalAnalysisAgent()

    # N: Small sample (N < 3)
    df_small = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    res_small = agent.run({"data": df_small})
    assert not res_small.is_success
    assert "at least 3" in (res_small.error_message or "")

    # O: All constant
    df_const = pd.DataFrame({"c1": [10.0] * 10, "c2": [20.0] * 10})
    res_const = agent.run({"data": df_const})
    assert not res_const.is_success
    assert "variance" in (res_const.error_message or "").lower() or "constant" in (res_const.error_message or "").lower()

    # P: Identifier-only
    df_id = pd.DataFrame({"id1": [f"user_{i}" for i in range(10)], "id2": [f"session_{i}" for i in range(10)]})
    res_id = agent.run({"data": df_id})
    assert not res_id.is_success


# ---------------------------------------------------------------------------
# Q & R. Multiple Testing Correction (Benjamini-Hochberg)
# ---------------------------------------------------------------------------
def test_Q_R_multiple_testing_correction():
    """Verify Benjamini-Hochberg FDR ensures adjusted_p_value >= raw p_value and in [0, 1]."""
    np.random.seed(42)
    n = 60
    # Generate 5 noise columns and 1 correlated pair
    df = pd.DataFrame({
        f"col_{i}": np.random.normal(0, 1, n) for i in range(6)
    })
    df["col_1"] = df["col_0"] * 2.0 + np.random.normal(0, 0.5, n)

    engine = StatisticalAnalysisEngine()
    res = engine.analyze(df)

    assert "error" not in res
    for rel in res["relationships"]:
        raw_p = rel["p_value"]
        adj_p = rel["adjusted_p_value"]
        assert 0.0 <= raw_p <= 1.0
        assert 0.0 <= adj_p <= 1.0
        assert adj_p >= raw_p  # Adjusted p-value is always >= raw p-value


# ---------------------------------------------------------------------------
# S, T, U, V, W, X. Invariants, Contracts & Non-Causal Language
# ---------------------------------------------------------------------------
def test_S_T_U_V_W_X_invariants_and_contracts():
    """Verify mathematical bounds, confidence bounds, evidence provenance, non-causal language, AgentResult."""
    np.random.seed(42)
    n = 45
    df = pd.DataFrame({
        "dim_1": np.linspace(10, 50, n),
        "dim_2": np.linspace(20, 100, n) + np.random.normal(0, 3, n),
    })

    # T: Deterministic execution
    engine1 = StatisticalAnalysisEngine()
    res1 = engine1.analyze(df)
    engine2 = StatisticalAnalysisEngine()
    res2 = engine2.analyze(df)
    assert res1["relationships"][0]["statistic"] == res2["relationships"][0]["statistic"]

    # X: AgentResult run
    agent = StatisticalAnalysisAgent()
    res: AgentResult = agent.run({"data": df})

    assert res.is_success
    assert_no_nan_or_inf(res.data)

    # S: Effect size in [0, 1]
    rel = res.data["relationships"][0]
    assert 0.0 <= rel["effect_size"] <= 1.0

    # U: Confidence in [0, 1]
    assert 0.0 <= res.confidence <= 1.0

    # V: Evidence provenance
    assert len(res.evidence) > 0
    assert res.evidence[0].claim_type == ClaimType.CORRELATION

    # W: Non-causal language check
    interp = rel["interpretation"].lower()
    for forbidden in ("causes", "caused", "drives", "driven by", "because of", "leads to"):
        assert forbidden not in interp


# ---------------------------------------------------------------------------
# Y. Natural Language Intent Routing
# ---------------------------------------------------------------------------
def test_Y_natural_language_intent_routing():
    """Verify intent analyzer correctly maps correlation and dependency queries."""
    analyzer = IntentAnalyzer()

    r1 = analyzer.analyze("find correlations between variables")
    assert r1.primary_intent == AnalyticalIntent.CORRELATION

    r2 = analyzer.analyze("what variables are related in this dataset?")
    assert r2.primary_intent == AnalyticalIntent.CORRELATION

    r3 = analyzer.analyze("which variables move together?")
    assert r3.primary_intent == AnalyticalIntent.CORRELATION

    r4 = analyzer.analyze("show statistical associations and dependencies")
    assert r4.primary_intent == AnalyticalIntent.CORRELATION


# ---------------------------------------------------------------------------
# Z. FastAPI Live HTTP Endpoints
# ---------------------------------------------------------------------------
def test_Z_fastapi_statistical_analysis_endpoints():
    """Verify POST /api/v1/statistical-analysis/run and POST /api/v1/statistical-analysis via TestClient."""
    client = TestClient(app)

    records = [{"var_a": float(i), "var_b": float(2 * i + np.random.normal(0, 0.5))} for i in range(25)]

    # 1. Valid request
    resp = client.post("/api/v1/statistical-analysis/run", json={"dataset": records})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("success", "completed")
    assert len(data["result"]["relationships"]) >= 1

    # 2. Empty dataset returns 400
    resp_empty = client.post("/api/v1/statistical-analysis", json={"dataset": []})
    assert resp_empty.status_code == 400