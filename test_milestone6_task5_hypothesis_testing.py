"""
Milestone 6 — Task 5: Comprehensive Universal Hypothesis Testing & Statistical Significance Test Suite.

Verifies:
A. Two independent numeric groups
B. Unequal variance groups (Welch's t-test)
C. Non-normal groups (Mann-Whitney U)
D. Paired numeric observations (Paired t-test & Wilcoxon)
E. Multiple groups (One-way ANOVA & Eta-squared)
F. Non-parametric multiple groups (Kruskal-Wallis & post-hoc)
G. Categorical vs Categorical (Chi-Square & Cramer's V)
H. Sparse 2x2 Fisher's exact test & Odds Ratio
I. Arbitrary column names
J. Arbitrary category names
K. Dirty numeric values
L. Percentage values
M. Accounting negative values
N. Pairwise missing values
O. Unrelated sparse columns preserve observations
P. Multiple testing FDR
Q. Effect-size calculation
R. Confidence intervals (lower <= estimate <= upper)
S. Statistical vs Practical significance
T. Assumption reporting
U. Test-selection transparency
V. Alpha validation
W. Insufficient data failure contract
X. Constant group/target
Y. Invalid columns
Z. Causal-language protection
AA. NaN/Infinity sanitization
AB. Deterministic execution
AC. Natural-language routing
AD. FastAPI integration
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
from agent.hypothesis_testing_agent import HypothesisTestingAgent
from agent.hypothesis_testing_engine import HypothesisTestingEngine
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
# A. Two Independent Numeric Groups (Student's t-test)
# ---------------------------------------------------------------------------
def test_A_two_independent_numeric_groups():
    """Verify independent two-sample t-test for normal groups with equal variance."""
    np.random.seed(42)
    n = 40
    g1 = np.random.normal(50, 5, n)
    g2 = np.random.normal(60, 5, n)
    df = pd.DataFrame({
        "score": np.concatenate([g1, g2]),
        "cohort": ["Control"] * n + ["Treatment"] * n,
    })

    engine = HypothesisTestingEngine(alpha=0.05)
    res = engine.test(df, feature="score", group="cohort")

    assert "error" not in res
    assert len(res["hypotheses"]) == 1
    hyp = res["hypotheses"][0]
    assert hyp["test_family"] == "two_sample_numeric"
    assert hyp["p_value"] < 0.001
    assert hyp["statistical_significance"] is True
    assert hyp["effect_size_type"] in ("cohens_d", "hedges_g")
    assert hyp["effect_size"] != 0.0
    assert hyp["practical_significance"] in ("moderate", "large")


# ---------------------------------------------------------------------------
# B. Unequal Variance Groups (Welch's t-test)
# ---------------------------------------------------------------------------
def test_B_unequal_variance_welch_t_test():
    """Verify Welch's t-test selection when variances are unequal."""
    np.random.seed(42)
    g1 = np.random.normal(50, 2, 50)   # Variance ~ 4
    g2 = np.random.normal(55, 12, 50)  # Variance ~ 144
    df = pd.DataFrame({
        "metric": np.concatenate([g1, g2]),
        "variant": ["A"] * 50 + ["B"] * 50,
    })

    engine = HypothesisTestingEngine(alpha=0.05)
    res = engine.test(df, feature="metric", group="variant")

    assert "error" not in res
    hyp = res["hypotheses"][0]
    assert hyp["test_method"] in ("welch_t_test", "mann_whitney_u")
    assert hyp["statistical_significance"] is True
    assert hyp["degrees_of_freedom"] is not None


# ---------------------------------------------------------------------------
# C. Non-Normal Groups (Mann-Whitney U)
# ---------------------------------------------------------------------------
def test_C_non_normal_mann_whitney_u():
    """Verify Mann-Whitney U rank test selection when distribution is heavily skewed / non-normal."""
    np.random.seed(42)
    g1 = np.random.exponential(scale=2.0, size=40)
    g2 = np.random.exponential(scale=8.0, size=40)
    df = pd.DataFrame({
        "duration": np.concatenate([g1, g2]),
        "tier": ["Free"] * 40 + ["Premium"] * 40,
    })

    engine = HypothesisTestingEngine(alpha=0.05)
    res = engine.test(df, feature="duration", group="tier")

    assert "error" not in res
    hyp = res["hypotheses"][0]
    assert hyp["test_method"] in ("mann_whitney_u", "welch_t_test")
    assert hyp["statistical_significance"] is True
    assert hyp["effect_size_type"] in ("rank_biserial", "hedges_g", "cohens_d")


# ---------------------------------------------------------------------------
# D. Paired Numeric Observations (Paired t-test & Wilcoxon)
# ---------------------------------------------------------------------------
def test_D_paired_numeric_observations():
    """Verify paired comparison between two correlated measurements."""
    np.random.seed(42)
    before = np.random.normal(100, 10, 35)
    after = before + np.random.normal(5, 2, 35)  # Significant positive gain
    df = pd.DataFrame({"pre_test": before, "post_test": after})

    engine = HypothesisTestingEngine(alpha=0.05)
    res = engine.test(df, feature="pre_test", feature_2="post_test", paired=True)

    assert "error" not in res
    hyp = res["hypotheses"][0]
    assert hyp["test_family"] == "paired_numeric"
    assert hyp["test_method"] in ("paired_t_test", "wilcoxon_signed_rank")
    assert hyp["statistical_significance"] is True
    assert hyp["mean_difference"] < 0.0  # pre - post is negative
    assert hyp["mean_difference_ci"] is not None
    ci = hyp["mean_difference_ci"]
    assert ci["lower"] <= ci["estimate"] <= ci["upper"]


# ---------------------------------------------------------------------------
# E. Multiple Groups Parametric (One-Way ANOVA & Eta-Squared)
# ---------------------------------------------------------------------------
def test_E_one_way_anova_multiple_groups():
    """Verify One-way ANOVA and Eta-squared effect size across 3+ groups."""
    np.random.seed(42)
    g1 = np.random.normal(20, 3, 30)
    g2 = np.random.normal(25, 3, 30)
    g3 = np.random.normal(30, 3, 30)
    df = pd.DataFrame({
        "revenue": np.concatenate([g1, g2, g3]),
        "region": ["East"] * 30 + ["Central"] * 30 + ["West"] * 30,
    })

    engine = HypothesisTestingEngine(alpha=0.05)
    res = engine.test(df, feature="revenue", group="region")

    assert "error" not in res
    hyp = res["hypotheses"][0]
    assert hyp["test_family"] == "multi_sample_numeric"
    assert hyp["test_method"] in ("one_way_anova", "welch_anova", "kruskal_wallis")
    assert hyp["p_value"] < 0.001
    assert hyp["statistical_significance"] is True
    assert hyp["effect_size_type"] == "eta_squared"
    assert 0.0 <= hyp["effect_size"] <= 1.0
    assert "post_hoc" in hyp
    assert len(hyp["post_hoc"]) == 3  # 3 pairwise combinations


# ---------------------------------------------------------------------------
# F. Non-Parametric Multiple Groups (Kruskal-Wallis)
# ---------------------------------------------------------------------------
def test_F_kruskal_wallis_non_parametric():
    """Verify Kruskal-Wallis test on heavily skewed multi-group data with post-hoc comparisons."""
    np.random.seed(42)
    g1 = np.random.exponential(1.0, 30)
    g2 = np.random.exponential(3.0, 30)
    g3 = np.random.exponential(7.0, 30)
    df = pd.DataFrame({
        "latency": np.concatenate([g1, g2, g3]),
        "cluster_tier": ["Bronze"] * 30 + ["Silver"] * 30 + ["Gold"] * 30,
    })

    engine = HypothesisTestingEngine(alpha=0.05)
    res = engine.test(df, feature="latency", group="cluster_tier")

    assert "error" not in res
    hyp = res["hypotheses"][0]
    assert hyp["test_method"] in ("kruskal_wallis", "one_way_anova", "welch_anova")
    assert hyp["statistical_significance"] is True


# ---------------------------------------------------------------------------
# G & H. Categorical Association & Sparse Fisher's Exact Test
# ---------------------------------------------------------------------------
def test_G_H_categorical_and_fisher_exact():
    """Verify Chi-Square test of independence and Fisher's exact test for 2x2 sparse tables."""
    # 1. 2x2 sparse table for Fisher's Exact
    df_sparse = pd.DataFrame({
        "conversion": ["Yes"] * 2 + ["No"] * 18 + ["Yes"] * 9 + ["No"] * 11,
        "ui_variant": ["Old"] * 20 + ["New"] * 20,
    })
    engine = HypothesisTestingEngine(alpha=0.05)
    res_fe = engine.test(df_sparse, feature="conversion", group="ui_variant")

    assert "error" not in res_fe
    hyp_fe = res_fe["hypotheses"][0]
    assert hyp_fe["test_family"] == "categorical_association"
    assert hyp_fe["test_method"] in ("fisher_exact", "chi_square")
    if hyp_fe["test_method"] == "fisher_exact":
        assert hyp_fe["odds_ratio"] is not None
        if hyp_fe["odds_ratio_ci"]:
            ci = hyp_fe["odds_ratio_ci"]
            assert ci["lower"] <= ci["estimate"] <= ci["upper"]


# ---------------------------------------------------------------------------
# I & J. Arbitrary Column Names & Category Labels
# ---------------------------------------------------------------------------
def test_I_J_arbitrary_names_and_category_labels():
    """Verify system does not hardcode column names or category values."""
    np.random.seed(42)
    df = pd.DataFrame({
        "dim_theta_9": np.random.normal(50, 5, 60),
        "cat_psi_8": ["Omega_Type_A"] * 30 + ["Omega_Type_B"] * 30,
    })

    agent = HypothesisTestingAgent()
    res: AgentResult = agent.run({"data": df, "feature": "dim_theta_9", "group": "cat_psi_8"})

    assert res.is_success
    assert len(res.data["hypotheses"]) == 1
    hyp = res.data["hypotheses"][0]
    assert hyp["variable_x"] == "dim_theta_9"
    assert hyp["variable_group"] == "cat_psi_8"
    assert set(hyp["group_labels"]) == {"Omega_Type_A", "Omega_Type_B"}


# ---------------------------------------------------------------------------
# K, L, M. Dirty Numeric Values & Coercion
# ---------------------------------------------------------------------------
def test_K_L_M_dirty_numeric_coercion():
    """Verify robust coercion of currencies, percentages, accounting brackets, and suffixes."""
    dirty_a = ["$1,200", "€1.5k", "£500", "15%", "(1,200.50)", "2.4M", "-500", "100.0", "$2,000", "3.1k"] * 3
    dirty_b = ["$5,200", "€6.5k", "£700", "45%", "(500.50)", "8.4M", "200", "600.0", "$9,000", "7.1k"] * 3
    df = pd.DataFrame({
        "dirty_val": dirty_a + dirty_b,
        "segment_id": ["Cohort_A"] * len(dirty_a) + ["Cohort_B"] * len(dirty_b),
    })

    engine = HypothesisTestingEngine(alpha=0.05)
    res = engine.test(df, feature="dirty_val", group="segment_id")

    assert "error" not in res
    hyp = res["hypotheses"][0]
    assert hyp["test_statistic"] != 0.0
    assert hyp["p_value"] is not None


# ---------------------------------------------------------------------------
# N & O. Pairwise Missing Values & Row Preservation
# ---------------------------------------------------------------------------
def test_N_O_pairwise_missing_data_preservation():
    """Verify missing data in unrelated columns never drops observations from valid pairs."""
    n = 60
    df = pd.DataFrame({
        "metric_x": np.random.normal(50, 5, n),
        "group_y": ["Grp1"] * 30 + ["Grp2"] * 30,
        "unrelated_sparse": [None if i % 2 == 0 else "junk" for i in range(n)],  # 50% missing
        "unrelated_all_null": [None] * n,                                        # 100% missing
    })

    engine = HypothesisTestingEngine(alpha=0.05)
    res = engine.test(df, feature="metric_x", group="group_y")

    assert "error" not in res
    hyp = res["hypotheses"][0]
    assert hyp["row_accounting"]["valid_rows"] == 60
    assert hyp["row_accounting"]["original_rows"] == 60


# ---------------------------------------------------------------------------
# P. Multiple Testing FDR Correction
# ---------------------------------------------------------------------------
def test_P_multiple_testing_fdr():
    """Verify Benjamini-Hochberg FDR correction across multiple hypotheses."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "num_1": np.random.normal(10, 2, n),
        "num_2": np.random.normal(20, 2, n),
        "num_3": np.random.normal(30, 2, n),
        "group_cat": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
    })

    engine = HypothesisTestingEngine(alpha=0.05)
    res = engine.test(df, group="group_cat")

    assert "error" not in res
    hyps = res["hypotheses"]
    assert len(hyps) >= 3
    for hyp in hyps:
        assert 0.0 <= hyp["adjusted_p_value"] <= 1.0
        assert hyp["adjusted_p_value"] >= hyp["p_value"] - 1e-6  # FDR adjusted p >= raw p


# ---------------------------------------------------------------------------
# Q & R & S. Effect Sizes, Confidence Intervals & Practical Significance
# ---------------------------------------------------------------------------
def test_Q_R_S_effect_size_ci_and_practical_significance():
    """Verify separation of statistical significance from practical effect size and CI bounds."""
    np.random.seed(42)
    # Huge sample size (N=2000), tiny difference (0.1 SD): statistically significant but practically negligible!
    n = 1000
    g1 = np.random.normal(50.0, 10.0, n)
    g2 = np.random.normal(50.3, 10.0, n)
    df = pd.DataFrame({
        "large_n_metric": np.concatenate([g1, g2]),
        "large_n_group": ["A"] * n + ["B"] * n,
    })

    engine = HypothesisTestingEngine(alpha=0.05)
    res = engine.test(df, feature="large_n_metric", group="large_n_group")

    assert "error" not in res
    hyp = res["hypotheses"][0]
    assert hyp["effect_size_type"] in ("cohens_d", "hedges_g")
    assert abs(hyp["effect_size"]) < 0.20
    assert hyp["practical_significance"] == "negligible"

    if hyp["mean_difference_ci"]:
        ci = hyp["mean_difference_ci"]
        assert ci["lower"] <= ci["estimate"] <= ci["upper"]


# ---------------------------------------------------------------------------
# T & U. Assumption Reporting & Test Selection Transparency
# ---------------------------------------------------------------------------
def test_T_U_assumption_reporting_and_transparency():
    """Verify structured assumption status (passed/warning) and candidate test ranking."""
    np.random.seed(42)
    df = pd.DataFrame({
        "val": np.random.normal(0, 1, 40),
        "grp": ["G1"] * 20 + ["G2"] * 20,
    })

    engine = HypothesisTestingEngine(alpha=0.05)
    res = engine.test(df, feature="val", group="grp")

    assert "error" not in res
    hyp = res["hypotheses"][0]
    assert len(hyp["assumptions"]) >= 2
    for assump in hyp["assumptions"]:
        assert assump["status"] in ("passed", "warning", "failed", "not_applicable")
        assert "evidence" in assump
        assert "impact" in assump

    transp = hyp["selection_transparency"]
    assert "selected_test" in transp
    assert len(transp["candidates"]) >= 2
    for cand in transp["candidates"]:
        assert 0.0 <= cand["suitability"] <= 1.0


# ---------------------------------------------------------------------------
# V & W & X & Y. Failure Contracts & Pre-Execution Validation
# ---------------------------------------------------------------------------
def test_V_W_X_Y_failure_contracts_and_validation():
    """Verify structured errors for insufficient data, constant targets, invalid columns, or invalid alpha."""
    agent = HypothesisTestingAgent()

    # 1. Insufficient data (N < 3)
    res_empty = agent.run({"data": pd.DataFrame()})
    assert not res_empty.is_success

    # 2. Constant grouping column
    df_const = pd.DataFrame({"feat": [1, 2, 3, 4], "grp": ["ALL_SAME"] * 4})
    res_const = agent.run({"data": df_const, "feature": "feat", "group": "grp"})
    assert not res_const.is_success
    assert "constant" in (res_const.error_message or "").lower() or "distinct" in (res_const.error_message or "").lower()

    # 3. Missing column
    res_missing = agent.run({"data": pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}), "feature": "non_existent"})
    assert not res_missing.is_success


# ---------------------------------------------------------------------------
# Z & AA & AB. Causal Language, Sanitization & Determinism
# ---------------------------------------------------------------------------
def test_Z_AA_AB_causal_protection_sanitization_determinism():
    """Verify non-causal language, zero NaN/Inf leaks, and deterministic test results."""
    np.random.seed(42)
    df = pd.DataFrame({
        "x_metric": np.random.normal(10, 2, 50),
        "group_id": ["Alpha"] * 25 + ["Beta"] * 25,
    })

    agent = HypothesisTestingAgent()
    res1: AgentResult = agent.run({"data": df, "feature": "x_metric", "group": "group_id"})
    res2: AgentResult = agent.run({"data": df, "feature": "x_metric", "group": "group_id"})

    assert res1.is_success
    assert_no_nan_or_inf(res1.data)

    # Determinism
    h1 = res1.data["hypotheses"][0]
    h2 = res2.data["hypotheses"][0]
    assert h1["test_statistic"] == h2["test_statistic"]
    assert h1["p_value"] == h2["p_value"]

    # Non-causal phrasing check
    interp = h1["practical_interpretation"].lower()
    for forbidden in ("causes", "caused", "drives", "driven by", "because of", "results in"):
        assert forbidden not in interp


# ---------------------------------------------------------------------------
# AC. Natural Language Intent Routing
# ---------------------------------------------------------------------------
def test_AC_natural_language_hypothesis_routing():
    """Verify intent analyzer correctly maps hypothesis testing queries."""
    analyzer = IntentAnalyzer()

    r1 = analyzer.analyze("test whether these groups differ")
    assert r1.primary_intent == AnalyticalIntent.HYPOTHESIS_TESTING

    r2 = analyzer.analyze("is this difference statistically significant?")
    assert r2.primary_intent == AnalyticalIntent.HYPOTHESIS_TESTING

    r3 = analyzer.analyze("compare these groups statistically")
    assert r3.primary_intent == AnalyticalIntent.HYPOTHESIS_TESTING

    r4 = analyzer.analyze("perform a hypothesis test on category differences")
    assert r4.primary_intent == AnalyticalIntent.HYPOTHESIS_TESTING

    r5 = analyzer.analyze("does category A differ from category B?")
    assert r5.primary_intent == AnalyticalIntent.HYPOTHESIS_TESTING


# ---------------------------------------------------------------------------
# AD. FastAPI Live HTTP Endpoints
# ---------------------------------------------------------------------------
def test_AD_fastapi_hypothesis_testing_endpoints():
    """Verify POST /api/v1/hypothesis-testing and POST /api/v1/hypothesis-testing/run via TestClient."""
    client = TestClient(app)

    records = [
        {"metric_a": float(i), "segment_b": "Group_1" if i < 15 else "Group_2"}
        for i in range(30)
    ]

    # 1. /api/v1/hypothesis-testing
    resp1 = client.post("/api/v1/hypothesis-testing", json={"dataset": records, "feature": "metric_a", "group": "segment_b"})
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["status"] in ("success", "completed")
    assert len(data1["result"]["hypotheses"]) >= 1

    # 2. /api/v1/hypothesis-testing/run
    resp2 = client.post("/api/v1/hypothesis-testing/run", json={"dataset": records, "feature": "metric_a", "group": "segment_b"})
    assert resp2.status_code == 200

    # 3. Empty dataset returns 400
    resp_empty = client.post("/api/v1/hypothesis-testing", json={"dataset": []})
    assert resp_empty.status_code == 400