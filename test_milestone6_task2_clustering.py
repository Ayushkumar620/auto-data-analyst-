"""
Milestone 6 — Task 2: Comprehensive Universal Clustering & Segmentation Test Suite.

Verifies:
A. Clear numeric clusters
B. Arbitrary column names
C. Different scales (large scale + small scale)
D. Noisy clusters
E. Overlapping clusters
F. Categorical + numeric mixed data
G. Missing values (imputed non-destructively)
H. Sparse unrelated columns (dropped without losing rows)
I. Identifier columns (UUIDs/sequential indices excluded)
J. Constant columns (0 variance excluded)
K. High-cardinality text (excluded)
L. Dirty numeric strings (currencies, commas, %, accounting brackets)
M. Negative / accounting values
N. Small dataset validation failure (N < 5)
O. Constant-only dataset rejection
P. Identifier-only dataset rejection
Q. Deterministic repeated execution
R. Dynamic cluster-count selection (k in [2, 8])
S. DBSCAN/HDBSCAN noise handling (noise_count, noise_ratio)
T. Metric mathematical invariants (Silhouette in [-1, 1], CH >= 0, DB >= 0)
U. Confidence bounds (0.0 <= confidence <= 1.0)
V. Evidence provenance (ClaimType.OBSERVATION)
W. Causal-language prevention (strictly descriptive)
X. AgentResult contract compatibility
Y. Natural-language routing
Z. FastAPI endpoint integration (POST /api/v1/clustering/run)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_blobs

from agent.agent_result import AgentError, AgentResult, AgentStatus, ClaimType, ErrorCategory
from agent.clustering_agent import ClusteringAgent
from agent.clustering_engine import ClusteringEngine
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
# A. Clear Numeric Clusters
# ---------------------------------------------------------------------------
def test_A_clear_numeric_clusters():
    """Verify engine partitions well-separated 3-cluster Gaussian synthetic blobs."""
    X, _ = make_blobs(n_samples=90, n_features=2, centers=3, cluster_std=0.8, random_state=42)
    df = pd.DataFrame(X, columns=["dim_alpha", "dim_beta"])

    engine = ClusteringEngine(random_state=42)
    res = engine.cluster(df, n_clusters=3)

    assert "error" not in res
    assert res["cluster_count"] == 3
    assert res["rows_analyzed"] == 90
    assert len(res["labels"]) == 90
    assert len(res["cluster_sizes"]) == 3
    assert res["validation_metrics"]["silhouette_score"] > 0.40


# ---------------------------------------------------------------------------
# B. Arbitrary Column Names
# ---------------------------------------------------------------------------
def test_B_arbitrary_column_names():
    """Verify clustering functions without keywords like customer, user, revenue."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "gamma_flux_1": np.concatenate([np.random.normal(10, 1, 30), np.random.normal(50, 2, 30)]),
        "theta_reading_2": np.concatenate([np.random.normal(5, 0.5, 30), np.random.normal(25, 1, 30)]),
    })

    agent = ClusteringAgent()
    res = agent.run({"data": df})

    assert res.is_success
    assert res.data["cluster_count"] >= 2
    assert "gamma_flux_1" in res.data["features_used"]
    assert "theta_reading_2" in res.data["features_used"]


# ---------------------------------------------------------------------------
# C. Different Scales (Robust Feature Scaling)
# ---------------------------------------------------------------------------
def test_C_different_feature_scales():
    """Verify large unit scale ($10^6$) does not dominate small unit scale ($10^{-2}$)."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "macro_metric": np.concatenate([np.random.normal(1e6, 1e4, 25), np.random.normal(5e6, 2e4, 25)]),
        "micro_metric": np.concatenate([np.random.normal(0.01, 0.001, 25), np.random.normal(0.08, 0.002, 25)]),
    })

    engine = ClusteringEngine(random_state=42)
    res = engine.cluster(df, n_clusters=2)

    assert "error" not in res
    assert res["validation_metrics"]["silhouette_score"] > 0.60


# ---------------------------------------------------------------------------
# D & E. Noisy & Overlapping Clusters
# ---------------------------------------------------------------------------
def test_D_E_noisy_and_overlapping_clusters():
    """Verify graceful handling of high noise and overlapping distributions."""
    X, _ = make_blobs(n_samples=80, n_features=3, centers=2, cluster_std=3.5, random_state=42)
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])

    engine = ClusteringEngine(random_state=42)
    res = engine.cluster(df, n_clusters="auto")

    assert "error" not in res
    assert res["cluster_count"] >= 2
    assert_no_nan_or_inf(res)


# ---------------------------------------------------------------------------
# F. Categorical + Numeric Mixed Data
# ---------------------------------------------------------------------------
def test_F_categorical_and_numeric_mixed_data():
    """Verify safe encoding of low-cardinality categorical features."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "group_tier": ["Tier_A"] * 30 + ["Tier_B"] * 30,
        "numeric_signal_1": np.concatenate([np.random.normal(10, 2, 30), np.random.normal(50, 3, 30)]),
        "numeric_signal_2": np.concatenate([np.random.normal(5, 1, 30), np.random.normal(20, 2, 30)]),
    })

    engine = ClusteringEngine(random_state=42)
    res = engine.cluster(df)

    assert "error" not in res
    assert "group_tier" in res["features_used"]
    assert res["cluster_count"] >= 2


# ---------------------------------------------------------------------------
# G & H. Missing Values & Sparse Unrelated Columns
# ---------------------------------------------------------------------------
def test_G_H_missing_values_and_sparse_columns():
    """Verify median imputation for valid features and exclusion of >60% null columns without row loss."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "meas_a": [None if i % 10 == 0 else float(i) for i in range(n)],     # 10% null (imputed)
        "meas_b": np.random.normal(20, 2, n),                                 # 0% null
        "sparse_unrelated": [None if i % 4 != 0 else f"s_{i}" for i in range(n)], # 75% null (excluded)
    })

    engine = ClusteringEngine(random_state=42)
    res = engine.cluster(df)

    assert "error" not in res
    assert res["rows_analyzed"] == n
    assert "sparse_unrelated" in res["excluded_features"]
    assert "meas_a" in res["features_used"]
    assert "meas_b" in res["features_used"]


# ---------------------------------------------------------------------------
# I, J, K. Identifier, Constant, and High-Cardinality Text Exclusion
# ---------------------------------------------------------------------------
def test_I_J_K_identifier_constant_text_exclusion():
    """Verify UUIDs, constant columns, and freeform text are excluded from distance metrics."""
    n = 40
    df = pd.DataFrame({
        "record_uuid": [f"uuid-seq-{i:05d}" for i in range(n)],
        "zero_var_const": [99.0] * n,
        "freeform_text": [f"This is freeform observation note number {i}" for i in range(n)],
        "valid_dim_x": np.random.normal(10, 1, n),
        "valid_dim_y": np.random.normal(20, 2, n),
    })

    engine = ClusteringEngine(random_state=42)
    res = engine.cluster(df)

    assert "error" not in res
    assert "record_uuid" not in res["features_used"]
    assert "zero_var_const" not in res["features_used"]
    assert "freeform_text" not in res["features_used"]
    assert "valid_dim_x" in res["features_used"]
    assert "valid_dim_y" in res["features_used"]


# ---------------------------------------------------------------------------
# L & M. Dirty Numeric Strings & Negative/Accounting Values
# ---------------------------------------------------------------------------
def test_L_M_dirty_numeric_and_negative_accounting():
    """Verify currencies, commas, percentages, and accounting brackets are coerced."""
    dirty_a = ["$1,200", "€2,300", "£4,500", "15.5%", "(1,250.50)", "2.5k", "3.2M", "100.0", "120.0", "$500"]
    dirty_b = ["-50.0", "(25.0)", "10.0%", "$5,000", "1,000", "2.0k", "-100", "40.0", "50.0", "(10.0)"]
    df = pd.DataFrame({"dirty_col_1": dirty_a * 3, "dirty_col_2": dirty_b * 3})

    engine = ClusteringEngine(random_state=42)
    res = engine.cluster(df)

    assert "error" not in res
    assert res["rows_analyzed"] == 30
    assert "dirty_col_1" in res["features_used"]
    assert "dirty_col_2" in res["features_used"]


# ---------------------------------------------------------------------------
# N, O, P. Rejection & Failure Invariants
# ---------------------------------------------------------------------------
def test_N_O_P_rejection_and_failure_invariants():
    """Verify structured errors for insufficient data, all constant, and identifier-only."""
    agent = ClusteringAgent()

    # N: Insufficient rows (N < 5)
    df_short = pd.DataFrame({"dim1": [1, 2, 3], "dim2": [4, 5, 6]})
    res_short = agent.run({"data": df_short})
    assert not res_short.is_success
    assert "at least 5" in (res_short.error_message or "")

    # O: All constant columns
    df_const = pd.DataFrame({"col_a": [5.0] * 10, "col_b": [10.0] * 10})
    res_const = agent.run({"data": df_const})
    assert not res_const.is_success
    assert "variance" in (res_const.error_message or "").lower() or "constant" in (res_const.error_message or "").lower()

    # P: Identifier-only dataset
    df_id = pd.DataFrame({"id1": [f"id_{i}" for i in range(15)], "id2": [f"key_{i}" for i in range(15)]})
    res_id = agent.run({"data": df_id})
    assert not res_id.is_success
    assert "identifier" in (res_id.error_message or "").lower() or "insufficient" in (res_id.error_message or "").lower()


# ---------------------------------------------------------------------------
# Q. Deterministic Repeated Execution
# ---------------------------------------------------------------------------
def test_Q_deterministic_repeated_execution():
    """Verify identical random seed produces identical partitions."""
    X, _ = make_blobs(n_samples=50, n_features=2, centers=3, random_state=42)
    df = pd.DataFrame(X, columns=["dim_1", "dim_2"])

    engine1 = ClusteringEngine(random_state=42)
    res1 = engine1.cluster(df, n_clusters=3, random_state=42)

    engine2 = ClusteringEngine(random_state=42)
    res2 = engine2.cluster(df, n_clusters=3, random_state=42)

    assert res1["labels"] == res2["labels"]
    assert res1["validation_metrics"] == res2["validation_metrics"]


# ---------------------------------------------------------------------------
# R. Dynamic Cluster-Count Selection
# ---------------------------------------------------------------------------
def test_R_dynamic_cluster_count_selection():
    """Verify k='auto' evaluates multiple k values and selects the best."""
    X, _ = make_blobs(n_samples=75, n_features=3, centers=4, cluster_std=0.7, random_state=42)
    df = pd.DataFrame(X, columns=["dim_a", "dim_b", "dim_c"])

    engine = ClusteringEngine(random_state=42)
    res = engine.cluster(df, n_clusters="auto")

    assert "error" not in res
    assert 2 <= res["cluster_count"] <= 8
    assert len(res["candidate_leaderboard"]) >= 2
    # Leaderboard sorted descending by composite score
    assert res["candidate_leaderboard"][0]["composite_score"] >= res["candidate_leaderboard"][-1]["composite_score"]


# ---------------------------------------------------------------------------
# S. DBSCAN / Density Noise Handling
# ---------------------------------------------------------------------------
def test_S_dbscan_noise_handling():
    """Verify DBSCAN explicit noise reporting and cluster profiling."""
    X, _ = make_blobs(n_samples=60, n_features=2, centers=2, cluster_std=0.5, random_state=42)
    # Add extreme outliers
    X[0] = [100.0, 100.0]
    X[1] = [-100.0, -100.0]
    df = pd.DataFrame(X, columns=["d1", "d2"])

    engine = ClusteringEngine(random_state=42)
    res = engine.cluster(df, method="dbscan")

    assert "error" not in res
    assert "noise_count" in res
    assert 0.0 <= res["noise_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# T, U, V, W, X. Mathematical Invariants & Epistemic Contracts
# ---------------------------------------------------------------------------
def test_T_U_V_W_X_invariants_and_contracts():
    """Verify mathematical bounds, confidence, evidence, non-causal language, and AgentResult."""
    X, _ = make_blobs(n_samples=45, n_features=2, centers=2, random_state=42)
    df = pd.DataFrame(X, columns=["f_alpha", "f_beta"])

    agent = ClusteringAgent()
    res: AgentResult = agent.run({"data": df})

    assert res.is_success
    assert_no_nan_or_inf(res.data)

    # T: Metrics
    metrics = res.data["validation_metrics"]
    assert -1.0 <= metrics["silhouette_score"] <= 1.0
    assert metrics["calinski_harabasz_score"] >= 0.0
    assert metrics["davies_bouldin_score"] >= 0.0

    # U: Confidence
    assert 0.0 <= res.confidence <= 1.0

    # V: Evidence
    assert len(res.evidence) > 0
    assert res.evidence[0].claim_type == ClaimType.OBSERVATION

    # W: Non-causal cluster characterizations
    for profile in res.data["cluster_profiles"]:
        char = profile["characterization"].lower()
        for forbidden in ("causes", "caused", "drives", "driven by", "because of"):
            assert forbidden not in char

    # X: AgentResult dictionary access & status
    assert res["cluster_count"] >= 2
    assert res.status in (AgentStatus.SUCCESS, AgentStatus.COMPLETED)


# ---------------------------------------------------------------------------
# Y. Natural Language Intent Routing
# ---------------------------------------------------------------------------
def test_Y_natural_language_intent_routing():
    """Verify intent analyzer correctly maps segmentation/clustering queries."""
    analyzer = IntentAnalyzer()

    r1 = analyzer.analyze("segment users into natural groups")
    assert r1.primary_intent == AnalyticalIntent.CLUSTERING

    r2 = analyzer.analyze("cluster customers and find customer segments")
    assert r2.primary_intent == AnalyticalIntent.CLUSTERING

    r3 = analyzer.analyze("discover natural groups in this dataset")
    assert r3.primary_intent == AnalyticalIntent.CLUSTERING


# ---------------------------------------------------------------------------
# Z. FastAPI Live HTTP Endpoints
# ---------------------------------------------------------------------------
def test_Z_fastapi_clustering_endpoints():
    """Verify POST /api/v1/clustering/run and POST /api/v1/clustering via TestClient."""
    client = TestClient(app)

    # 1. Valid request
    records = [{"dim_x": float(i % 3 * 10 + np.random.normal(0, 0.5)), "dim_y": float(i % 3 * 20 + np.random.normal(0, 0.5))} for i in range(30)]
    resp = client.post("/api/v1/clustering/run", json={
        "dataset": records,
        "n_clusters": 3,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("success", "completed")
    assert data["result"]["cluster_count"] == 3

    # 2. Empty dataset returns 400
    resp_empty = client.post("/api/v1/clustering", json={"dataset": []})
    assert resp_empty.status_code == 400
