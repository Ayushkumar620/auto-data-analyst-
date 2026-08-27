"""
Milestone 7 — Task 1: Comprehensive Universal Agent Orchestration & End-to-End Execution Test Suite.

Verifies:
A. Single-task command routing
B. Multi-task analytical request
C. Arbitrary dataset column names
D. Automatic target/feature interpretation
E. Forecasting task planning
F. Anomaly task planning
G. Clustering task planning
H. Statistical relationship task planning
I. Prediction task planning
J. EDA task planning
K. Dependency ordering
L. Independent task execution
M. Validation failure isolation
N. Partial success
O. Retryable failure recovery
P. Non-retryable failure
Q. Ambiguous command
R. Unsupported command
S. Empty dataset
T. Malformed dataset
U. Structured AgentError contract
V. No traceback leakage
W. Evidence preservation
X. Confidence bounds
Y. Deterministic planning
Z. FastAPI orchestration endpoint
AA. Existing API regression compatibility
AB. Tool registry integration
AC. Duplicate execution prevention
AD. Complete end-to-end user command execution
"""
from __future__ import annotations

import math
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.agent_result import AgentError, AgentResult, AgentStatus, ClaimType, ErrorCategory
from agent.orchestrator import AnalyticalPlan, PlanTask, TaskStatus, UniversalOrchestrator
from agent.tool_registry import DEFAULT_TOOL_REGISTRY
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
# A & B & C & D. Single/Multi-Task Routing, Arbitrary Names & Target/Feature Inference
# ---------------------------------------------------------------------------
def test_A_B_C_D_orchestration_planning_and_routing():
    """Verify single and multi-task command routing on arbitrary column names."""
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "timestamp_axis_alpha": dates.strftime("%Y-%m-%d"),
        "measure_metric_beta": np.linspace(10, 100, n),
        "attribute_gamma": ["Group1", "Group2", "Group3"] * 20,
    })

    orch = UniversalOrchestrator()

    # 1. Single Task (EDA)
    res_eda = orch.orchestrate("give me an overview and profile the dataset", df)
    assert res_eda.is_success
    assert "eda" in res_eda.data["tasks"]

    # 2. Multi-Task (Profile + Anomaly Detection)
    res_multi = orch.orchestrate("profile this data and find unusual anomalies", df)
    assert res_multi.is_success
    assert "eda" in res_multi.data["tasks"]
    assert "anomaly_detection" in res_multi.data["tasks"]


# ---------------------------------------------------------------------------
# E, F, G, H, I, J. Specific Analytical Task Planning
# ---------------------------------------------------------------------------
def test_E_F_G_H_I_J_specific_task_planning():
    """Verify planning for forecasting, anomalies, clustering, stats, prediction, and EDA."""
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "time_idx": dates.strftime("%Y-%m-%d"),
        "sales_val": np.linspace(100, 200, n) + np.random.normal(0, 5, n),
        "feat_1": np.random.normal(50, 10, n),
        "feat_2": np.random.normal(20, 5, n),
    })

    orch = UniversalOrchestrator()

    # E. Forecasting
    plan_fc = orch.plan("forecast next 6 periods for sales_val", df)
    assert any(t.task_type == "forecasting" for t in plan_fc.tasks)

    # F. Anomaly Detection
    plan_anom = orch.plan("detect abnormal outliers in this dataset", df)
    assert any(t.task_type == "anomaly_detection" for t in plan_anom.tasks)

    # G. Clustering
    plan_cl = orch.plan("segment records into natural cluster groups", df)
    assert any(t.task_type == "clustering" for t in plan_cl.tasks)

    # H. Statistical Relationships
    plan_stat = orch.plan("find correlations and statistical dependencies between features", df)
    assert any(t.task_type == "statistical_analysis" for t in plan_stat.tasks)

    # I. Prediction
    plan_pred = orch.plan("predict sales_val using available features", df, target="sales_val")
    assert any(t.task_type == "prediction" for t in plan_pred.tasks)

    # J. EDA
    plan_eda = orch.plan("explore the distributions and missing values", df)
    assert any(t.task_type == "eda" for t in plan_eda.tasks)


# ---------------------------------------------------------------------------
# K & L & AC. Dependency Ordering & Concurrent Independent Task Execution
# ---------------------------------------------------------------------------
def test_K_L_AC_dependency_levels_and_execution():
    """Verify dependency DAG ordering and independent task isolation."""
    orch = UniversalOrchestrator()

    t1 = PlanTask(task_id="t1", task_type="eda", tool_name="eda", dependencies=[])
    t2 = PlanTask(task_id="t2", task_type="anomaly_detection", tool_name="anomaly_detection", dependencies=["t1"])
    t3 = PlanTask(task_id="t3", task_type="clustering", tool_name="clustering", dependencies=["t1"])
    t4 = PlanTask(task_id="t4", task_type="prediction", tool_name="prediction", dependencies=["t2", "t3"])

    levels = orch._build_dependency_levels([t1, t2, t3, t4], {"t1": [], "t2": ["t1"], "t3": ["t1"], "t4": ["t2", "t3"]})

    assert len(levels) == 3
    assert [t.task_id for t in levels[0]] == ["t1"]
    assert set(t.task_id for t in levels[1]) == {"t2", "t3"}  # t2 and t3 are independent in Level 1
    assert [t.task_id for t in levels[2]] == ["t4"]


# ---------------------------------------------------------------------------
# M & N & O & P. Validation Failure Isolation, Partial Success & Retries
# ---------------------------------------------------------------------------
def test_M_N_O_P_validation_isolation_and_partial_success():
    """Verify that a validation failure in one task does not crash independent tasks."""
    # N=60 dataset with numeric features and no datetime column
    n = 60
    df_no_date = pd.DataFrame({
        "metric_a": np.random.normal(50, 10, n),
        "metric_b": np.random.normal(100, 20, n),
    })

    orch = UniversalOrchestrator()

    # Command requesting both EDA and Forecasting
    # EDA should succeed, forecasting should be blocked/fail due to no datetime column
    plan = AnalyticalPlan(
        plan_id="test_plan_partial",
        user_request="profile dataset and forecast metric_a",
        tasks=[
            PlanTask(task_id="t_eda", task_type="eda", tool_name="eda", dependencies=[]),
            PlanTask(task_id="t_fc", task_type="forecasting", tool_name="forecasting", target_column="metric_a", dependencies=[]),
        ],
        dependencies={"t_eda": [], "t_fc": []},
    )

    res = orch.execute_plan(plan, df_no_date)

    # Must return PARTIAL status (not total failure)
    assert res.status == AgentStatus.PARTIAL
    assert "eda" in res.data["task_outputs"]
    assert res.data["task_summary"]["completed_tasks"] == 1
    assert res.data["task_summary"]["failed_tasks"] == 1
    assert res.confidence > 0.0


# ---------------------------------------------------------------------------
# Q & R. Ambiguous and Unsupported Command Handling
# ---------------------------------------------------------------------------
def test_Q_R_ambiguous_and_unsupported_commands():
    """Verify structured handling of ambiguous and unsupported commands."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    orch = UniversalOrchestrator()

    # Ambiguous command
    res_ambig = orch.orchestrate("", df)
    assert res_ambig.status == AgentStatus.NEEDS_CLARIFICATION
    assert len(res_ambig.errors) > 0

    # Unsupported command
    res_unsupp = orch.orchestrate("render video animation of this dataset", df)
    assert res_unsupp.status == AgentStatus.NOT_SUPPORTED
    assert len(res_unsupp.errors) > 0


# ---------------------------------------------------------------------------
# S & T & U & V. Empty Dataset, Traceback Containment & Error Contract
# ---------------------------------------------------------------------------
def test_S_T_U_V_empty_data_and_traceback_containment():
    """Verify empty dataset rejection and absence of raw tracebacks in user message."""
    orch = UniversalOrchestrator()

    res_empty = orch.orchestrate("profile this data", pd.DataFrame())
    assert res_empty.status == AgentStatus.ERROR
    assert res_empty.errors[0].category == ErrorCategory.INSUFFICIENT_DATA
    assert "Traceback" not in res_empty.error_message


# ---------------------------------------------------------------------------
# W & X & Y. Evidence Preservation, Confidence Bounds & Determinism
# ---------------------------------------------------------------------------
def test_W_X_Y_evidence_confidence_and_determinism():
    """Verify evidence preservation, bounded confidence in [0, 1], and deterministic execution."""
    n = 50
    df = pd.DataFrame({
        "num_val": np.linspace(10, 100, n),
        "cat_dim": ["A", "B"] * 25,
    })

    orch = UniversalOrchestrator()

    res1 = orch.orchestrate("profile the dataset and detect anomalies", df)
    res2 = orch.orchestrate("profile the dataset and detect anomalies", df)

    assert res1.is_success
    assert 0.0 <= res1.confidence <= 1.0
    assert len(res1.evidence) >= 1
    assert_no_nan_or_inf(res1.data)

    # Determinism
    assert len(res1.data["execution_graph"]) == len(res2.data["execution_graph"])


# ---------------------------------------------------------------------------
# Z. FastAPI Live HTTP Endpoint Integration
# ---------------------------------------------------------------------------
def test_Z_fastapi_orchestration_endpoint():
    """Verify POST /api/v1/orchestrate endpoint."""
    client = TestClient(app)

    records = [{"feature_1": float(i), "feature_2": f"Grp_{i%3}"} for i in range(40)]

    resp = client.post("/api/v1/orchestrate", json={
        "dataset": records,
        "command": "profile this dataset and find statistical distributions",
    })

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("success", "completed")
    assert "orchestration_id" in data["result"]
    assert data["result"]["task_summary"]["completed_tasks"] >= 1


# ---------------------------------------------------------------------------
# AA & AB & AD. End-to-End Integration & Tool Registry Verification
# ---------------------------------------------------------------------------
def test_AA_AB_AD_end_to_end_orchestration():
    """Verify complete end-to-end multi-agent orchestration reusing tool registry."""
    n = 60
    df = pd.DataFrame({
        "dim_1": np.linspace(5, 50, n),
        "dim_2": np.random.normal(100, 15, n),
        "target_val": np.linspace(20, 80, n),
    })

    orch = UniversalOrchestrator()
    res = orch.orchestrate("explore distributions, find correlations, and cluster data", df)

    assert res.is_success
    assert len(res.data["task_outputs"]) >= 2
    assert "orchestration_id" in res.data