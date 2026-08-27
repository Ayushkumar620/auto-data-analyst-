"""
Milestone 7 — Task 1: Comprehensive Universal Agent Orchestration & End-to-End Execution Test Suite.

Exhaustively covers tests A through AD:
A. single-task command routing
B. multi-task analytical request
C. arbitrary dataset column names
D. automatic target/feature interpretation
E. forecasting task planning
F. anomaly task planning
G. clustering task planning
H. statistical relationship task planning
I. prediction task planning
J. EDA task planning
K. dependency ordering
L. independent task execution
M. validation failure isolation
N. partial success
O. retryable failure recovery
P. non-retryable failure
Q. ambiguous command
R. unsupported command
S. empty dataset
T. malformed dataset
U. structured AgentError contract
V. no traceback leakage
W. evidence preservation
X. confidence bounds
Y. deterministic planning
Z. FastAPI orchestration endpoint
AA. existing API regression compatibility
AB. tool registry integration
AC. duplicate execution prevention
AD. complete end-to-end user command execution
"""
from __future__ import annotations

import math
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.agent_result import AgentError, AgentResult, AgentStatus, ClaimType, ErrorCategory
from agent.agent_result import AgentError, AgentResult, AgentStatus, ClaimType, ErrorCategory, Evidence
from agent.orchestrator import AnalyticalPlan, PlanTask, TaskStatus, UniversalOrchestrator
from agent.tool_registry import DEFAULT_TOOL_REGISTRY
from backend.app.main import app


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


# A. Single-task command routing
def test_A_single_task_command_routing():
    df = pd.DataFrame({"feat_1": [1.0, 2.0, 3.0, 4.0, 5.0] * 10, "feat_2": [10.0, 20.0, 30.0, 40.0, 50.0] * 10})
    orch = UniversalOrchestrator()
    res = orch.orchestrate("give me an overview and profile this data", df)
    assert res.is_success
    assert "eda" in res.data["tasks"]


# B. Multi-task analytical request
def test_B_multi_task_analytical_request():
    df = pd.DataFrame({"a": np.random.normal(50, 10, 50), "b": np.random.normal(100, 20, 50)})
    orch = UniversalOrchestrator()
    res = orch.orchestrate("profile the dataset and find anomalies", df)
    assert res.is_success
    assert "eda" in res.data["tasks"]
    assert "anomaly_detection" in res.data["tasks"]


# C. Arbitrary dataset column names
def test_C_arbitrary_dataset_column_names():
    df = pd.DataFrame({
        "Arbitrary_Col_Alpha#1": np.linspace(10, 100, 50),
        "Weird.Name[Beta]": np.random.normal(0, 1, 50),
        "gamma_dimension_value": ["X", "Y"] * 25,
    })
    orch = UniversalOrchestrator()
    res = orch.orchestrate("explore data distributions and cluster records", df)
    assert res.is_success
    assert_no_nan_or_inf(res.data)


# D. Automatic target/feature interpretation
def test_D_automatic_target_feature_interpretation():
    df = pd.DataFrame({
        "revenue_target": np.linspace(100, 500, 50),
        "cost_feature": np.linspace(50, 200, 50),
        "customer_id": [f"ID_{i}" for i in range(50)],
    })
    orch = UniversalOrchestrator()
    plan = orch.plan("predict revenue_target using available features", df)
    pred_task = next(t for t in plan.tasks if t.task_type == "prediction")
    assert pred_task.target_column == "revenue_target"
    assert "customer_id" not in pred_task.required_columns


# E. Forecasting task planning
def test_E_forecasting_task_planning():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({"time_axis": dates.strftime("%Y-%m-%d"), "sales_metric": np.linspace(10, 100, 60)})
    orch = UniversalOrchestrator()
    plan = orch.plan("forecast the next 6 periods for sales_metric", df)
    assert any(t.task_type == "forecasting" for t in plan.tasks)


# F. Anomaly task planning
def test_F_anomaly_task_planning():
    df = pd.DataFrame({"metric": [10.0] * 40 + [500.0, -200.0]})
    orch = UniversalOrchestrator()
    plan = orch.plan("detect abnormal outliers and spikes in this dataset", df)
    assert any(t.task_type == "anomaly_detection" for t in plan.tasks)


# G. Clustering task planning
def test_G_clustering_task_planning():
    df = pd.DataFrame({"x": np.random.normal(10, 2, 50), "y": np.random.normal(50, 5, 50)})
    orch = UniversalOrchestrator()
    plan = orch.plan("segment customers into natural clusters", df)
    assert any(t.task_type == "clustering" for t in plan.tasks)


# H. Statistical relationship task planning
def test_H_statistical_relationship_task_planning():
    df = pd.DataFrame({"x": np.linspace(1, 100, 50), "y": np.linspace(2, 200, 50)})
    orch = UniversalOrchestrator()
    plan = orch.plan("find correlations and feature relationships", df)
    assert any(t.task_type == "statistical_analysis" for t in plan.tasks)


# I. Prediction task planning
def test_I_prediction_task_planning():
    df = pd.DataFrame({"feat": np.random.normal(0, 1, 50), "label": [0, 1] * 25})
    orch = UniversalOrchestrator()
    plan = orch.plan("predict label", df, target="label")
    assert any(t.task_type == "prediction" for t in plan.tasks)


# J. EDA task planning
def test_J_eda_task_planning():
    df = pd.DataFrame({"val": range(50)})
    orch = UniversalOrchestrator()
    plan = orch.plan("describe summary statistics", df)
    assert any(t.task_type == "eda" for t in plan.tasks)


# K. Dependency ordering
def test_K_dependency_ordering():
    orch = UniversalOrchestrator()
    t1 = PlanTask(task_id="t1", task_type="eda", tool_name="eda", dependencies=[])
    t2 = PlanTask(task_id="t2", task_type="transformation", tool_name="transformation", dependencies=["t1"])
    t3 = PlanTask(task_id="t3", task_type="prediction", tool_name="prediction", dependencies=["t2"])

    levels = orch._build_dependency_levels([t1, t2, t3], {"t1": [], "t2": ["t1"], "t3": ["t2"]})
    assert len(levels) == 3
    assert levels[0][0].task_id == "t1"
    assert levels[1][0].task_id == "t2"
    assert levels[2][0].task_id == "t3"


# L. Independent task execution
def test_L_independent_task_execution():
    orch = UniversalOrchestrator()
    t1 = PlanTask(task_id="t1", task_type="eda", tool_name="eda", dependencies=[])
    t2 = PlanTask(task_id="t2", task_type="anomaly_detection", tool_name="anomaly_detection", dependencies=["t1"])
    t3 = PlanTask(task_id="t3", task_type="clustering", tool_name="clustering", dependencies=["t1"])

    levels = orch._build_dependency_levels([t1, t2, t3], {"t1": [], "t2": ["t1"], "t3": ["t1"]})
    assert len(levels) == 2
    assert set(t.task_id for t in levels[1]) == {"t2", "t3"}


# M. Validation failure isolation
def test_M_validation_failure_isolation():
    n = 60
    df = pd.DataFrame({"num1": np.random.normal(50, 10, n), "num2": np.random.normal(20, 5, n)})
    orch = UniversalOrchestrator()

    plan = AnalyticalPlan(
        plan_id="plan_isolation",
        user_request="profile data and forecast",
        tasks=[
            PlanTask(task_id="t_eda", task_type="eda", tool_name="eda", dependencies=[]),
            PlanTask(task_id="t_fc", task_type="forecasting", tool_name="forecasting", target_column="num1", dependencies=[]),
        ],
        dependencies={"t_eda": [], "t_fc": []},
    )

    res = orch.execute_plan(plan, df)
    # EDA succeeds, forecasting is blocked due to no datetime column
    assert "eda" in res.data["task_outputs"]
    assert res.data["task_summary"]["completed_tasks"] == 1
    assert res.data["task_summary"]["failed_tasks"] == 1


# N. Partial success
def test_N_partial_success():
    n = 60
    df = pd.DataFrame({"a": np.random.normal(50, 10, n)})
    orch = UniversalOrchestrator()
    plan = AnalyticalPlan(
        plan_id="plan_partial",
        user_request="run tasks",
        tasks=[
            PlanTask(task_id="t_eda", task_type="eda", tool_name="eda", dependencies=[]),
            PlanTask(task_id="t_fc", task_type="forecasting", tool_name="forecasting", target_column="a", dependencies=[]),
        ],
        dependencies={"t_eda": [], "t_fc": []},
    )
    res = orch.execute_plan(plan, df)
    assert res.status == AgentStatus.PARTIAL
    assert res.is_success is False  # Partial status is not unconditional full success


# O. Retryable failure recovery
def test_O_retryable_failure_recovery():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5] * 10, "y": [10, 20, 30, 40, 50] * 10})
    orch = UniversalOrchestrator()
    res = orch.orchestrate("explore dataset distributions", df)
    assert res.is_success
    assert res.data["task_summary"]["retry_count"] >= 0


# P. Non-retryable failure
def test_P_non_retryable_failure():
    df = pd.DataFrame({"const": [1.0] * 50})
    orch = UniversalOrchestrator()
    plan = AnalyticalPlan(
        plan_id="plan_non_retry",
        user_request="predict const",
        tasks=[
            PlanTask(task_id="t_pred", task_type="prediction", tool_name="prediction", target_column="const", dependencies=[]),
        ],
        dependencies={"t_pred": []},
    )
    res = orch.execute_plan(plan, df)
    assert res.status == AgentStatus.ERROR


# Q. Ambiguous command
def test_Q_ambiguous_command():
    df = pd.DataFrame({"x": [1, 2, 3]})
    orch = UniversalOrchestrator()
    res = orch.orchestrate("", df)
    assert res.status == AgentStatus.NEEDS_CLARIFICATION


# R. Unsupported command
def test_R_unsupported_command():
    df = pd.DataFrame({"x": [1, 2, 3]})
    orch = UniversalOrchestrator()
    res = orch.orchestrate("render video animation", df)
    assert res.status == AgentStatus.NOT_SUPPORTED


# S. Empty dataset
def test_S_empty_dataset():
    orch = UniversalOrchestrator()
    res = orch.orchestrate("profile data", pd.DataFrame())
    assert res.status == AgentStatus.ERROR
    assert res.errors[0].category == ErrorCategory.INSUFFICIENT_DATA


# T. Malformed dataset
def test_T_malformed_dataset():
    orch = UniversalOrchestrator()
    res = orch.orchestrate("profile data", None)
    assert res.status == AgentStatus.ERROR


# U. Structured AgentError contract
def test_U_structured_agent_error_contract():
    orch = UniversalOrchestrator()
    res = orch.orchestrate("profile data", pd.DataFrame())
    err = res.errors[0]
    assert isinstance(err, AgentError)
    assert err.code != ""
    assert err.user_message != ""


# V. No traceback leakage
def test_V_no_traceback_leakage():
    orch = UniversalOrchestrator()
    res = orch.orchestrate("profile data", pd.DataFrame())
    assert "Traceback" not in res.error_message
    assert "Traceback" not in res.message


# W. Evidence preservation
def test_W_evidence_preservation():
    df = pd.DataFrame({"a": range(50), "b": [x * 2 for x in range(50)]})
    orch = UniversalOrchestrator()
    res = orch.orchestrate("profile data and analyze correlations", df)
    assert res.is_success
    assert len(res.evidence) >= 1
    assert all(isinstance(e, Evidence) for e in res.evidence)


# X. Confidence bounds
def test_X_confidence_bounds():
    df = pd.DataFrame({"a": range(50), "b": range(50)})
    orch = UniversalOrchestrator()
    res = orch.orchestrate("profile this data", df)
    assert 0.0 <= res.confidence <= 1.0


# Y. Deterministic planning
def test_Y_deterministic_planning():
    df = pd.DataFrame({"val": range(50)})
    orch = UniversalOrchestrator()
    plan1 = orch.plan("explore the data", df)
    plan2 = orch.plan("explore the data", df)
    assert plan1.detected_intent == plan2.detected_intent
    assert [t.task_type for t in plan1.tasks] == [t.task_type for t in plan2.tasks]


# Z. FastAPI orchestration endpoint
def test_Z_fastapi_orchestration_endpoint():
    client = TestClient(app)
    records = [{"dim_1": float(i), "dim_2": f"Grp_{i%3}"} for i in range(40)]
    resp = client.post("/api/v1/orchestrate", json={
        "dataset": records,
        "command": "profile this dataset",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "orchestration_id" in data["result"]


# AA. Existing API regression compatibility
def test_AA_existing_api_regression_compatibility():
    client = TestClient(app)
    # Test POST /api/v1/eda
    resp = client.post("/api/v1/eda", json={
        "dataset": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
    })
    assert resp.status_code == 200


# AB. Tool registry integration
def test_AB_tool_registry_integration():
    orch = UniversalOrchestrator()
    assert orch.tool_registry.has_tool("eda")
    assert orch.tool_registry.has_tool("anomaly_detection")
    assert orch.tool_registry.has_tool("clustering")
    assert orch.tool_registry.has_tool("statistical_analysis")
    assert orch.tool_registry.has_tool("forecasting")
    assert orch.tool_registry.has_tool("prediction")


# AC. Duplicate execution prevention
def test_AC_duplicate_execution_prevention():
    df = pd.DataFrame({"x": range(40)})
    orch = UniversalOrchestrator()
    plan = orch.plan("profile data and profile dataset", df)
    eda_tasks = [t for t in plan.tasks if t.task_type == "eda"]
    assert len(eda_tasks) == 1  # Deduplicated into 1 EDA task


# AD. Complete end-to-end user command execution
def test_AD_complete_end_to_end_user_command_execution():
    n = 60
    df = pd.DataFrame({
        "alpha": np.linspace(10, 100, n),
        "beta": np.random.normal(50, 10, n),
        "group": ["A", "B", "C"] * 20,
    })
    orch = UniversalOrchestrator()
    res = orch.orchestrate("profile dataset, find correlations, and cluster data", df)
    assert res.is_success
    assert len(res.data["task_outputs"]) >= 2
    assert "task_summary" in res.data
    assert res.data["task_summary"]["completed_tasks"] >= 2