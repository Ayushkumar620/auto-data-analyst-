"""
Comprehensive Test Suite for Milestone 7 — Task 3:
Universal Conversational Analytical Context & Session Memory Layer.

Tests:
A. New session creation
B. Dataset context creation
C. Dataset continuity
D. Target continuity
E. Feature continuity
F. Time-column continuity
G. Previous-result reference
H. "it" resolution
I. "that" resolution
J. "those features" resolution
K. "the target" resolution
L. Forecast horizon modification
M. Cluster reference resolution
N. Relationship reference resolution
O. Multiple datasets in one session
P. Dataset switching
Q. Context invalidation
R. Failed execution handling
S. Missing dataset clarification
T. Ambiguous reference clarification
U. Session isolation
V. Bounded history
W. No raw DataFrame storage in context
X. AgentResult compatibility
Y. execution_id continuity
Z. API follow-up execution
AA. Natural-language follow-up routing
AB. Context-aware orchestration
AC. Frontend/backend session contract
AD. Deterministic reference resolution
"""
import math
import uuid
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.agent_result import AgentResult, AgentStatus, ClaimType, Evidence
from agent.analytical_context import (
    AnalyticalContext,
    DatasetSnapshot,
    DEFAULT_SESSION_CONTEXT_MANAGER,
    ExecutionRecord,
    SessionContextManager,
    UniversalReferenceResolver,
)
from agent.orchestrator import UniversalOrchestrator
from backend.app.main import app


# ------------------------------------------------------------------------------
# Test A: New Session Creation
# ------------------------------------------------------------------------------
def test_A_new_session_creation():
    mgr = SessionContextManager()
    session_id = "test_session_a"
    ctx = mgr.get_or_create_context(session_id)
    assert isinstance(ctx, AnalyticalContext)
    assert ctx.session_id == session_id
    assert ctx.created_at is not None
    assert len(ctx.datasets) == 0
    assert len(ctx.execution_history) == 0


# ------------------------------------------------------------------------------
# Test B: Dataset Context Creation
# ------------------------------------------------------------------------------
def test_B_dataset_context_creation():
    mgr = SessionContextManager()
    session_id = "test_session_b"
    df = pd.DataFrame({
        "num_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cat_1": ["A", "B", "A", "B", "A"],
        "date_col": pd.date_range("2023-01-01", periods=5),
    })
    snapshot = mgr.register_dataset(session_id, df, dataset_id="ds_b", dataset_name="custom_dataset")
    assert isinstance(snapshot, DatasetSnapshot)
    assert snapshot.dataset_id == "ds_b"
    assert snapshot.dataset_name == "custom_dataset"
    assert "num_1" in snapshot.numeric_columns
    assert "cat_1" in snapshot.categorical_columns
    assert "date_col" in snapshot.datetime_columns
    assert snapshot.original_rows == 5


# ------------------------------------------------------------------------------
# Test C: Dataset Continuity
# ------------------------------------------------------------------------------
def test_C_dataset_continuity():
    mgr = SessionContextManager()
    session_id = "test_session_c"
    df = pd.DataFrame({"x": range(10), "y": range(10)})
    mgr.register_dataset(session_id, df, dataset_id="ds_c")

    retrieved = mgr.get_dataset(session_id)
    assert retrieved is not None
    assert len(retrieved) == 10
    assert list(retrieved.columns) == ["x", "y"]


# ------------------------------------------------------------------------------
# Test D: Target Continuity
# ------------------------------------------------------------------------------
def test_D_target_continuity():
    mgr = SessionContextManager()
    session_id = "test_session_d"
    ctx = mgr.get_or_create_context(session_id)
    res = AgentResult.success(output={"metrics": {}}, agent_name="Predictor")
    mgr.record_execution(session_id, res, user_command="predict sales", target="sales")

    assert ctx.active_target == "sales"
    res2 = mgr.resolver.resolve("which feature predicts it best?", ctx)
    assert res2.target == "sales"
    assert "sales" in res2.resolved_command


# ------------------------------------------------------------------------------
# Test E: Feature Continuity
# ------------------------------------------------------------------------------
def test_E_feature_continuity():
    mgr = SessionContextManager()
    session_id = "test_session_e"
    ctx = mgr.get_or_create_context(session_id)
    res = AgentResult.success(output={"metrics": {}}, agent_name="Analyzer")
    mgr.record_execution(session_id, res, user_command="analyze features", features=["price", "cost", "quantity"])

    assert ctx.active_features == ["price", "cost", "quantity"]
    res2 = mgr.resolver.resolve("train a model with those features", ctx)
    assert res2.is_follow_up is True
    assert "price, cost, quantity" in res2.resolved_command


# ------------------------------------------------------------------------------
# Test F: Time-Column Continuity
# ------------------------------------------------------------------------------
def test_F_time_column_continuity():
    mgr = SessionContextManager()
    session_id = "test_session_f"
    df = pd.DataFrame({"date_idx": pd.date_range("2024-01-01", periods=20), "val": range(20)})
    mgr.register_dataset(session_id, df, dataset_id="ds_f")
    ctx = mgr.get_context(session_id)

    assert ctx.active_time_column == "date_idx"
    res = mgr.resolver.resolve("forecast next 5 periods", ctx)
    assert res.time_column == "date_idx"


# ------------------------------------------------------------------------------
# Test G: Previous-Result Reference
# ------------------------------------------------------------------------------
def test_G_previous_result_reference():
    mgr = SessionContextManager()
    session_id = "test_session_g"
    ctx = mgr.get_or_create_context(session_id)
    res = AgentResult.success(output={"metrics": {}}, agent_name="ClusteringAgent", task_type="clustering")
    mgr.record_execution(session_id, res, user_command="segment customers")

    assert ctx.active_task == "clustering"
    res2 = mgr.resolver.resolve("tell me more about that", ctx)
    assert res2.is_follow_up is True
    assert "clustering" in res2.resolved_command


# ------------------------------------------------------------------------------
# Test H: "it" Resolution
# ------------------------------------------------------------------------------
def test_H_it_resolution():
    mgr = SessionContextManager()
    session_id = "test_session_h"
    ctx = mgr.get_or_create_context(session_id)
    res = AgentResult.success(output={}, agent_name="Predictor")
    mgr.record_execution(session_id, res, user_command="train model on revenue", target="revenue")

    res2 = mgr.resolver.resolve("forecast it", ctx)
    assert res2.is_follow_up is True
    assert "forecast revenue" in res2.resolved_command.lower()


# ------------------------------------------------------------------------------
# Test I: "that" Resolution
# ------------------------------------------------------------------------------
def test_I_that_resolution():
    mgr = SessionContextManager()
    session_id = "test_session_i"
    ctx = mgr.get_or_create_context(session_id)
    res = AgentResult.success(
        output={"tasks": {"statistical_analysis": {"ranked_relationships": [{"feature_1": "temperature", "feature_2": "ice_cream_sales", "correlation": 0.88}]}}},
        agent_name="StatisticalAgent",
        task_type="statistical_analysis",
    )
    mgr.record_execution(session_id, res, user_command="find correlations")

    res2 = mgr.resolver.resolve("why is that relationship important?", ctx)
    assert res2.is_follow_up is True
    assert "temperature" in res2.resolved_command or "ice_cream_sales" in res2.resolved_command


# ------------------------------------------------------------------------------
# Test J: "those features" Resolution
# ------------------------------------------------------------------------------
def test_J_those_features_resolution():
    mgr = SessionContextManager()
    session_id = "test_session_j"
    ctx = mgr.get_or_create_context(session_id)
    res = AgentResult.success(output={}, agent_name="Analyzer")
    mgr.record_execution(session_id, res, user_command="select features", features=["f_alpha", "f_beta"])

    res2 = mgr.resolver.resolve("predict using those features", ctx)
    assert "f_alpha, f_beta" in res2.resolved_command


# ------------------------------------------------------------------------------
# Test K: "the target" Resolution
# ------------------------------------------------------------------------------
def test_K_the_target_resolution():
    mgr = SessionContextManager()
    session_id = "test_session_k"
    ctx = mgr.get_or_create_context(session_id)
    res = AgentResult.success(output={}, agent_name="Predictor")
    mgr.record_execution(session_id, res, user_command="train on profit", target="profit")

    res2 = mgr.resolver.resolve("which variables influence the target?", ctx)
    assert "profit" in res2.resolved_command


# ------------------------------------------------------------------------------
# Test L: Forecast Horizon Modification
# ------------------------------------------------------------------------------
def test_L_forecast_horizon_modification():
    mgr = SessionContextManager()
    session_id = "test_session_l"
    ctx = mgr.get_or_create_context(session_id)
    res = AgentResult.success(
        output={"tasks": {"forecasting": {"horizon": 5, "model_selected": "ARIMA"}}},
        agent_name="ForecastAgent",
        task_type="forecasting",
    )
    mgr.record_execution(session_id, res, user_command="forecast 5 periods", target="churn")

    res2 = mgr.resolver.resolve("make it 12", ctx)
    assert res2.is_follow_up is True
    assert res2.parameters.get("horizon") == 12 or res2.parameters.get("periods") == 12
    assert "12" in res2.resolved_command


# ------------------------------------------------------------------------------
# Test M: Cluster Reference Resolution
# ------------------------------------------------------------------------------
def test_M_cluster_reference_resolution():
    mgr = SessionContextManager()
    session_id = "test_session_m"
    ctx = mgr.get_or_create_context(session_id)
    res = AgentResult.success(
        output={"tasks": {"clustering": {"cluster_count": 4}}},
        agent_name="ClusteringAgent",
        task_type="clustering",
    )
    mgr.record_execution(session_id, res, user_command="cluster into 4 groups")

    res2 = mgr.resolver.resolve("focus on cluster 2", ctx)
    assert res2.is_follow_up is True
    assert res2.parameters.get("cluster_id") == 2
    assert "cluster 2" in res2.resolved_command


# ------------------------------------------------------------------------------
# Test N: Relationship Reference Resolution
# ------------------------------------------------------------------------------
def test_N_relationship_reference_resolution():
    mgr = SessionContextManager()
    session_id = "test_session_n"
    ctx = mgr.get_or_create_context(session_id)
    res = AgentResult.success(
        output={
            "tasks": {
                "statistical_analysis": {
                    "ranked_relationships": [
                        {"feature_1": "ad_spend", "feature_2": "revenue", "correlation": 0.92},
                        {"feature_1": "discount", "feature_2": "margin", "correlation": -0.65},
                    ]
                }
            }
        },
        agent_name="StatisticalAgent",
    )
    mgr.record_execution(session_id, res, user_command="find correlations")

    res1 = mgr.resolver.resolve("tell me more about the strongest relationship", ctx)
    assert "ad_spend" in res1.resolved_command and "revenue" in res1.resolved_command

    res2 = mgr.resolver.resolve("what about the second strongest relationship?", ctx)
    assert "discount" in res2.resolved_command and "margin" in res2.resolved_command


# ------------------------------------------------------------------------------
# Test O: Multiple Datasets in One Session
# ------------------------------------------------------------------------------
def test_O_multiple_datasets_in_one_session():
    mgr = SessionContextManager()
    session_id = "test_session_o"
    df1 = pd.DataFrame({"sales": [100, 200, 300]})
    df2 = pd.DataFrame({"customers": ["Alice", "Bob", "Charlie"]})

    mgr.register_dataset(session_id, df1, dataset_id="ds_sales", dataset_name="sales_data")
    mgr.register_dataset(session_id, df2, dataset_id="ds_customers", dataset_name="customer_data")

    ctx = mgr.get_context(session_id)
    assert len(ctx.datasets) == 2
    assert "ds_sales" in ctx.datasets
    assert "ds_customers" in ctx.datasets
    assert ctx.active_dataset_id == "ds_customers"


# ------------------------------------------------------------------------------
# Test P: Dataset Switching
# ------------------------------------------------------------------------------
def test_P_dataset_switching():
    mgr = SessionContextManager()
    session_id = "test_session_p"
    df1 = pd.DataFrame({"sales": [100, 200, 300]})
    df2 = pd.DataFrame({"customers": ["Alice", "Bob", "Charlie"]})

    mgr.register_dataset(session_id, df1, dataset_id="ds_sales", dataset_name="sales_data")
    mgr.register_dataset(session_id, df2, dataset_id="ds_customers", dataset_name="customer_data")

    ctx = mgr.get_context(session_id)
    res = mgr.resolver.resolve("switch to sales_data dataset", ctx)
    assert res.dataset_id == "ds_sales"
    assert res.is_follow_up is True


# ------------------------------------------------------------------------------
# Test Q: Context Invalidation
# ------------------------------------------------------------------------------
def test_Q_context_invalidation():
    mgr = SessionContextManager()
    session_id = "test_session_q"
    ctx = mgr.get_or_create_context(session_id)
    ctx.active_target = "old_target"
    ctx.active_features = ["f1", "f2"]

    mgr.invalidate_target(session_id)
    assert ctx.active_target is None

    mgr.invalidate_features(session_id)
    assert ctx.active_features == []


# ------------------------------------------------------------------------------
# Test R: Failed Execution Handling
# ------------------------------------------------------------------------------
def test_R_failed_execution_handling():
    mgr = SessionContextManager()
    session_id = "test_session_r"
    err_res = AgentResult.error(error="Task failed due to invalid column", agent_name="Tester")
    mgr.record_execution(session_id, err_res, user_command="invalid operation")

    ctx = mgr.get_context(session_id)
    assert len(ctx.execution_history) == 1
    assert ctx.execution_history[0].status == "error"


# ------------------------------------------------------------------------------
# Test S: Missing Dataset Clarification
# ------------------------------------------------------------------------------
def test_S_missing_dataset_clarification():
    orch = UniversalOrchestrator()
    session_id = f"test_session_s_{uuid.uuid4().hex[:6]}"
    res = orch.orchestrate("analyze dataset", data=None, session_id=session_id)
    assert res.status == AgentStatus.NEEDS_CLARIFICATION
    assert "dataset" in res.error_message.lower() or "dataset" in str(res.result).lower()


# ------------------------------------------------------------------------------
# Test T: Ambiguous Reference Clarification
# ------------------------------------------------------------------------------
def test_T_ambiguous_reference_clarification():
    mgr = SessionContextManager()
    session_id = "test_session_t"
    ctx = mgr.get_or_create_context(session_id)
    res1 = AgentResult.success(output={}, agent_name="ForecastAgent", task_type="forecasting")
    res2 = AgentResult.success(output={}, agent_name="StatisticalAgent", task_type="statistical_analysis")
    mgr.record_execution(session_id, res1, user_command="forecast sales")
    mgr.record_execution(session_id, res2, user_command="find correlations")

    resolution = mgr.resolver.resolve("compare that with the other one", ctx)
    assert resolution.needs_clarification is True
    assert len(resolution.suggested_options) >= 2


# ------------------------------------------------------------------------------
# Test U: Session Isolation
# ------------------------------------------------------------------------------
def test_U_session_isolation():
    mgr = SessionContextManager()
    sess_1 = "session_user_1"
    sess_2 = "session_user_2"

    df1 = pd.DataFrame({"user1_metric": [10, 20, 30]})
    df2 = pd.DataFrame({"user2_metric": [100, 200, 300]})

    mgr.register_dataset(sess_1, df1, dataset_id="ds_1")
    mgr.register_dataset(sess_2, df2, dataset_id="ds_2")

    ctx1 = mgr.get_context(sess_1)
    ctx2 = mgr.get_context(sess_2)

    assert "ds_1" in ctx1.datasets and "ds_2" not in ctx1.datasets
    assert "ds_2" in ctx2.datasets and "ds_1" not in ctx2.datasets
    assert mgr.get_dataset(sess_1, "ds_2") is None


# ------------------------------------------------------------------------------
# Test V: Bounded History
# ------------------------------------------------------------------------------
def test_V_bounded_history():
    mgr = SessionContextManager(max_history_per_session=3)
    session_id = "test_session_v"
    ctx = mgr.get_or_create_context(session_id)

    for i in range(7):
        res = AgentResult.success(output={}, agent_name=f"Agent_{i}")
        mgr.record_execution(session_id, res, user_command=f"command_{i}")

    assert len(ctx.execution_history) == 3
    assert ctx.execution_history[-1].user_command == "command_6"


# ------------------------------------------------------------------------------
# Test W: No Raw DataFrame Storage in Context
# ------------------------------------------------------------------------------
def test_W_no_raw_dataframe_storage_in_context():
    mgr = SessionContextManager()
    session_id = "test_session_w"
    df = pd.DataFrame({"big_data": range(1000)})
    mgr.register_dataset(session_id, df, dataset_id="ds_w")

    ctx = mgr.get_context(session_id)
    ctx_dict = ctx.to_dict()
    # Check that raw 1000 items are not inside context serialization
    ds_dict = ctx_dict["datasets"]["ds_w"]
    assert len(ds_dict["preview_sample"]) <= 5
    assert "big_data" in ds_dict["columns"]
    assert ds_dict["original_rows"] == 1000


# ------------------------------------------------------------------------------
# Test X: AgentResult Compatibility
# ------------------------------------------------------------------------------
def test_X_agentresult_compatibility():
    orch = UniversalOrchestrator()
    session_id = "test_session_x"
    df = pd.DataFrame({"a": range(20), "b": [x * 2 for x in range(20)]})
    res = orch.orchestrate("analyze relationships", df, session_id=session_id)

    assert isinstance(res, AgentResult)
    assert res.provenance.get("session_id") == session_id
    assert res.status == AgentStatus.COMPLETED
    assert res.status in (AgentStatus.SUCCESS, AgentStatus.COMPLETED)


# ------------------------------------------------------------------------------
# Test Y: Execution ID Continuity
# ------------------------------------------------------------------------------
def test_Y_execution_id_continuity():
    orch = UniversalOrchestrator()
    session_id = f"test_session_y_{uuid.uuid4().hex[:6]}"
    df = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=30), "metric": range(30)})

    res1 = orch.orchestrate("profile dataset", df, session_id=session_id)
    exec_id_1 = res1.execution_id

    ctx = DEFAULT_SESSION_CONTEXT_MANAGER.get_context(session_id)
    assert ctx.last_execution_id == exec_id_1

    res2 = orch.orchestrate("forecast metric for 5 periods", session_id=session_id)
    exec_id_2 = res2.execution_id

    assert exec_id_1 != exec_id_2
    assert ctx.last_execution_id == exec_id_2
    assert len(ctx.execution_history) == 2


# ------------------------------------------------------------------------------
# Test Z: API Follow-up Execution
# ------------------------------------------------------------------------------
def test_Z_api_follow_up_execution():
    client = TestClient(app)
    session_id = f"api_session_{uuid.uuid4().hex[:6]}"
    records = [{"x": i, "y": i * 3} for i in range(25)]

    # Turn 1: Initial upload & analysis
    r1 = client.post(
        "/api/v1/orchestrate",
        json={"command": "profile this data and analyze correlation", "dataset": records, "session_id": session_id},
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["status"] == "completed"
    assert d1["status"] in ("success", "completed")

    # Turn 2: Follow-up without resending dataset
    r2 = client.post(
        "/api/v1/orchestrate",
        json={"command": "tell me more about the strongest relationship", "session_id": session_id},
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["status"] == "completed"
    assert d2["status"] in ("success", "completed")
    assert d2.get("result", {}).get("is_follow_up") is True or d2.get("provenance", {}).get("is_follow_up") is True


# ------------------------------------------------------------------------------
# Test AA: Natural-Language Follow-up Routing
# ------------------------------------------------------------------------------
def test_AA_natural_language_follow_up_routing():
    mgr = SessionContextManager()
    session_id = "test_session_aa"
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=40),
        "sales": np.linspace(100, 200, 40),
        "cost": np.random.normal(50, 5, 40),
    })
    mgr.register_dataset(session_id, df, dataset_id="ds_aa")
    ctx = mgr.get_context(session_id)
    ctx.active_target = "sales"
    ctx.previous_task = "forecasting"

    res = mgr.resolver.resolve("increase horizon to 8 periods", ctx)
    assert res.detected_intent == "forecasting"
    assert res.target == "sales"
    assert res.parameters.get("horizon") == 8


# ------------------------------------------------------------------------------
# Test AB: Context-Aware Orchestration
# ------------------------------------------------------------------------------
def test_AB_context_aware_orchestration():
    orch = UniversalOrchestrator()
    session_id = f"test_session_ab_{uuid.uuid4().hex[:6]}"
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=30),
        "revenue": np.linspace(500, 1000, 30),
        "ads": np.random.normal(100, 10, 30),
    })

    # 1. First command
    res1 = orch.orchestrate("analyze correlation with revenue", df, session_id=session_id, target="revenue")
    assert res1.status == AgentStatus.COMPLETED
    assert res1.status in (AgentStatus.SUCCESS, AgentStatus.COMPLETED)

    # 2. Contextual follow-up command
    res2 = orch.orchestrate("forecast it for the next 6 periods", session_id=session_id)
    assert res2.status == AgentStatus.COMPLETED
    assert res2.status in (AgentStatus.SUCCESS, AgentStatus.COMPLETED)
    assert res2.provenance.get("is_follow_up") is True


# ------------------------------------------------------------------------------
# Test AC: Frontend/Backend Session Contract
# ------------------------------------------------------------------------------
def test_AC_frontend_backend_session_contract():
    client = TestClient(app)
    session_id = f"contract_session_{uuid.uuid4().hex[:6]}"
    records = [{"val": i} for i in range(15)]

    # Initial command
    client.post(
        "/api/v1/orchestrate",
        json={"command": "profile data", "dataset": records, "session_id": session_id},
    )

    # Fetch context endpoint
    r = client.get(f"/api/v1/orchestrate/context/{session_id}")
    assert r.status_code == 200
    ctx_data = r.json()
    assert ctx_data["session_id"] == session_id
    assert len(ctx_data["execution_history"]) >= 1

    # Clear context endpoint
    r_del = client.delete(f"/api/v1/orchestrate/context/{session_id}")
    assert r_del.status_code == 200
    assert r_del.json()["status"] == "cleared"


# ------------------------------------------------------------------------------
# Test AD: Deterministic Reference Resolution
# ------------------------------------------------------------------------------
def test_AD_deterministic_reference_resolution():
    resolver = UniversalReferenceResolver()
    ctx = AnalyticalContext(session_id="det_session")
    ctx.active_target = "profit"
    ctx.active_features = ["c1", "c2"]

    res1 = resolver.resolve("predict it using those features", ctx)
    res2 = resolver.resolve("predict it using those features", ctx)

    assert res1.resolved_command == res2.resolved_command
    assert res1.target == res2.target == "profit"
    assert res1.features == res2.features == ["c1", "c2"]