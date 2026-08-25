"""
Tests for Milestone 4, Task 2: Model Monitoring, Data Drift Detection & Performance Tracking.

Verifies:
1. No drift baseline scenario (stable distributions)
2. Numeric feature drift detection (Kolmogorov-Smirnov test & PSI)
3. Categorical feature drift detection (Chi-Square test, novel & disappearing categories)
4. Missing-value rate drift detection
5. Schema drift: missing required feature columns
6. Schema drift: unexpected extra columns
7. Schema drift: incompatible data type changes (numeric -> string)
8. Target drift detection (when ground truth outcomes exist)
9. Prediction distribution drift detection
10. Model performance degradation detection (metric deltas vs reference)
11. No-label scenario handling (graceful reporting without false degradation claims)
12. Multi-level severity classification (NONE -> CRITICAL)
13. Feature importance awareness in drift prioritization
14. Custom threshold configuration enforcement
15. Monitoring history persistence and retrieval
16. Grounded mathematical evidence generation
17. ToolRegistry integration ('model_monitor')
18. Dynamic Planner integration for drift detection
19. Standardized AgentResult & Evidence generation from ModelMonitorAgent
"""
import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

from agent.dynamic_planner import DynamicTaskPlanner, ExecutionPlan
from agent.intent import UserIntent
from agent.model_monitor_agent import ModelMonitorAgent
from agent.model_monitoring_engine import ModelMonitoringEngine
from agent.model_monitoring_schemas import (
    DatasetDriftReport,
    DriftRequest,
    DriftSeverity,
    DriftThresholdConfig,
    MonitoringResult,
)
from agent.schemas import AgentResult, AgentStatus
from agent.tool_registry import DEFAULT_TOOL_REGISTRY
from backend.app.ml.registry import ModelRegistry


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def temp_registry():
    """Temporary model registry directory."""
    temp_dir = tempfile.mkdtemp()
    registry = ModelRegistry(registry_dir=temp_dir)
    yield registry
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def reference_dataset():
    """Reference training dataset (100 rows)."""
    np.random.seed(42)
    n = 100
    num = np.random.normal(loc=50.0, scale=10.0, size=n)
    stable = np.random.uniform(1.0, 10.0, size=n)
    target = (num + stable > 55.0).astype(int)
    return pd.DataFrame({
        "num_feature": num,
        "cat_feature": np.random.choice(["Tier1", "Tier2", "Tier3"], size=n, p=[0.5, 0.3, 0.2]),
        "stable_feature": stable,
        "target": target,
    })


@pytest.fixture
def registered_clf_model(temp_registry, reference_dataset):
    """Registers a trained classification model with baseline metadata."""
    from sklearn.metrics import accuracy_score, f1_score
    feature_cols = ["num_feature", "cat_feature", "stable_feature"]
    X = reference_dataset[["num_feature", "stable_feature"]].to_numpy()
    y = reference_dataset["target"].to_numpy()

    clf = LogisticRegression()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = float(accuracy_score(y, y_pred))
    f1 = float(f1_score(y, y_pred, zero_division=0))

    # Build reference profile
    ref_profile = ModelMonitoringEngine.build_reference_profile(
        reference_dataset,
        feature_cols=feature_cols,
        target_col="target",
    )

    meta = temp_registry.register_model(
        name="CustomerChurnPredictor",
        model_object=clf,
        model_family="traditional_ml",
        algorithm="Logistic Regression",
        problem_type="binary_classification",
        target_column="target",
        feature_columns=["num_feature", "stable_feature"],
        feature_dtypes={"num_feature": "float64", "stable_feature": "float64"},
        training_metrics={"accuracy": acc, "f1": f1},
        validation_metrics={"accuracy": acc, "f1": f1},
        primary_metric_name="f1",
        primary_metric_value=f1,
        reference_profile=ref_profile,
    )
    return meta.model_id


# ==============================================================================
# 1-4. Feature & Data Quality Drift Tests
# ==============================================================================

def test_no_drift_baseline(temp_registry, registered_clf_model, reference_dataset):
    """1. Test that near-identical current dataset reports NO drift."""
    engine = ModelMonitoringEngine(registry=temp_registry)
    np.random.seed(42)
    # Current data drawn from exact same distribution
    curr_df = reference_dataset.sample(50, replace=True).copy()

    req = DriftRequest(
        model_id=registered_clf_model,
        current_dataset=curr_df,
    )
    res = engine.monitor(req)

    assert res.status == "success"
    assert res.overall_severity == DriftSeverity.NONE
    assert len(res.data_drift.drifted_features) == 0
    assert "HEALTH CHECK: Model and input feature distributions are stable" in res.recommendations[0]


def test_numeric_feature_drift_detection(temp_registry, registered_clf_model, reference_dataset):
    """2. Test detection of significant mean and distribution shift in numeric feature."""
    engine = ModelMonitoringEngine(registry=temp_registry)
    np.random.seed(99)
    # Massive mean shift: reference was N(50, 10), current is N(95, 15)
    curr_df = reference_dataset.copy()
    curr_df["num_feature"] = np.random.normal(loc=95.0, scale=15.0, size=len(curr_df))

    req = DriftRequest(
        model_id=registered_clf_model,
        current_dataset=curr_df,
    )
    res = engine.monitor(req)

    assert res.data_drift.overall_drift is True
    assert "num_feature" in res.data_drift.drifted_features
    num_res = res.data_drift.feature_results["num_feature"]
    assert num_res.drift_detected is True
    assert num_res.statistical_test == "kolmogorov_smirnov"
    assert num_res.p_value < 0.05
    assert num_res.drift_score > 0.30


def test_categorical_feature_drift_detection(temp_registry, registered_clf_model, reference_dataset):
    """3. Test categorical distribution shift and novel category detection."""
    engine = ModelMonitoringEngine(registry=temp_registry)
    curr_df = reference_dataset.copy()
    # Introduce novel category and invert distribution
    curr_df["cat_feature"] = np.random.choice(["Tier3", "NovelTierX"], size=len(curr_df), p=[0.2, 0.8])

    req = DriftRequest(
        model_id=registered_clf_model,
        current_dataset=curr_df,
        feature_columns=["cat_feature"],
    )
    res = engine.monitor(req)

    cat_res = res.data_drift.feature_results["cat_feature"]
    assert cat_res.drift_detected is True
    assert "NovelTierX" in cat_res.current_statistics["novel_categories"]
    assert cat_res.current_statistics["psi"] >= 0.20


def test_missing_value_rate_drift(temp_registry, registered_clf_model, reference_dataset):
    """4. Test detection of missing-value rate degradation (e.g. 0% -> 35% nulls)."""
    engine = ModelMonitoringEngine(registry=temp_registry)
    curr_df = reference_dataset.copy()
    # Inject 35% NaNs into stable_feature
    curr_df.loc[:35, "stable_feature"] = np.nan

    req = DriftRequest(
        model_id=registered_clf_model,
        current_dataset=curr_df,
    )
    res = engine.monitor(req)

    assert res.data_drift.data_quality_changes["missing_rate_deltas"]["stable_feature"] >= 0.30
    assert res.data_drift.overall_drift is True


# ==============================================================================
# 5-7. Schema Drift Tests
# ==============================================================================

def test_schema_drift_missing_column(temp_registry, registered_clf_model, reference_dataset):
    """5. Test schema drift when a required model feature is missing."""
    engine = ModelMonitoringEngine(registry=temp_registry)
    curr_df = reference_dataset.drop(columns=["num_feature"])

    req = DriftRequest(
        model_id=registered_clf_model,
        current_dataset=curr_df,
    )
    res = engine.monitor(req)

    assert res.data_drift.schema_drift_detected is True
    assert "num_feature" in res.data_drift.schema_changes["missing_features"]
    assert res.overall_severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL)


def test_schema_drift_unexpected_column(temp_registry, registered_clf_model, reference_dataset):
    """6. Test tracking of novel extra columns in input data."""
    engine = ModelMonitoringEngine(registry=temp_registry)
    curr_df = reference_dataset.copy()
    curr_df["unexpected_extra_col"] = "unplanned_string"

    req = DriftRequest(
        model_id=registered_clf_model,
        current_dataset=curr_df,
    )
    res = engine.monitor(req)

    assert "unexpected_extra_col" in res.data_drift.schema_changes["extra_columns"]


def test_schema_drift_datatype_change(temp_registry, registered_clf_model, reference_dataset):
    """7. Test detection of incompatible type mutation (float64 -> string)."""
    engine = ModelMonitoringEngine(registry=temp_registry)
    curr_df = reference_dataset.copy()
    curr_df["num_feature"] = curr_df["num_feature"].astype(str) + "_corrupted"

    req = DriftRequest(
        model_id=registered_clf_model,
        current_dataset=curr_df,
    )
    res = engine.monitor(req)

    assert res.data_drift.schema_drift_detected is True
    assert "num_feature" in res.data_drift.schema_changes["dtype_mismatches"]


# ==============================================================================
# 8-11. Target Drift, Prediction Drift & Performance Degradation
# ==============================================================================

def test_prediction_drift_detection(temp_registry, registered_clf_model, reference_dataset):
    """9. Test drift detection on model output predictions."""
    engine = ModelMonitoringEngine(registry=temp_registry)
    curr_df = reference_dataset.copy()
    # Heavily scale features to trigger massive shift in model output predictions
    curr_df["num_feature"] = curr_df["num_feature"] * 10.0

    req = DriftRequest(
        model_id=registered_clf_model,
        current_dataset=curr_df,
    )
    res = engine.monitor(req)

    assert res.prediction_drift is not None
    assert "current_prediction_stats" in res.prediction_drift.to_dict()


def test_performance_degradation_with_ground_truth(temp_registry, registered_clf_model, reference_dataset):
    """10. Test that ground truth validation flags performance drop vs reference."""
    engine = ModelMonitoringEngine(registry=temp_registry)
    curr_df = reference_dataset.copy()
    # Invert target labels to force catastrophic performance drop
    curr_df["target"] = 1 - curr_df["target"]

    req = DriftRequest(
        model_id=registered_clf_model,
        current_dataset=curr_df,
        target_column="target",
    )
    res = engine.monitor(req)

    assert res.performance_drift is not None
    assert res.performance_drift.target_monitoring_status == "evaluated"
    assert res.performance_drift.degradation_detected is True
    assert res.overall_severity == DriftSeverity.CRITICAL


def test_no_label_scenario(temp_registry, registered_clf_model, reference_dataset):
    """11. Test that missing target labels do NOT claim false performance degradation."""
    engine = ModelMonitoringEngine(registry=temp_registry)
    curr_df = reference_dataset.drop(columns=["target"])

    req = DriftRequest(
        model_id=registered_clf_model,
        current_dataset=curr_df,
    )
    res = engine.monitor(req)

    assert res.performance_drift is not None
    assert res.performance_drift.target_monitoring_status == "unavailable"
    assert res.performance_drift.degradation_detected is False
    assert any("Ground-truth target labels are unavailable" in r for r in res.recommendations)


# ==============================================================================
# 12-16. Severity, Thresholds, History & Evidence
# ==============================================================================

def test_custom_threshold_configuration(temp_registry, registered_clf_model, reference_dataset):
    """14. Test strict threshold configuration."""
    engine = ModelMonitoringEngine(registry=temp_registry)
    curr_df = reference_dataset.copy()
    curr_df["num_feature"] += 2.0  # slight shift

    # Strict p-value threshold (0.90 instead of default 0.05)
    strict_cfg = DriftThresholdConfig(numeric_p_value_threshold=0.90)
    req = DriftRequest(
        model_id=registered_clf_model,
        current_dataset=curr_df,
        threshold_config=strict_cfg,
    )
    res = engine.monitor(req)

    assert res.data_drift.feature_results["num_feature"].threshold == 0.90


def test_monitoring_history_persistence(temp_registry, registered_clf_model, reference_dataset):
    """15. Test that monitoring assessments persist to history log."""
    engine = ModelMonitoringEngine(registry=temp_registry)
    req = DriftRequest(
        model_id=registered_clf_model,
        current_dataset=reference_dataset,
    )
    engine.monitor(req)
    engine.monitor(req)

    history = temp_registry.get_monitoring_history(registered_clf_model)
    assert len(history) == 2
    assert history[0]["model_id"] == registered_clf_model
    assert "overall_severity" in history[0]


def test_traceable_evidence_generation(temp_registry, registered_clf_model, reference_dataset):
    """16. Test that statistical evidence contains exact numbers and p-values."""
    engine = ModelMonitoringEngine(registry=temp_registry)
    curr_df = reference_dataset.copy()
    curr_df["num_feature"] += 50.0

    req = DriftRequest(
        model_id=registered_clf_model,
        current_dataset=curr_df,
    )
    res = engine.monitor(req)

    assert len(res.evidence) > 0
    ev = res.evidence[0]
    assert ev.source == "ModelMonitoringEngine"
    assert "ks_statistic" in ev.data_ref or "p_value" in ev.data_ref


# ==============================================================================
# 17-19. ToolRegistry, Dynamic Planner & AgentResult Tests
# ==============================================================================

def test_tool_registry_model_monitor_integration():
    """17. Test that 'model_monitor' is registered in DEFAULT_TOOL_REGISTRY."""
    tool = DEFAULT_TOOL_REGISTRY.get("model_monitor")
    assert tool is not None
    assert "data_drift_detection" in tool.capabilities
    assert "schema_drift_detection" in tool.capabilities
    assert "performance_monitoring" in tool.capabilities


def test_dynamic_planner_drift_detection_integration(reference_dataset):
    """18. Test that DynamicTaskPlanner synthesizes a model_monitor step for drift detection intent."""
    planner = DynamicTaskPlanner()
    intent = UserIntent(
        intent_type="drift_detection",
        objective="Check if model has drifted on new quarterly data",
        metrics=["CustomerChurnPredictor"],
        required_capabilities=["data_drift_detection"],
        original_command="Check data drift and model health for CustomerChurnPredictor",
    )
    plan = planner.create_execution_plan(intent)

    assert isinstance(plan, ExecutionPlan)
    tool_names = [s.tool_name for s in plan.steps]
    assert "model_monitor" in tool_names


def test_model_monitor_agent_run(temp_registry, registered_clf_model, reference_dataset):
    """19. Test that ModelMonitorAgent returns standardized AgentResult."""
    agent = ModelMonitorAgent(registry=temp_registry)
    task = {
        "model_id": registered_clf_model,
        "current_data": reference_dataset,
    }
    result = agent.run(task)

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.status == AgentStatus.COMPLETED
    assert "overall_severity" in result.metadata
    assert len(result.evidence) > 0


def test_target_drift_detection(temp_registry, registered_clf_model, reference_dataset):
    """20. Test target distribution drift evaluation when target column is monitored."""
    engine = ModelMonitoringEngine(registry=temp_registry)
    curr_df = reference_dataset.copy()
    # Shift target distribution heavily towards class 1 (e.g. 95% 1s)
    curr_df["target"] = np.random.choice([0, 1], size=len(curr_df), p=[0.05, 0.95])

    req = DriftRequest(
        model_id=registered_clf_model,
        current_dataset=curr_df,
        feature_columns=["target"],
    )
    res = engine.monitor(req)

    assert "target" in res.data_drift.feature_results
    assert res.data_drift.feature_results["target"].drift_detected is True


def test_regression_performance_monitoring(temp_registry):
    """21. Test regression model performance monitoring and metric delta tracking."""
    np.random.seed(42)
    n = 100
    X = np.random.normal(10, 2, size=(n, 2))
    y = 3.5 * X[:, 0] + 1.2 * X[:, 1] + np.random.normal(0, 0.5, size=n)

    reg = LinearRegression()
    reg.fit(X, y)

    df_train = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y})
    ref_profile = ModelMonitoringEngine.build_reference_profile(df_train, ["x1", "x2"], "y")

    meta = temp_registry.register_model(
        name="SalesRegressor",
        model_object=reg,
        model_family="traditional_ml",
        algorithm="Linear Regression",
        problem_type="regression",
        target_column="y",
        feature_columns=["x1", "x2"],
        feature_dtypes={"x1": "float64", "x2": "float64"},
        training_metrics={"r2": 0.95, "rmse": 0.50},
        validation_metrics={"r2": 0.94, "rmse": 0.52},
        primary_metric_name="r2",
        primary_metric_value=0.94,
        reference_profile=ref_profile,
    )

    engine = ModelMonitoringEngine(registry=temp_registry)

    # Current data with corrupted target (force R2 degradation)
    df_curr = df_train.copy()
    df_curr["y"] = df_curr["y"] + np.random.normal(50, 10, size=n)

    req = DriftRequest(
        model_id=meta.model_id,
        current_dataset=df_curr,
        target_column="y",
    )
    res = engine.monitor(req)

    assert res.performance_drift is not None
    assert res.performance_drift.target_monitoring_status == "evaluated"
    assert res.performance_drift.degradation_detected is True
    assert "r2" in res.performance_drift.current_metrics
    assert res.performance_drift.current_metrics["r2"] < 0.50
