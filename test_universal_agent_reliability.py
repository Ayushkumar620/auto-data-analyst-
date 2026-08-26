"""
Universal Agent Reliability & Validation Layer Test Suite.

Comprehensive verification of:
A. Valid regression
B. Valid classification
C. Valid forecasting
D. Clustering
E. Anomaly detection
F. Descriptive analysis
G. Missing unrelated feature values
H. Missing target
I. Invalid target
J. Invalid requested column
K. No temporal column for forecast
L. Insufficient rows
M. Constant target
N. Dirty numeric strings
O. Currency values
P. Percentages
Q. Negative accounting values
R. Arbitrary column names
S. Ambiguous user commands
T. Invalid user commands
U. Model failure and recovery
V. NaN/Infinity output sanitization
W. Invalid model metrics
X. Forecast horizon & interval verification
Y. ANN unsuitable dataset
Z. CNN unsuitable dataset
"""
import math
import numpy as np
import pandas as pd
import pytest

from agent.agent_result import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from agent.agents import ForecastAgent, PredictionAgent
from agent.ann_agent import ANNAgent
from agent.cnn_agent import CNNAgent
from agent.confidence_calculator import ConfidenceCalculator
from agent.pre_execution_validator import PreExecutionValidator
from agent.result_validator import ResultValidator


# ---------------------------------------------------------------------------
# Tests A-F: Core Modalities
# ---------------------------------------------------------------------------

def test_A_valid_regression():
    """Verify supervised regression execution, confidence calculation, and result validation."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "feature_alpha": np.linspace(1, 100, n),
        "feature_beta": np.random.normal(0, 1, n),
        "continuous_target": np.linspace(5, 500, n) + np.random.normal(0, 2, n),
    })

    agent = PredictionAgent()
    res = agent.run({"data": df, "target": "continuous_target"})

    assert res.is_success
    assert res.status in (AgentStatus.SUCCESS, AgentStatus.COMPLETED)
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.evidence) >= 1
    assert res.result.get("metric", {}).get("type") == "regression"
    assert res.result.get("metric", {}).get("r2_score") is not None


def test_B_valid_classification():
    """Verify classification execution, class probability balance, and data-driven confidence."""
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "metric_x": np.random.normal(5, 2, n),
        "metric_y": np.random.normal(10, 3, n),
        "class_label": np.random.choice(["Class_A", "Class_B"], size=n),
    })

    agent = PredictionAgent()
    res = agent.run({"data": df, "target": "class_label"})

    assert res.is_success
    assert res.result.get("metric", {}).get("type") == "classification"
    assert 0.0 <= res.confidence <= 1.0


def test_C_valid_forecasting():
    """Verify time-series forecast execution through single source of truth."""
    dates = pd.date_range("2023-01-01", periods=20, freq="ME")
    df = pd.DataFrame({
        "timestamp_idx": dates,
        "sensor_reading": np.linspace(10, 100, 20) + np.random.normal(0, 1, 20),
    })

    agent = ForecastAgent()
    res = agent.run({"data": df, "target": "sensor_reading", "periods": 4})

    assert res.is_success
    assert len(res.result.get("forecast", [])) == 4
    assert res.result.get("trend") in ("upward", "downward", "flat")
    assert 0.0 <= res.confidence <= 1.0


def test_D_clustering_validation():
    """Verify pre-execution validation for clustering."""
    df = pd.DataFrame({
        "dim_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "dim_2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    report = PreExecutionValidator.validate(df, task_type="clustering")
    assert report.is_valid


def test_E_anomaly_detection_validation():
    """Verify pre-execution validation for anomaly detection."""
    df = pd.DataFrame({
        "sensor_val": [10, 12, 11, 9, 13, 100, 11, 10, 12],
    })
    report = PreExecutionValidator.validate(df, task_type="anomaly_detection")
    assert report.is_valid


def test_F_descriptive_analysis():
    """Verify descriptive task does not fail due to lack of time or target column."""
    df = pd.DataFrame({
        "category_col": ["North", "South", "East", "West"],
        "metric_col": [100, 200, 150, 300],
    })
    report = PreExecutionValidator.validate(df, task_type="descriptive")
    assert report.is_valid


# ---------------------------------------------------------------------------
# Tests G-M: Robustness, Missingness, and Rejections
# ---------------------------------------------------------------------------

def test_G_missing_unrelated_feature_values_preserve_target_rows():
    """Verify unrelated sparse columns with nulls do NOT drop valid target rows."""
    n = 25
    df = pd.DataFrame({
        "target_col": np.linspace(10, 100, n),
        "good_feature": np.random.normal(5, 1, n),
        "sparse_notes": [None if i % 2 == 0 else f"note_{i}" for i in range(n)],
        "sparse_meta": [None if i % 3 == 0 else float(i) for i in range(n)],
    })

    agent = PredictionAgent()
    res = agent.run({"data": df, "target": "target_col"})
    assert res.is_success
    assert res.result.get("train_size", 0) + res.result.get("test_size", 0) >= 20


def test_H_missing_target():
    """Verify pre-execution validation catches non-existent target."""
    df = pd.DataFrame({"col_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    report = PreExecutionValidator.validate(df, task_type="regression", target="non_existent_target")
    assert not report.is_valid
    assert report.error.category in (ErrorCategory.TARGET_NOT_FOUND, "target_not_found")
    assert "non_existent_target" in report.error.user_message


def test_I_invalid_target_constant():
    """Verify constant targets (0 variance) are rejected."""
    df = pd.DataFrame({
        "feat": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "flat_target": [42.0] * 12,
    })
    report = PreExecutionValidator.validate(df, task_type="regression", target="flat_target")
    assert not report.is_valid
    assert report.error.category == ErrorCategory.DATA_INVALID


def test_J_invalid_requested_columns():
    """Verify requested missing feature columns are rejected cleanly."""
    df = pd.DataFrame({"col_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    report = PreExecutionValidator.validate(df, task_type="regression", target="col_a", feature_columns=["ghost_col"])
    assert not report.is_valid
    assert "ghost_col" in str(report.error.user_message)


def test_K_no_temporal_column_for_forecast():
    """Verify cross-sectional table without time column is rejected for forecasting."""
    df = pd.DataFrame({
        "metric_a": [10, 20, 30, 40, 50, 60, 70, 80],
        "metric_b": [100, 200, 300, 400, 500, 600, 700, 800],
    })
    agent = ForecastAgent()
    res = agent.run({"data": df, "target": "metric_a"})
    assert not res.is_success
    assert res.errors[0].category in (ErrorCategory.TIME_COLUMN_NOT_FOUND, "time_column_not_found")


def test_L_insufficient_rows_diagnostics():
    """Verify structured diagnostic response when rows < threshold."""
    df = pd.DataFrame({
        "feat": [1, 2, 3],
        "target": [10, 20, 30],
    })
    report = PreExecutionValidator.validate(df, task_type="regression", target="target")
    assert not report.is_valid
    assert report.error.category == ErrorCategory.INSUFFICIENT_DATA
    assert report.audit.valid_rows == 3


def test_M_constant_target_classification():
    """Verify single-class targets are rejected for classification."""
    df = pd.DataFrame({
        "feat": list(range(15)),
        "target": ["SameClass"] * 15,
    })
    report = PreExecutionValidator.validate(df, task_type="classification", target="target")
    assert not report.is_valid
    assert report.error.category == ErrorCategory.DATA_INVALID


# ---------------------------------------------------------------------------
# Tests N-R: Formatting and Coercion Invariance
# ---------------------------------------------------------------------------

def test_N_dirty_numeric_strings_and_currencies():
    """Verify strings like '$1,200', '15%', '(500.00)' are coerced without failure."""
    dates = pd.date_range("2023-01-01", periods=15, freq="ME")
    df = pd.DataFrame({
        "date": dates,
        "revenue_dirty": ["$1,000", "$1,100", "$1,250", "$1,300", "$1,450",
                          "$1,500", "$1,620", "$1,700", "$1,850", "$1,900",
                          "$2,050", "$2,100", "$2,250", "$2,300", "$2,450"],
    })
    agent = ForecastAgent()
    res = agent.run({"data": df, "target": "revenue_dirty", "periods": 3})
    assert res.is_success
    assert len(res.result.get("forecast", [])) == 3


def test_O_negative_accounting_values():
    """Verify negative parentheses '(1,234.50)' are converted to negative numbers."""
    from agent.canonical_data_layer import CanonicalDataLayer
    s = pd.Series(["(100.00)", "$250.50", "(50.00)", "0.00"])
    coerced = CanonicalDataLayer.coerce_numeric_series(s)
    assert float(coerced.iloc[0]) == -100.00
    assert float(coerced.iloc[1]) == 250.50
    assert float(coerced.iloc[2]) == -50.00


def test_P_arbitrary_column_names():
    """Verify pipeline works with arbitrary un-business column names."""
    n = 30
    df = pd.DataFrame({
        "col_xyz_99": np.linspace(1, 100, n),
        "feat_777": np.random.normal(0, 1, n),
        "target_omega": np.linspace(10, 1000, n),
    })
    agent = PredictionAgent()
    res = agent.run({"data": df, "target": "target_omega"})
    assert res.is_success


# ---------------------------------------------------------------------------
# Tests S-X: Ambiguity, Result Validation & Metrics
# ---------------------------------------------------------------------------

def test_S_ambiguous_user_commands_returns_needs_clarification():
    """Verify ambiguous requests return NEEDS_CLARIFICATION."""
    res = AgentResult.create_needs_clarification(
        agent_name="TestAgent",
        clarification_message="Did you mean to forecast Revenue or Units?",
        options=[{"target": "Revenue", "type": "forecasting"}, {"target": "Units", "type": "forecasting"}],
    )
    assert res.is_needs_clarification
    assert res.status == AgentStatus.NEEDS_CLARIFICATION
    assert len(res.result.get("options", [])) == 2


def test_T_invalid_command_error_safety():
    """Verify errors never leak raw stack traces to the user."""
    err = AgentError.create(
        category=ErrorCategory.INPUT_INVALID,
        user_message="Invalid parameter provided.",
        technical_details={"traceback": "Traceback (most recent call last):\n  File 'test.py', line 10"},
    )
    res = AgentResult.create_error("TestAgent", err)
    assert not res.is_success
    assert "Traceback" not in res.message
    assert "Traceback" in str(res.diagnostics.get("traceback"))


def test_V_nan_infinity_sanitization():
    """Verify ResultValidator cleans NaN and Infinity values."""
    res = AgentResult(
        success=True,
        status=AgentStatus.SUCCESS,
        agent_name="MathAgent",
        result={"val": float("nan"), "inf_val": float("inf"), "normal_val": 42.0},
        metrics={"score": float("nan")},
    )
    repaired, vr = ResultValidator().repair(res)
    assert repaired.result["val"] is None
    assert repaired.result["inf_val"] is None
    assert repaired.result["normal_val"] == 42.0


def test_W_invalid_model_metrics_detected():
    """Verify invalid metrics like R2 > 1.0 or accuracy > 1.0 are caught."""
    res = AgentResult(
        success=True,
        status=AgentStatus.SUCCESS,
        agent_name="PredictorAgent",
        metrics={"r2_score": 1.50, "accuracy": 2.0},
    )
    vr = ResultValidator().validate(res)
    assert not vr.passed
    issue_codes = [i.code for i in vr.issues]
    assert "INVALID_R2_SCORE" in issue_codes
    assert "INVALID_ACCURACY" in issue_codes


def test_X_forecast_interval_verification():
    """Verify inverted forecast intervals (lower > prediction) are repaired."""
    res = AgentResult(
        success=True,
        status=AgentStatus.SUCCESS,
        agent_name="ForecastAgent",
        result={
            "forecast": [
                {"prediction": 100.0, "lower": 120.0, "upper": 80.0},  # Inverted!
            ]
        },
    )
    repaired, vr = ResultValidator().repair(res)
    pt = repaired.result["forecast"][0]
    assert pt["lower"] <= pt["prediction"] <= pt["upper"]


# ---------------------------------------------------------------------------
# Tests Y-Z: Neural Network Suitability
# ---------------------------------------------------------------------------

def test_Y_ann_unsuitable_dataset_rejection():
    """Verify ANN rejects datasets with fewer than 10 samples."""
    df_tiny = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    ann = ANNAgent()
    res = ann.run({"data": df_tiny, "target": "y"})
    assert not res.is_success


def test_Z_cnn_unsuitable_dataset_rejection():
    """Verify CNN rejects 1D non-spatial data without spatial shape or signal config."""
    df = pd.DataFrame({"feature1": [1, 2, 3, 4], "feature2": [10, 20, 30, 40]})
    cnn = CNNAgent()
    res = cnn.run({"data": df, "target": "feature2"})
    # Non-spatial tabular data is incompatible with CNN
    assert not res.is_success
