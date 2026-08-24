"""Comprehensive test suite for Phase 8: Validation Engine Across All Engines."""
import numpy as np
import pandas as pd
import pytest

from agent.schemas import AgentResult, AgentStatus, ClaimType
from agent.validation_agent import DataValidationAgent
from agent.planner import PlannerAgent
from backend.app.ml.validation_engine import (
    DataModelValidator,
    IssueSeverity,
    ValidationAuditReport,
    ValidationCheckType,
    ValidationIssue,
)


# ==============================================================================
# 1. Data Leakage & Multicollinearity Tests
# ==============================================================================

def test_target_leakage_detection():
    """Verify detection of leaking features (near-perfect correlation and identical columns)."""
    validator = DataModelValidator()
    np.random.seed(42)
    n = 60
    target = np.random.uniform(10, 100, n)

    df = pd.DataFrame({
        "clean_feature": np.random.normal(0, 1, n),
        "leaking_feature_corr": target * 1.0001 + np.random.normal(0, 0.0001, n),  # corr > 0.999
        "identical_target_col": target.copy(),  # Identical column
        "target_col": target,
    })

    issues, diagnostics = validator.check_data_leakage(df, target_column="target_col")
    assert len(issues) >= 2
    assert any(i.severity == IssueSeverity.CRITICAL for i in issues)
    assert "leaking_feature_corr" in diagnostics["leaking_features"]
    assert "identical_target_col" in diagnostics["leaking_features"]


def test_multicollinearity_detection():
    """Verify detection of highly collinear feature pairs."""
    validator = DataModelValidator()
    np.random.seed(42)
    n = 50
    f1 = np.random.uniform(10, 50, n)
    f2 = f1 * 2.0 + np.random.normal(0, 0.01, n)  # correlation > 0.99
    f3 = np.random.uniform(1, 10, n)

    df = pd.DataFrame({"feat_1": f1, "feat_2": f2, "feat_3": f3, "target": f3 * 2})
    issues, diagnostics = validator.check_multicollinearity(df, target_column="target", collinear_threshold=0.95)

    assert len(issues) == 1
    assert issues[0].check_type == ValidationCheckType.STATISTICAL_ASSUMPTIONS
    assert "feat_1" in issues[0].affected_columns
    assert "feat_2" in issues[0].affected_columns


# ==============================================================================
# 2. Class Imbalance Tests
# ==============================================================================

def test_class_imbalance_detection():
    """Verify that severe class imbalance triggers warnings with remediation advice."""
    validator = DataModelValidator()

    # Severe 95/5 imbalance
    y_imbalanced = pd.Series([0] * 95 + [1] * 5, name="is_fraud")
    issues, diag = validator.check_class_imbalance(y_imbalanced, imbalance_threshold=0.15)

    assert len(issues) == 1
    assert issues[0].check_type == ValidationCheckType.CLASS_IMBALANCE
    assert issues[0].severity == IssueSeverity.CRITICAL
    assert "ROC-AUC" in issues[0].description
    assert diag["minority_class_ratio"] == 0.05

    # Balanced 50/50
    y_balanced = pd.Series([0] * 50 + [1] * 50, name="binary_label")
    issues_bal, diag_bal = validator.check_class_imbalance(y_balanced, imbalance_threshold=0.15)
    assert len(issues_bal) == 0
    assert diag_bal["is_imbalanced"] is False


# ==============================================================================
# 3. Overfitting & Underfitting Tests
# ==============================================================================

def test_overfitting_underfitting_detection():
    """Verify detection of train/test score divergence and underfitting."""
    validator = DataModelValidator()

    # Overfitting scenario
    of_issues, of_diag = validator.check_overfitting_underfitting(
        train_score=0.98, test_score=0.68, primary_metric_name="R2"
    )
    assert len(of_issues) == 1
    assert of_issues[0].check_type == ValidationCheckType.OVERFIT_UNDERFIT
    assert of_diag["is_overfitting"] is True

    # Underfitting scenario
    uf_issues, uf_diag = validator.check_overfitting_underfitting(
        train_score=0.32, test_score=0.28, primary_metric_name="accuracy"
    )
    assert len(uf_issues) == 1
    assert uf_diag["is_underfitting"] is True


# ==============================================================================
# 4. Temporal Leakage Tests
# ==============================================================================

def test_temporal_leakage_detection():
    """Verify detection of lookahead bias in time series splits."""
    validator = DataModelValidator()

    # Lookahead leakage: train split contains 2024-03-01 which is after test min 2024-02-01
    train_dates = pd.date_range("2024-01-01", "2024-03-01", freq="D")
    test_dates = pd.date_range("2024-02-01", "2024-02-28", freq="D")

    issues, diag = validator.check_temporal_leakage(train_dates, test_dates)
    assert len(issues) == 1
    assert issues[0].check_type == ValidationCheckType.TEMPORAL_LEAKAGE
    assert issues[0].severity == IssueSeverity.CRITICAL
    assert diag["has_temporal_leakage"] is True

    # Proper chronological split
    valid_train = pd.date_range("2024-01-01", "2024-01-31", freq="D")
    valid_test = pd.date_range("2024-02-01", "2024-02-28", freq="D")
    issues_clean, diag_clean = validator.check_temporal_leakage(valid_train, valid_test)
    assert len(issues_clean) == 0
    assert diag_clean["has_temporal_leakage"] is False


# ==============================================================================
# 5. Full Pipeline Audit & Agent Integration Tests
# ==============================================================================

def test_audit_pipeline_full_synthesis():
    """Verify comprehensive pipeline audit report synthesis."""
    validator = DataModelValidator()
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feat_b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],  # Collinear with feat_a
        "target": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # Imbalanced
    })

    report = validator.audit_pipeline(
        df=df,
        target_column="target",
        train_score=0.99,
        test_score=0.72,
    )

    assert isinstance(report, ValidationAuditReport)
    assert report.overall_status in ("FAILED", "PASSED_WITH_WARNINGS")
    assert report.warnings_count >= 1
    assert "leakage" in report.diagnostics
    assert "imbalance" in report.diagnostics
    assert "overfit_underfit" in report.diagnostics


def test_data_validation_agent_auto_repair():
    """Verify DataValidationAgent runs and automatically removes leaking features."""
    agent = DataValidationAgent()
    np.random.seed(42)
    target = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    df = pd.DataFrame({
        "normal_feat": [5, 1, 9, 2, 8, 3, 7, 4, 6, 2],  # Non-collinear with target
        "leaking_proxy": target * 1.0001,  # Critical leakage
        "target": target,
    })

    result = agent.run({
        "data": df,
        "target": "target",
        "train_score": 0.90,
        "test_score": 0.88,
        "auto_repair": True,
    })

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.agent == "Data Validation Agent"
    assert "repaired_data" in result.output
    assert "leaking_proxy" not in result.output["repaired_data"].columns
    assert "normal_feat" in result.output["repaired_data"].columns
    assert len(result.evidence) >= 1


def test_planner_agent_validate_action():
    """Verify PlannerAgent routing for 'validate' action."""
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    })
    planner = PlannerAgent(data=df)
    result = planner.run_agent({"action": "validate", "target": "y", "train_score": 0.95, "test_score": 0.92})

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.validation is not None
    assert result.validation.passed is True
    assert "overall_status" in result.output
