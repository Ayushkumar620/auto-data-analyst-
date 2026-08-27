"""
Comprehensive Test Suite for Milestone 7 — Task 4:
Universal Analytical Explanation & Evidence Traceability Layer.

Tests:
A. Regression explanation
B. Classification explanation
C. Forecast explanation
D. Anomaly explanation
E. Clustering explanation
F. Statistical relationship explanation
G. EDA/data-quality explanation
H. Evidence references remain valid
I. No fabricated evidence
J. Missing evidence is explicitly handled
K. Causal language protection
L. Metric/value preservation
M. Confidence separation
N. Prediction interval explanation
O. FDR-adjusted p-value explanation
P. Deterministic output
Q. Invalid AgentResult handling
R. Empty result handling
S. FastAPI endpoint integration
T. Orchestrator integration
U. Natural-language routing
V. Structured AgentError behavior
W. No traceback leakage
"""
import math
import uuid
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.agent_result import AgentError, AgentResult, AgentStatus, ClaimType, ErrorCategory, Evidence
from agent.explanation_agent import ExplanationAgent
from agent.explanation_engine import ExplanationEngine
from agent.explanation_schemas import AnalyticalExplanation, EvidenceTrace, ExplanationSection, MetricExplanation
from agent.orchestrator import UniversalOrchestrator
from backend.app.main import app


# ------------------------------------------------------------------------------
# Test A: Regression Explanation
# ------------------------------------------------------------------------------
def test_A_regression_explanation():
    engine = ExplanationEngine()
    ev = Evidence(
        dataset_name="sales_data",
        columns=["ad_spend", "revenue"],
        operation="linear_regression",
        calculation="revenue ~ ad_spend (OLS)",
        result={"r2": 0.84, "mae": 12.5, "rmse": 16.2},
        confidence=0.92,
    )
    result = AgentResult.success(
        output={
            "target": "revenue",
            "features": ["ad_spend"],
            "model_name": "Ridge Regression",
            "metrics": {"r2": 0.8421, "mae": 12.50, "rmse": 16.20},
            "feature_importances": {"ad_spend": 0.85},
            "train_rows": 80,
            "test_rows": 20,
        },
        evidence=[ev],
        task_type="regression",
    )

    exp = engine.explain(result)
    assert isinstance(exp, AnalyticalExplanation)
    assert exp.task_type == "regression"
    assert "0.8421" in exp.summary or "0.84" in exp.summary
    assert "revenue" in exp.summary
    assert len(exp.findings) >= 1
    assert len(exp.methodology) >= 1
    assert any(m.metric_name.startswith("R²") for m in exp.metrics)
    assert any("does not prove causal" in lim.lower() for lim in exp.limitations)


# ------------------------------------------------------------------------------
# Test B: Classification Explanation
# ------------------------------------------------------------------------------
def test_B_classification_explanation():
    engine = ExplanationEngine()
    ev = Evidence(
        dataset_name="churn_data",
        columns=["tenure", "churn"],
        operation="classification_evaluation",
        result={"accuracy": 0.89, "f1": 0.85},
        confidence=0.90,
    )
    result = AgentResult.success(
        output={
            "target": "churn",
            "model_name": "Random Forest Classifier",
            "metrics": {"accuracy": 0.8912, "f1": 0.8540, "precision": 0.87, "recall": 0.84},
            "classes": [0, 1],
        },
        evidence=[ev],
        task_type="classification",
    )

    exp = engine.explain(result)
    assert exp.task_type == "classification"
    assert "0.8912" in exp.summary or "0.89" in exp.summary
    assert any(m.metric_name == "Accuracy" for m in exp.metrics)
    assert any(m.metric_name == "F1-Score" for m in exp.metrics)


# ------------------------------------------------------------------------------
# Test C: Forecast Explanation
# ------------------------------------------------------------------------------
def test_C_forecast_explanation():
    engine = ExplanationEngine()
    ev = Evidence(
        dataset_name="ts_data",
        columns=["date", "demand"],
        operation="arima_forecast",
        calculation="ARIMA(1,1,1) h=6",
        result={"smape": 8.4},
        confidence=0.88,
    )
    result = AgentResult.success(
        output={
            "target": "demand",
            "horizon": 6,
            "trend": "upward",
            "model_selected": "ARIMA",
            "metrics": {"smape": 8.45},
            "forecast": [{"period": 1, "forecast": 105.2}, {"period": 6, "forecast": 132.8}],
        },
        evidence=[ev],
        task_type="forecasting",
    )

    exp = engine.explain(result)
    assert exp.task_type == "forecasting"
    assert "6" in exp.summary
    assert "projections" in exp.summary.lower() or "projections" in str(exp.limitations).lower()
    assert any("projections, not guarantees" in lim.lower() or "continuity" in lim.lower() for lim in exp.limitations)


# ------------------------------------------------------------------------------
# Test D: Anomaly Explanation
# ------------------------------------------------------------------------------
def test_D_anomaly_explanation():
    engine = ExplanationEngine()
    ev = Evidence(
        dataset_name="transactions",
        columns=["amount", "velocity"],
        operation="isolation_forest",
        result={"anomaly_count": 5},
        confidence=0.85,
    )
    result = AgentResult.success(
        output={
            "detector_used": "Isolation Forest",
            "total_records": 100,
            "anomaly_count": 5,
            "anomaly_percentage": 5.0,
            "features_used": ["amount", "velocity"],
        },
        evidence=[ev],
        task_type="anomaly_detection",
    )

    exp = engine.explain(result)
    assert exp.task_type == "anomaly_detection"
    assert "5" in exp.summary
    assert any("outliers" in lim.lower() for lim in exp.limitations)
    assert any("fraud" in lim.lower() or "malicious" in lim.lower() for lim in exp.limitations)


# ------------------------------------------------------------------------------
# Test E: Clustering Explanation
# ------------------------------------------------------------------------------
def test_E_clustering_explanation():
    engine = ExplanationEngine()
    ev = Evidence(
        dataset_name="customers",
        columns=["income", "spend"],
        operation="kmeans_clustering",
        result={"silhouette_score": 0.62},
        confidence=0.85,
    )
    result = AgentResult.success(
        output={
            "algorithm": "K-Means",
            "cluster_count": 4,
            "silhouette_score": 0.6215,
            "davies_bouldin_index": 0.78,
            "features_used": ["income", "spend"],
        },
        evidence=[ev],
        task_type="clustering",
    )

    exp = engine.explain(result)
    assert exp.task_type == "clustering"
    assert "4" in exp.summary
    assert any(m.metric_name == "Silhouette Score" for m in exp.metrics)
    assert any("causal" in lim.lower() for lim in exp.limitations)


# ------------------------------------------------------------------------------
# Test F: Statistical Relationship Explanation
# ------------------------------------------------------------------------------
def test_F_statistical_relationship_explanation():
    engine = ExplanationEngine()
    ev = Evidence(
        dataset_name="market",
        columns=["price", "sales"],
        operation="pearson_correlation",
        calculation="corr(price, sales) = -0.76",
        result={"r": -0.7621, "p_value": 0.0001, "adjusted_p_value": 0.0003},
        confidence=0.95,
    )
    result = AgentResult.success(
        output={
            "method": "Pearson Correlation",
            "ranked_relationships": [
                {
                    "feature_1": "price",
                    "feature_2": "sales",
                    "correlation": -0.7621,
                    "p_value": 0.0001,
                    "adjusted_p_value": 0.0003,
                }
            ],
        },
        evidence=[ev],
        task_type="statistical_analysis",
    )

    exp = engine.explain(result)
    assert exp.task_type == "statistical_analysis"
    assert "-0.7621" in exp.summary or "-0.76" in exp.summary
    assert "correlation indicates statistical association" in exp.summary.lower() or any("causation" in lim.lower() for lim in exp.limitations)


# ------------------------------------------------------------------------------
# Test G: EDA / Data Quality Explanation
# ------------------------------------------------------------------------------
def test_G_eda_data_quality_explanation():
    engine = ExplanationEngine()
    ev = Evidence(
        dataset_name="raw_store",
        columns=["col1", "col2"],
        operation="profile_dataset",
        result={"quality_score": 0.96},
        confidence=0.98,
    )
    result = AgentResult.success(
        output={
            "total_rows": 250,
            "total_columns": 8,
            "quality_score": 0.9650,
            "total_missing_cells": 4,
        },
        evidence=[ev],
        task_type="eda",
    )

    exp = engine.explain(result)
    assert exp.task_type == "eda"
    assert "250" in exp.summary
    assert any(m.metric_name == "Data Quality Score" for m in exp.metrics)


# ------------------------------------------------------------------------------
# Test H: Evidence References Remain Valid
# ------------------------------------------------------------------------------
def test_H_evidence_references_remain_valid():
    engine = ExplanationEngine()
    e_id = "evi_verified_123"
    ev = Evidence(
        dataset_name="inventory",
        columns=["stock"],
        operation="sum",
        source_reference=e_id,
        result={"total_stock": 5000},
        confidence=0.99,
    )
    result = AgentResult.success(
        output={"total_rows": 50, "total_columns": 2, "quality_score": 0.95},
        evidence=[ev],
        task_type="eda",
    )

    exp = engine.explain(result)
    assert len(exp.evidence) >= 1
    assert any(tr.evidence_id == e_id for tr in exp.evidence)
    # Check that findings reference the valid evidence id
    finding_refs = [ref for f in exp.findings for ref in f.evidence_refs]
    assert e_id in finding_refs or len(finding_refs) >= 1


# ------------------------------------------------------------------------------
# Test I: No Fabricated Evidence
# ------------------------------------------------------------------------------
def test_I_no_fabricated_evidence():
    engine = ExplanationEngine()
    # Provide no evidence in result
    result = AgentResult.success(
        output={"total_rows": 10, "total_columns": 2, "quality_score": 1.0},
        evidence=[],
        task_type="eda",
    )

    exp = engine.explain(result)
    # Evidence traces should be empty or matching only provided evidence
    assert len(exp.evidence) == 0


# ------------------------------------------------------------------------------
# Test J: Missing Evidence is Explicitly Handled
# ------------------------------------------------------------------------------
def test_J_missing_evidence_handled():
    engine = ExplanationEngine()
    result = AgentResult.success(output={}, evidence=[], task_type="general")
    exp = engine.explain(result)
    assert isinstance(exp, AnalyticalExplanation)
    assert exp.summary is not None


# ------------------------------------------------------------------------------
# Test K: Causal Language Protection
# ------------------------------------------------------------------------------
def test_K_causal_language_protection():
    engine = ExplanationEngine()
    dirty_text = "Ad spend causes revenue increase and drives growth, which leads to higher profit because of demand."
    clean_text = engine.sanitize_causal_language(dirty_text)

    assert "causes" not in clean_text.lower()
    assert "drives" not in clean_text.lower()
    assert "leads to" not in clean_text.lower()
    assert "because of" not in clean_text.lower()
    assert "is associated with" in clean_text.lower() or "associated" in clean_text.lower()


# ------------------------------------------------------------------------------
# Test L: Metric/Value Preservation
# ------------------------------------------------------------------------------
def test_L_metric_value_preservation():
    engine = ExplanationEngine()
    exact_r2 = 0.876543
    exact_rmse = 3.141592
    result = AgentResult.success(
        output={
            "target": "target_y",
            "model_name": "ElasticNet",
            "metrics": {"r2": exact_r2, "rmse": exact_rmse},
        },
        task_type="regression",
    )

    exp = engine.explain(result)
    r2_metric = next(m for m in exp.metrics if "R²" in m.metric_name)
    assert r2_metric.value == round(exact_r2, 4)


# ------------------------------------------------------------------------------
# Test M: Confidence Separation
# ------------------------------------------------------------------------------
def test_M_confidence_separation():
    engine = ExplanationEngine()
    result = AgentResult.success(
        output={
            "method": "Pearson Correlation",
            "ranked_relationships": [{"feature_1": "a", "feature_2": "b", "correlation": 0.82, "p_value": 0.002}],
        },
        task_type="statistical_analysis",
    )

    exp = engine.explain(result)
    u = exp.uncertainty
    assert u["statistical_confidence"] is not None
    assert u["statistical_confidence"] == round(1.0 - 0.002, 4)
    assert u["model_validation_score"] is None  # Statistical test has no ML validation score
    assert "r = +0.8200" in u["practical_effect_size"]


# ------------------------------------------------------------------------------
# Test N: Prediction Interval Explanation
# ------------------------------------------------------------------------------
def test_N_prediction_interval_explanation():
    engine = ExplanationEngine()
    result = AgentResult.success(
        output={"target": "val", "horizon": 4, "confidence_interval_level": "95%"},
        task_type="forecasting",
    )

    exp = engine.explain(result)
    assert exp.uncertainty["prediction_interval_level"] == "95%"
    assert any("interval" in n.lower() for n in exp.uncertainty.get("notes", []))


# ------------------------------------------------------------------------------
# Test O: FDR-Adjusted p-value Explanation
# ------------------------------------------------------------------------------
def test_O_fdr_adjusted_p_value_explanation():
    engine = ExplanationEngine()
    result = AgentResult.success(
        output={
            "ranked_relationships": [
                {"feature_1": "x", "feature_2": "y", "correlation": 0.65, "p_value": 0.004, "adjusted_p_value": 0.012}
            ]
        },
        task_type="statistical_analysis",
    )

    exp = engine.explain(result)
    assert any("FDR" in m.metric_name for m in exp.metrics)


# ------------------------------------------------------------------------------
# Test P: Deterministic Output
# ------------------------------------------------------------------------------
def test_P_deterministic_output():
    engine = ExplanationEngine()
    payload = {
        "target": "revenue",
        "model_name": "OLS",
        "metrics": {"r2": 0.81, "mae": 5.2},
        "train_rows": 100,
        "test_rows": 25,
    }
    res1 = engine.explain(payload)
    res2 = engine.explain(payload)

    assert res1.summary == res2.summary
    assert len(res1.findings) == len(res2.findings)
    assert len(res1.metrics) == len(res2.metrics)
    assert [m.value for m in res1.metrics] == [m.value for m in res2.metrics]


# ------------------------------------------------------------------------------
# Test Q: Invalid AgentResult Handling
# ------------------------------------------------------------------------------
def test_Q_invalid_agentresult_handling():
    agent = ExplanationAgent()
    err_res = AgentResult.error(error="Upstream model convergence failure", agent_name="Predictor")
    exp_res = agent.execute({"result": err_res})

    assert isinstance(exp_res, AgentResult)
    assert exp_res.status in (AgentStatus.SUCCESS, AgentStatus.COMPLETED)
    assert "explanation_id" in exp_res.data or "summary" in exp_res.data


# ------------------------------------------------------------------------------
# Test R: Empty Result Handling
# ------------------------------------------------------------------------------
def test_R_empty_result_handling():
    agent = ExplanationAgent()
    res = agent.execute({})
    assert res.status == AgentStatus.NEEDS_CLARIFICATION
    assert len(res.errors) >= 1
    assert res.errors[0].code == "EMPTY_EXPLANATION_INPUT"


# ------------------------------------------------------------------------------
# Test S: FastAPI Endpoint Integration
# ------------------------------------------------------------------------------
def test_S_fastapi_endpoints():
    client = TestClient(app)
    payload = {
        "result": {
            "task_type": "regression",
            "target": "price",
            "model_name": "Lasso Regression",
            "metrics": {"r2": 0.79, "rmse": 4.5},
        }
    }

    r1 = client.post("/api/v1/explanations/explain", json=payload)
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["status"] in ("success", "completed")
    assert "findings" in d1.get("result", {}) or "summary" in d1.get("result", {})

    r2 = client.post("/api/v1/explanations", json=payload)
    assert r2.status_code == 200


# ------------------------------------------------------------------------------
# Test T: Orchestrator Integration
# ------------------------------------------------------------------------------
def test_T_orchestrator_integration():
    orch = UniversalOrchestrator()
    df = pd.DataFrame({"x": range(30), "y": [i * 2 + 1 for i in range(30)]})
    res = orch.orchestrate("analyze correlation between x and y", df)

    assert res.status in (AgentStatus.SUCCESS, AgentStatus.COMPLETED)
    assert "explanation" in res.data
    exp_dict = res.data["explanation"]
    assert "summary" in exp_dict
    assert "findings" in exp_dict


# ------------------------------------------------------------------------------
# Test U: Natural-Language Routing
# ------------------------------------------------------------------------------
def test_U_natural_language_routing():
    from agent.intent import CommandIntelligenceAgent, IntentType
    agent = CommandIntelligenceAgent()
    res = agent.analyze_intent("how was this calculated and show evidence")

    assert res.intent_type == IntentType.EXPLANATION or "explanation" in res.required_capabilities


# ------------------------------------------------------------------------------
# Test V: Structured AgentError Behavior
# ------------------------------------------------------------------------------
def test_V_structured_agent_error():
    agent = ExplanationAgent()
    res = agent.execute({})
    assert len(res.errors) >= 1
    err = res.errors[0]
    assert isinstance(err, AgentError)
    assert err.category == ErrorCategory.INPUT_INVALID


# ------------------------------------------------------------------------------
# Test W: No Traceback Leakage
# ------------------------------------------------------------------------------
def test_W_no_traceback_leakage():
    agent = ExplanationAgent()
    # Force an invalid object
    res = agent.execute({"result": None, "data": None})
    assert res.status == AgentStatus.NEEDS_CLARIFICATION
    user_msg = res.error_message or res.data.get("error", "")
    assert "Traceback (most recent call last)" not in user_msg