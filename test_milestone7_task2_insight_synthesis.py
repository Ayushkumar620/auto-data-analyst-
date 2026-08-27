"""
Milestone 7 — Task 2: Comprehensive Universal Insight Synthesis & Decision Layer Test Suite.

Exhaustively covers tests A through AB:
A. single-agent insight synthesis
B. multi-agent synthesis
C. arbitrary column names
D. evidence preservation
E. provenance preservation
F. confidence bounds
G. duplicate insight merging
H. contradiction detection
I. correlation/causation protection
J. statistically significant but practically weak relationship
K. strong practical effect
L. forecast interpretation
M. anomaly interpretation
N. clustering interpretation
O. prediction interpretation
P. data-quality insight generation
Q. missing analytical result
R. failed analytical task
S. partial orchestration result
T. empty result set
U. malformed result
V. fabricated metric prevention
W. fabricated column prevention
X. deterministic ordering
Y. FastAPI integration
Z. orchestrator integration
AA. existing API compatibility
AB. complete end-to-end natural-language workflow
"""
from __future__ import annotations

import math
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.agent_result import AgentError, AgentResult, AgentStatus, ClaimType, ErrorCategory, Evidence
from agent.insight_synthesis_engine import (
    Contradiction,
    InsightCategory,
    InsightSynthesisEngine,
    SynthesisReport,
    SynthesizedInsight,
)
from agent.insight_synthesis_agent import InsightSynthesisAgent
from agent.orchestrator import UniversalOrchestrator
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


# A. Single-agent insight synthesis
def test_A_single_agent_insight_synthesis():
    df = pd.DataFrame({"feat_a": range(50), "feat_b": [i * 2 for i in range(50)]})
    orch = UniversalOrchestrator()
    res = orch.orchestrate("profile this data", df)
    engine = InsightSynthesisEngine()
    report = engine.synthesize(res, df)
    assert len(report.key_insights) > 0
    assert any(i.category == InsightCategory.DATA_QUALITY.value for i in report.data_quality_findings)


# B. Multi-agent synthesis
def test_B_multi_agent_synthesis():
    df = pd.DataFrame({
        "num_1": np.random.normal(50, 10, 60),
        "num_2": np.random.normal(100, 20, 60),
    })
    orch = UniversalOrchestrator()
    res = orch.orchestrate("profile dataset, detect anomalies, and cluster records", df)
    report = res.data.get("synthesis", {})
    assert "key_insights" in report
    assert len(report["key_insights"]) >= 2


# C. Arbitrary column names
def test_C_arbitrary_column_names():
    df = pd.DataFrame({
        "Arbitrary%_Column_Alpha[1]": np.linspace(10, 100, 50),
        "Weird.Beta*Value": np.linspace(5, 50, 50),
    })
    orch = UniversalOrchestrator()
    res = orch.orchestrate("explore data distributions and find correlations", df)
    report = res.data.get("synthesis", {})
    assert len(report.get("relationships", [])) >= 1
    assert_no_nan_or_inf(report)


# D. Evidence preservation
def test_D_evidence_preservation():
    df = pd.DataFrame({"x": range(40), "y": [i * 3 for i in range(40)]})
    engine = InsightSynthesisEngine()
    ev = Evidence(operation="test_calc", calculation="r=1.0", result={"val": 1.0}, confidence=0.95)
    report = engine.synthesize(
        orchestration_result={"task_outputs": {}, "evidence": [ev], "confidence": 0.90},
        dataframe=df,
    )
    assert len(report.evidence) >= 1
    assert any(e.operation == "test_calc" for e in report.evidence)


# E. Provenance preservation
def test_E_provenance_preservation():
    engine = InsightSynthesisEngine()
    mock_stats = {
        "ranked_relationships": [{
            "feature_1": "alpha",
            "feature_2": "beta",
            "correlation": 0.85,
            "p_value": 0.001,
            "effect_size": 0.85,
        }]
    }
    report = engine.synthesize(
        orchestration_result={"task_outputs": {"statistical_analysis": mock_stats}},
    )
    assert len(report.relationships) >= 1
    ins = report.relationships[0]
    assert "agent" in ins.provenance
    assert ins.provenance["agent"] == "StatisticalAnalysisAgent"


# F. Confidence bounds
def test_F_confidence_bounds():
    engine = InsightSynthesisEngine()
    report = engine.synthesize(
        orchestration_result={"task_outputs": {}, "confidence": 0.85},
        dataframe=pd.DataFrame({"x": range(30)}),
    )
    assert 0.0 <= report.overall_confidence <= 1.0
    for ins in report.key_insights:
        assert 0.0 <= ins.confidence <= 1.0
        assert 0.0 <= ins.importance <= 1.0


# G. Duplicate insight merging
def test_G_duplicate_insight_merging():
    engine = InsightSynthesisEngine()
    i1 = SynthesizedInsight(
        category=InsightCategory.RELATIONSHIP.value,
        title="Association: x & y",
        statement="x is correlated with y",
        provenance={"pair": "x_y"},
        confidence=0.85,
    )
    i2 = SynthesizedInsight(
        category=InsightCategory.RELATIONSHIP.value,
        title="Association: y & x",
        statement="y is correlated with x",
        provenance={"pair": "y_x"},
        confidence=0.90,
    )
    merged = engine._suppress_duplicates([i1, i2])
    assert len(merged) == 1
    assert merged[0].confidence == 0.90


# H. Contradiction detection
def test_H_contradiction_detection():
    engine = InsightSynthesisEngine()
    task_outputs = {
        "forecasting": {"confidence": 0.95, "trend_direction": "upward", "target_column": "sales"},
        "anomaly_detection": {"anomaly_count": 25, "detector_used": "IsolationForest"},
    }
    contras = engine._detect_contradictions([], task_outputs)
    assert len(contras) >= 1
    assert isinstance(contras[0], Contradiction)
    assert "anomaly" in contras[0].explanation.lower()


# I. Correlation / Causation protection
def test_I_correlation_causation_protection():
    engine = InsightSynthesisEngine()
    unsafe_text = "Feature A causes higher sales and drives customer behavior because of marketing."
    sanitized = engine._sanitize_causality(unsafe_text)
    assert "causes" not in sanitized.lower()
    assert "drives" not in sanitized.lower()
    assert "because of" not in sanitized.lower()
    assert "associated with" in sanitized.lower()


# J. Statistically significant but practically weak relationship
def test_J_statistically_significant_but_practically_weak_relationship():
    engine = InsightSynthesisEngine()
    mock_stats = {
        "ranked_relationships": [{
            "feature_1": "huge_sample_var_1",
            "feature_2": "huge_sample_var_2",
            "correlation": 0.05,
            "p_value": 0.000001,
            "effect_size": 0.05,
        }]
    }
    report = engine.synthesize(
        orchestration_result={"task_outputs": {"statistical_analysis": mock_stats}},
    )
    ins = report.relationships[0]
    # Weak effect size should result in low-to-moderate importance despite tiny p-value
    assert "weak" in ins.statement.lower()
    assert ins.importance <= 0.85


# K. Strong practical effect
def test_K_strong_practical_effect():
    engine = InsightSynthesisEngine()
    mock_stats = {
        "ranked_relationships": [{
            "feature_1": "spend",
            "feature_2": "revenue",
            "correlation": 0.92,
            "p_value": 0.001,
            "effect_size": 0.92,
        }]
    }
    report = engine.synthesize(
        orchestration_result={"task_outputs": {"statistical_analysis": mock_stats}},
    )
    ins = report.relationships[0]
    assert "strong" in ins.statement.lower()
    assert ins.importance >= 0.80


# L. Forecast interpretation
def test_L_forecast_interpretation():
    engine = InsightSynthesisEngine()
    fc_data = {
        "target_column": "revenue",
        "horizon": 6,
        "model_selected": "AutoARIMA",
        "forecast": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
    }
    report = engine.synthesize(
        orchestration_result={"task_outputs": {"forecasting": fc_data}},
    )
    assert len(report.forecast_findings) >= 1
    fc_ins = report.forecast_findings[0]
    assert "projects" in fc_ins.statement
    assert "revenue" in fc_ins.statement


# M. Anomaly interpretation
def test_M_anomaly_interpretation():
    engine = InsightSynthesisEngine()
    anom_data = {
        "anomaly_count": 5,
        "anomaly_rate": 0.05,
        "detector_used": "IsolationForest",
        "features_analyzed": ["metric_a", "metric_b"],
    }
    report = engine.synthesize(
        orchestration_result={"task_outputs": {"anomaly_detection": anom_data}},
    )
    assert len(report.anomalies) >= 1
    anom_ins = report.anomalies[0]
    assert "statistically unusual" in anom_ins.statement


# N. Clustering interpretation
def test_N_clustering_interpretation():
    engine = InsightSynthesisEngine()
    clust_data = {
        "cluster_count": 3,
        "algorithm": "K-Means",
        "silhouette_score": 0.65,
        "cluster_sizes": {"0": 20, "1": 15, "2": 15},
    }
    report = engine.synthesize(
        orchestration_result={"task_outputs": {"clustering": clust_data}},
    )
    assert len(report.segments) >= 1
    seg_ins = report.segments[0]
    assert "3 natural groups" in seg_ins.statement


# O. Prediction interpretation
def test_O_prediction_interpretation():
    engine = InsightSynthesisEngine()
    pred_data = {
        "selected_model": "RandomForestRegressor",
        "task_type": "regression",
        "target_column": "price",
        "metrics": {"r2": 0.88, "rmse": 4.5},
    }
    report = engine.synthesize(
        orchestration_result={"task_outputs": {"prediction": pred_data}},
    )
    assert len(report.model_findings) >= 1
    pred_ins = report.model_findings[0]
    assert "RandomForestRegressor" in pred_ins.statement
    assert "R² = 0.880" in pred_ins.statement


# P. Data quality insight generation
def test_P_data_quality_insight_generation():
    df = pd.DataFrame({"id_col": [f"ID_{i}" for i in range(40)], "val": [10.0] * 35 + [None] * 5})
    engine = InsightSynthesisEngine()
    report = engine.synthesize(
        orchestration_result={"task_outputs": {}},
        dataframe=df,
    )
    assert len(report.data_quality_findings) >= 1


# Q. Missing analytical result
def test_Q_missing_analytical_result():
    engine = InsightSynthesisEngine()
    report = engine.synthesize(
        orchestration_result={},
    )
    assert isinstance(report, SynthesisReport)
    assert report.executive_summary != ""


# R. Failed analytical task
def test_R_failed_analytical_task():
    engine = InsightSynthesisEngine()
    err_result = AgentResult.error(
        error="Model training failed due to singular matrix.",
        agent_name="Prediction Agent",
    )
    report = engine.synthesize(
        orchestration_result={"task_outputs": {"prediction": err_result.to_dict()}},
    )
    assert isinstance(report, SynthesisReport)


# S. Partial orchestration result
def test_S_partial_orchestration_result():
    df = pd.DataFrame({"a": range(40)})
    orch = UniversalOrchestrator()
    # Profile succeeds, forecasting fails (no date)
    res = orch.orchestrate("profile data and forecast", df)
    assert "synthesis" in res.data
    synthesis = res.data["synthesis"]
    assert "executive_summary" in synthesis


# T. Empty result set
def test_T_empty_result_set():
    engine = InsightSynthesisEngine()
    report = engine.synthesize(
        orchestration_result={"task_outputs": {}},
    )
    assert report.overall_confidence >= 0.0


# U. Malformed result
def test_U_malformed_result():
    engine = InsightSynthesisEngine()
    report = engine.synthesize(
        orchestration_result={"task_outputs": "invalid_structure"},
    )
    assert isinstance(report, SynthesisReport)


# V. Fabricated metric prevention
def test_V_fabricated_metric_prevention():
    engine = InsightSynthesisEngine()
    report = engine.synthesize(
        orchestration_result={"task_outputs": {}},
        dataframe=pd.DataFrame({"x": [1, 2, 3]}),
    )
    # With no forecasting or prediction tasks, no fake R2 or forecast metrics should appear
    assert len(report.forecast_findings) == 0
    assert len(report.model_findings) == 0


# W. Fabricated column prevention
def test_W_fabricated_column_prevention():
    df = pd.DataFrame({"RealCol1": range(30), "RealCol2": range(30)})
    engine = InsightSynthesisEngine()
    report = engine.synthesize(
        orchestration_result={"task_outputs": {}},
        dataframe=df,
    )
    for ins in report.key_insights:
        for c in ins.provenance.get("columns", []):
            assert c in df.columns


# X. Deterministic ordering
def test_X_deterministic_ordering():
    df = pd.DataFrame({"a": range(40), "b": [x * 2 for x in range(40)]})
    orch = UniversalOrchestrator()
    res1 = orch.orchestrate("profile and find correlations", df)
    res2 = orch.orchestrate("profile and find correlations", df)
    rep1 = res1.data["synthesis"]["key_insights"]
    rep2 = res2.data["synthesis"]["key_insights"]
    assert [i["title"] for i in rep1] == [i["title"] for i in rep2]


# Y. FastAPI integration
def test_Y_fastapi_integration():
    client = TestClient(app)
    records = [{"x": float(i), "y": float(i * 2)} for i in range(40)]
    resp = client.post("/api/v1/insights/synthesize", json={
        "dataset": records,
        "command": "profile data and find correlations",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "executive_summary" in data
    assert "key_insights" in data


# Z. Orchestrator integration
def test_Z_orchestrator_integration():
    df = pd.DataFrame({"a": range(50), "b": [x * 2 for x in range(50)]})
    orch = UniversalOrchestrator()
    res = orch.orchestrate("profile data and analyze relationships", df)
    assert res.is_success
    assert "synthesis" in res.data
    assert "executive_summary" in res.data
    assert "key_insights" in res.data


# AA. Existing API compatibility
def test_AA_existing_api_compatibility():
    client = TestClient(app)
    resp = client.post("/api/v1/orchestrate", json={
        "dataset": [{"val_a": 1, "val_b": 2}, {"val_a": 3, "val_b": 4}] * 20,
        "command": "profile this data",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data
    assert "synthesis" in data["result"]


# AB. Complete end-to-end natural-language workflow
def test_AB_complete_end_to_end_natural_language_workflow():
    n = 60
    df = pd.DataFrame({
        "revenue": np.linspace(100, 500, n),
        "marketing_cost": np.random.normal(50, 10, n),
        "region": ["North", "South", "East"] * 20,
    })
    agent = InsightSynthesisAgent()
    orch = UniversalOrchestrator()
    orch_res = orch.orchestrate("profile dataset, find correlations, and segment records", df)
    agent_res = agent.run({"orchestration_result": orch_res.to_dict(), "data": df})
    assert agent_res.is_success
    output = agent_res.output
    assert "executive_summary" in output
    assert len(output["key_insights"]) >= 1