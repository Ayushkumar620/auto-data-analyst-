"""End-to-end verification of statistical relationship analysis pipeline, evidence claims, subgroup consistency, FDR correction, and outlier analysis."""
import pandas as pd
import pytest

from agent.loader import load_data
from agent.agent_result import AgentResult, AgentStatus, ClaimType
from agent.command_orchestrator import AutonomousCommandOrchestrator
from agent.orchestrator import UniversalOrchestrator
from agent.statistical_analysis_engine import StatisticalAnalysisEngine
from agent.statistical_analysis_agent import StatisticalAnalysisAgent


PROMPT = (
    "Analyze the relationships in this dataset comprehensively. Identify the strongest statistically "
    "significant relationships using Pearson and Spearman correlations, report both raw and FDR-adjusted "
    "p-values, distinguish statistical significance from practical effect size, identify relationships "
    "that are sensitive to outliers, and explain whether the relationships are consistent across customer "
    "segments and product groups. Do not make causal claims. Also identify any relationship that appears "
    "weak overall but becomes strong within a subgroup."
)


@pytest.fixture
def stress_df():
    return load_data("uploads/real_world_analyst_stress_test.csv")


def test_statistical_analysis_engine_stress_dataset(stress_df):
    engine = StatisticalAnalysisEngine()
    res = engine.analyze(data=stress_df)

    assert "relationships" in res
    assert len(res["relationships"]) >= 100
    assert "top_relationships" in res
    assert len(res["top_relationships"]) > 0

    # Subgroup analysis verification
    assert "subgroup_analysis" in res
    sub = res["subgroup_analysis"]
    dims = sub.get("dimensions_evaluated", [])
    assert "client_segment" in dims
    assert "product_group" in dims
    assert len(sub.get("subgroup_consistency_summary", [])) > 0

    # Metric verification on first relationship
    rel = res["relationships"][0]
    assert "pearson_r" in rel
    assert "spearman_rho" in rel
    assert "p_value" in rel
    assert "adjusted_p_value" in rel
    assert "valid_rows" in rel
    assert "outlier_sensitive" in rel
    assert "effect_strength" in rel


def test_statistical_analysis_agent_evidence_generation(stress_df):
    agent = StatisticalAnalysisAgent()
    res = agent.run({"data": stress_df})

    assert res.is_success
    assert res.evidence is not None
    assert len(res.evidence) >= 15

    for ev in res.evidence:
        assert ev.claim_type == ClaimType.CORRELATION
        assert ev.operation is not None
        assert "feature_x" in ev.data_ref
        assert "feature_y" in ev.data_ref
        assert "pearson_r" in ev.data_ref
        assert "spearman_rho" in ev.data_ref
        assert "adjusted_p_value" in ev.data_ref


def test_autonomous_command_orchestrator_fulfillment(stress_df):
    orch = AutonomousCommandOrchestrator()
    res = orch.execute_command(PROMPT, stress_df)

    assert res.final_explanation is not None
    assert len(res.evidence) >= 15
    assert len(res.relationships) >= 100
    assert len(res.top_relationships) > 0
    assert "client_segment" in res.subgroup_analysis.get("dimensions_evaluated", [])
    assert "product_group" in res.subgroup_analysis.get("dimensions_evaluated", [])

    explanation = res.final_explanation
    assert "Pearson r" in explanation
    assert "Spearman ρ" in explanation
    assert "FDR-adjusted p-value" in explanation
    assert "Outlier Sensitivity" in explanation
    assert "Subgroup Consistency" in explanation
    assert "client_segment" in explanation or "product_group" in explanation


def test_universal_orchestrator_fulfillment(stress_df):
    orch = UniversalOrchestrator()
    res = orch.orchestrate(command=PROMPT, data=stress_df)

    assert res.is_success or res.status == AgentStatus.PARTIAL
    assert len(res.evidence) >= 15
    assert "relationships" in res.result
    assert len(res.result["relationships"]) >= 100
    assert "top_relationships" in res.result
    assert "subgroup_analysis" in res.result
    assert "client_segment" in res.result["subgroup_analysis"].get("dimensions_evaluated", [])
    assert "product_group" in res.result["subgroup_analysis"].get("dimensions_evaluated", [])
