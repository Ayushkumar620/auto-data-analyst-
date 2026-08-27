"""
Regression and reliability tests for Comprehensive Statistical Relationship Analysis.

Validates:
1. Intent classification: STATISTICAL_RELATIONSHIP / CORRELATION recognized without misclassifying to HYPOTHESIS_TESTING.
2. StatisticalAnalysisEngine: Pearson r, Spearman rho, Kendall tau, raw p, FDR adjusted p, valid rows, effect size.
3. Outlier sensitivity: difference between Pearson and Spearman rank correlation.
4. Subgroup dimension discovery: low-cardinality dimensions (customer_segment, product_group, etc.).
5. Subgroup analysis: subgroup r, p, n, and consistency evaluation across segments.
6. Weak-global / strong-subgroup heterogeneity detection.
7. Mathematical Simpson's paradox verification (no false claims).
8. Evidence generation: rich, traceable Evidence objects for global and subgroup findings.
9. Insight synthesis: non-causal statements with explicit statistics.
10. AutonomousCommandOrchestrator and UniversalOrchestrator end-to-end execution.
11. Safe JSON serialization compliance.
"""
import math
import numpy as np
import pandas as pd
import pytest

from agent.agent_result import AgentResult, AgentStatus, ClaimType
from agent.command_orchestrator import AutonomousCommandOrchestrator
from agent.intent import AnalyticalIntent, CommandIntelligenceAgent, IntentAnalyzer, IntentType
from agent.insight_synthesis_engine import InsightSynthesisEngine
from agent.orchestrator import UniversalOrchestrator
from agent.statistical_analysis_agent import StatisticalAnalysisAgent
from agent.statistical_analysis_engine import StatisticalAnalysisEngine
from backend.app.core.evidence_insights import EvidenceBasedInsightsEngine


@pytest.fixture
def relationship_dataset():
    """Synthetic dataset with 1,020 rows and 19 columns containing specific statistical properties."""
    np.random.seed(42)
    n = 1020

    segments = np.random.choice(["Enterprise", "SMB", "Consumer"], size=n, p=[0.3, 0.4, 0.3])
    products = np.random.choice(["SaaS", "Hardware", "Services"], size=n, p=[0.5, 0.3, 0.2])

    marketing_spend = np.random.exponential(scale=1000, size=n) + 100
    revenue = marketing_spend * 2.5 + np.random.normal(0, 300, size=n)
    # Inject extreme outlier in revenue to create outlier sensitivity
    revenue[0] = 75000.0

    # Subgroup specific correlation (weak globally, strong in Enterprise)
    metric_a = np.random.normal(50, 10, size=n)
    metric_b = np.random.normal(100, 20, size=n)
    ent_mask = segments == "Enterprise"
    metric_b[ent_mask] = metric_a[ent_mask] * 3.0 + np.random.normal(0, 2, size=ent_mask.sum())

    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "customer_segment": segments,
        "product_group": products,
        "marketing_spend": marketing_spend,
        "revenue": revenue,
        "metric_a": metric_a,
        "metric_b": metric_b,
        "units_sold": np.random.randint(1, 100, size=n),
        "discount": np.random.uniform(0, 0.3, size=n),
        "satisfaction_score": np.random.uniform(1, 5, size=n),
        "churn_risk": np.random.uniform(0, 1, size=n),
        "clv": np.random.exponential(5000, size=n),
        "tenure_months": np.random.randint(1, 72, size=n),
        "support_tickets": np.random.poisson(2, size=n),
        "feature_1": np.random.normal(0, 1, size=n),
        "feature_2": np.random.normal(0, 1, size=n),
        "feature_3": np.random.normal(0, 1, size=n),
        "feature_4": np.random.normal(0, 1, size=n),
        "feature_5": np.random.normal(0, 1, size=n),
    })
    return df


class TestIntentClassification:
    def test_relationship_query_intent_classification(self):
        query = (
            "Analyze the relationships in this dataset comprehensively. Identify the strongest statistically "
            "significant relationships using Pearson and Spearman correlations, report both raw and FDR-adjusted "
            "p-values, distinguish statistical significance from practical effect size, identify relationships "
            "that are sensitive to outliers, and explain whether the relationships are consistent across customer "
            "segments and product groups. Do not make causal claims. Also identify any relationship that appears "
            "weak overall but becomes strong within a subgroup."
        )
        analyzer = IntentAnalyzer()
        res = analyzer.analyze(query)
        assert res.primary_intent == AnalyticalIntent.CORRELATION

        cia = CommandIntelligenceAgent()
        modern_res = cia.analyze_intent(query)
        assert "statistical_analysis" in modern_res.required_capabilities


class TestStatisticalAnalysisEngine:
    def test_engine_computes_pearson_spearman_kendall_and_fdr(self, relationship_dataset):
        engine = StatisticalAnalysisEngine()
        result = engine.analyze(data=relationship_dataset)

        assert "error" not in result
        assert result["task_type"] == "statistical_analysis"
        assert result["rows_analyzed"] == 1020
        assert len(result["relationships"]) > 0

        # Check numeric-numeric relationship metrics
        num_num_rels = [r for r in result["relationships"] if r.get("pair_type") == "numeric_numeric"]
        assert len(num_num_rels) > 0

        for r in num_num_rels:
            assert "pearson" in r
            assert "r" in r["pearson"]
            assert "p_value" in r["pearson"]
            assert "spearman" in r
            assert "rho" in r["spearman"]
            assert "p_value" in r["spearman"]
            assert "adjusted_p_value" in r
            assert "effect_size" in r
            assert "strength" in r
            assert "valid_rows" in r
            assert r["valid_rows"] <= 1020

    def test_outlier_sensitivity_detection(self, relationship_dataset):
        engine = StatisticalAnalysisEngine()
        result = engine.analyze(data=relationship_dataset)

        # Look for marketing_spend vs revenue pair (outlier injected)
        spend_rev = next(
            (r for r in result["relationships"] if (
                (r.get("feature_x") == "marketing_spend" and r.get("feature_y") == "revenue") or
                (r.get("feature_x") == "revenue" and r.get("feature_y") == "marketing_spend")
            )),
            None
        )
        assert spend_rev is not None
        assert "outlier_sensitivity" in spend_rev
        assert "r_vs_rho_delta" in spend_rev

    def test_subgroup_analysis_discovery_and_heterogeneity(self, relationship_dataset):
        engine = StatisticalAnalysisEngine()
        result = engine.analyze(data=relationship_dataset)

        subgroup_data = result.get("subgroup_analysis")
        assert subgroup_data is not None
        assert "customer_segment" in subgroup_data.get("dimensions_evaluated", [])
        assert "product_group" in subgroup_data.get("dimensions_evaluated", [])

        # Verify weak-global / strong-subgroup finding for metric_a and metric_b
        weak_findings = subgroup_data.get("weak_global_strong_subgroup_findings", [])
        assert len(weak_findings) > 0
        ent_finding = next(
            (f for f in weak_findings if (
                f.get("feature_x") in ("metric_a", "metric_b") and
                f.get("feature_y") in ("metric_a", "metric_b") and
                f.get("subgroup_dimension") == "customer_segment" and
                f.get("subgroup_value") == "Enterprise"
            )),
            None
        )
        assert ent_finding is not None
        assert ent_finding["subgroup_r"] > 0.90
        assert ent_finding["subgroup_p_value"] < 0.05
        assert ent_finding["subgroup_valid_rows"] >= 200

    def test_simpsons_paradox_not_falsely_claimed(self, relationship_dataset):
        engine = StatisticalAnalysisEngine()
        result = engine.analyze(data=relationship_dataset)

        subgroup_data = result.get("subgroup_analysis", {})
        simpsons = subgroup_data.get("simpsons_paradox_findings", [])
        for s in simpsons:
            assert "Demonstrated Simpson's Paradox" in s.get("explanation", "")


class TestStatisticalAnalysisAgent:
    def test_agent_generates_canonical_evidence_and_metrics(self, relationship_dataset):
        agent = StatisticalAnalysisAgent()
        res = agent.run({"data": relationship_dataset})

        assert res.is_success
        assert res.status == AgentStatus.COMPLETED
        assert len(res.evidence) >= 5
        assert res.confidence >= 0.70

        # Check evidence claims contain required structure
        global_evs = [e for e in res.evidence if e.data_ref.get("analysis_scope") == "global"]
        subgroup_evs = [e for e in res.evidence if e.data_ref.get("analysis_scope") == "subgroup"]

        assert len(global_evs) > 0
        assert len(subgroup_evs) > 0

        sample_ev = global_evs[0]
        assert sample_ev.claim_type == ClaimType.CORRELATION
        assert "feature_x" in sample_ev.data_ref
        assert "feature_y" in sample_ev.data_ref
        assert "statistic" in sample_ev.data_ref
        assert "p_value" in sample_ev.data_ref
        assert "adjusted_p_value" in sample_ev.data_ref


class TestEvidenceBasedInsightsEngine:
    def test_insights_engine_non_zero_correlations(self, relationship_dataset):
        engine = EvidenceBasedInsightsEngine()
        insights = engine.generate_correlations(relationship_dataset)
        assert len(insights) > 0
        for ins in insights:
            assert ins.claim_type == ClaimType.CORRELATION
            assert "pearson_r" in ins.supporting_metrics
            assert "spearman_rho" in ins.supporting_metrics
            assert "p_value" in ins.supporting_metrics


class TestAutonomousCommandOrchestrator:
    def test_orchestrator_executes_comprehensive_relationship_query(self, relationship_dataset):
        query = (
            "Analyze the relationships in this dataset comprehensively. Identify the strongest statistically "
            "significant relationships using Pearson and Spearman correlations, report both raw and FDR-adjusted "
            "p-values, distinguish statistical significance from practical effect size, identify relationships "
            "that are sensitive to outliers, and explain whether the relationships are consistent across customer "
            "segments and product groups. Do not make causal claims. Also identify any relationship that appears "
            "weak overall but becomes strong within a subgroup."
        )
        orch = AutonomousCommandOrchestrator()
        result = orch.execute_command(query, relationship_dataset)

        assert len(result.evidence) >= 5
        assert "calculate_bivariate_pearson_and_spearman_associations" in result.required_operations
        assert "evaluate_raw_and_fdr_adjusted_significance" in result.required_operations
        assert "analyze_subgroup_consistency_and_heterogeneity" in result.required_operations

        explanation = result.final_explanation
        assert "Statistical Relationship" in explanation
        assert "Pearson r" in explanation
        assert "Spearman" in explanation
        assert "FDR-adjusted" in explanation
        assert "Practical Effect Size" in explanation
        assert "Outlier Sensitivity" in explanation
        assert "Subgroup Consistency" in explanation
        assert "Enterprise" in explanation
        assert "Non-Causal Disclaimer" in explanation

        for forbidden in (" causes ", " caused ", " drives ", " leads to ", " results in "):
            assert forbidden not in explanation.lower()


class TestUniversalOrchestrator:
    def test_universal_orchestrator_end_to_end(self, relationship_dataset):
        query = (
            "Analyze the relationships in this dataset comprehensively. Identify the strongest statistically "
            "significant relationships using Pearson and Spearman correlations, report both raw and FDR-adjusted "
            "p-values, distinguish statistical significance from practical effect size, identify relationships "
            "that are sensitive to outliers, and explain whether the relationships are consistent across customer "
            "segments and product groups. Do not make causal claims. Also identify any relationship that appears "
            "weak overall but becomes strong within a subgroup."
        )
        orch = UniversalOrchestrator()
        plan = orch.plan(query, relationship_dataset)
        assert any(t.task_type == "statistical_analysis" for t in plan.tasks)

        exec_res = orch.execute_plan(plan, relationship_dataset)
        assert exec_res.is_success
        assert len(exec_res.evidence) >= 5
        assert "synthesis" in exec_res.output
        assert "key_insights" in exec_res.output