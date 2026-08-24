"""Comprehensive test suite for Phase 9: Evidence-Based Insights & Structured Claim Attribution."""
import numpy as np
import pandas as pd
import pytest

from agent.schemas import AgentResult, AgentStatus, ClaimType
from agent.agents import InsightAgent
from agent.insights import InsightsEngine
from agent.planner import PlannerAgent
from backend.app.core.evidence_insights import (
    EvidenceBasedInsightsEngine,
    InsightsCatalog,
    StructuredInsight,
)


@pytest.fixture
def sales_df():
    np.random.seed(42)
    n = 60
    ad_spend = np.random.uniform(1000, 5000, n)
    # Strong correlation with noise
    revenue = ad_spend * 3.5 + np.random.normal(0, 500, n)
    # Inject an outlier in revenue
    revenue[0] = 50000.0

    return pd.DataFrame({
        "ad_spend": ad_spend,
        "revenue": revenue,
        "region": np.random.choice(["North", "South", "East", "West"], n, p=[0.6, 0.2, 0.1, 0.1]),  # High concentration
        "units_sold": (revenue / 50).astype(int),
    })


# ==============================================================================
# 1. Epistemic Claim Types & Facts Generation Tests
# ==============================================================================

def test_generate_facts_arithmetic_precision(sales_df):
    """Verify that facts contain exact calculations, 1.0 confidence, and FACT claim type."""
    engine = EvidenceBasedInsightsEngine(sales_df)
    facts = engine.generate_facts(sales_df)

    assert len(facts) >= 2
    for f in facts:
        assert isinstance(f, StructuredInsight)
        assert f.claim_type == ClaimType.FACT
        assert f.confidence == 1.0
        assert len(f.supporting_metrics) > 0
        assert len(f.caveats) >= 1

    # Fact on dataset volume
    vol_fact = next(f for f in facts if "60 records" in f.text or "total_records" in f.supporting_metrics)
    assert vol_fact.supporting_metrics["total_records"] == 60
    assert vol_fact.supporting_metrics["completeness_pct"] == 100.0


# ==============================================================================
# 2. Observations & Anomaly Detection Tests
# ==============================================================================

def test_generate_observations_outliers_and_skew(sales_df):
    """Verify that statistical observations are flagged with OBSERVATION claim type."""
    engine = EvidenceBasedInsightsEngine(sales_df)
    observations = engine.generate_observations(sales_df)

    assert len(observations) >= 1
    for obs in observations:
        assert obs.claim_type == ClaimType.OBSERVATION
        assert obs.confidence >= 0.90
        assert obs.recommended_chart is not None

    # Check that the injected outlier in revenue was captured
    outlier_obs = next((o for o in observations if "outlier" in o.text.lower() and "revenue" in o.text.lower()), None)
    assert outlier_obs is not None
    assert outlier_obs.supporting_metrics["outlier_count"] >= 1


# ==============================================================================
# 3. Correlation & Mandatory Non-Causal Disclaimers
# ==============================================================================

def test_generate_correlations_non_causal_disclaimers(sales_df):
    """Verify that correlations strictly include non-causal disclaimers and CORRELATION claim type."""
    engine = EvidenceBasedInsightsEngine(sales_df)
    correlations = engine.generate_correlations(sales_df, threshold=0.50)

    assert len(correlations) >= 1
    for corr in correlations:
        assert corr.claim_type == ClaimType.CORRELATION
        assert "pearson_r" in corr.supporting_metrics
        assert "r_squared" in corr.supporting_metrics

        # Critical architectural invariant: Correlation is NEVER presented as causation
        has_non_causal_caveat = any(
            "not imply causation" in c.lower() or "correlation does not" in c.lower()
            for c in corr.caveats
        )
        assert has_non_causal_caveat is True, f"Correlation insight missing non-causal disclaimer: {corr.caveats}"


# ==============================================================================
# 4. Inferences, Recommendations & Catalog Synthesis
# ==============================================================================

def test_catalog_synthesis_and_model_inference(sales_df):
    """Verify full catalog assembly with model-based inferences and recommendations."""
    engine = EvidenceBasedInsightsEngine(sales_df)
    mock_model_result = {
        "best_model": {
            "model_name": "Gradient Boosting Regressor",
            "primary_metric_name": "R2",
            "primary_metric_value": 0.94,
            "metrics": {"r2_score": 0.94, "rmse": 320.5},
            "feature_importances": {"ad_spend": 0.88, "units_sold": 0.12},
        }
    }

    catalog = engine.build_catalog(sales_df, model_result=mock_model_result)
    assert isinstance(catalog, InsightsCatalog)
    assert catalog.total_insights == len(catalog.insights)
    assert len(catalog.facts) >= 1
    assert len(catalog.observations) >= 1
    assert len(catalog.correlations) >= 1
    assert len(catalog.inferences) >= 1
    assert len(catalog.recommendations) >= 1

    # Check inference
    inf = catalog.inferences[0]
    assert inf.claim_type == ClaimType.INFERENCE
    assert "Gradient Boosting" in inf.text
    assert inf.supporting_metrics["r2_score"] == 0.94

    # Check recommendation
    rec = catalog.recommendations[0]
    assert rec.claim_type == ClaimType.RECOMMENDATION
    assert len(rec.data_references) >= 1


# ==============================================================================
# 5. Agent & Planner Integration Tests
# ==============================================================================

def test_insight_agent_structured_insights_action(sales_df):
    """Verify InsightAgent runs structured insights and attaches verified Evidence."""
    agent = InsightAgent()
    result = agent.run({"data": sales_df, "type": "structured"})

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.agent == "Insight Agent"
    assert "facts" in result.output["result"]
    assert "correlations" in result.output["result"]
    assert len(result.evidence) >= 1


def test_planner_agent_structured_insights(sales_df):
    """Verify PlannerAgent routing for 'structured_insights' action."""
    planner = PlannerAgent(data=sales_df)
    result = planner.run_agent({"action": "structured_insights"})

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.validation is not None
    assert result.validation.passed is True
    assert "result" in result.output
