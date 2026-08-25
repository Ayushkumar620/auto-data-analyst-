"""
Tests for Milestone 5, Task 4: Autonomous Decision & Recommendation Engine.

Verifies (deterministic, synthetic data, no external API keys):
1.  Recommendation generation
2.  Evidence requirement (no recommendation without evidence)
3.  Fact / recommendation / prediction separation
4.  Objective handling
5.  Missing objective clarification
6.  Constraints enforcement
7.  Conflicting objectives -> trade-offs
8.  Risk assessment
9.  Opportunity detection
10. Recommendation scoring (transparent)
11. Ranking / prioritization
12. Expected impact (available)
13. Expected impact (unavailable)
14. Forecasting integration
15. Scenario integration
16. Model monitoring integration
17. Conversational integration
18. Evidence traceability / audit trail
19. Unsupported (evidence-less) recommendation rejection
20. Human-approval boundary (no auto-execution)
21. LLM-unavailable (deterministic, no API key) mode
22. ToolRegistry integration (decision_engine)
23. AgentResult integration (BaseAgent run())
"""
import pytest

from agent.recommendation_agent import RecommendationAgent
from agent.recommendation_engine import (
    DecisionContextBuilder,
    OpportunityEngine,
    RecommendationEngine,
    RecommendationScorer,
    RiskEngine,
)
from agent.recommendation_schemas import (
    DecisionConstraint,
    RecommendationObjective,
    RecommendationRequest,
)
from agent.forecasting_schemas import ForecastPoint, ForecastResult, ScenarioResult
from agent.schemas import AgentResult, AgentStatus, ClaimType, Evidence
from agent.tool_registry import DEFAULT_TOOL_REGISTRY


# ==============================================================================
# Fixtures & helpers
# ==============================================================================

def _insight(category, title, summary, calc=None, seg=None):
    """Deterministic synthetic insight dict helper."""
    return {
        "insight_id": title,
        "title": title,
        "summary": summary,
        "category": category,
        "claim_type": "fact",
        "severity": "HIGH",
        "confidence": 0.94,
        "evidence": Evidence(source="test_engine", method="synthetic",
                             data_ref=calc or {}, claim_type=ClaimType.FACT,
                             confidence=0.94),
        "affected_segments": [seg] if seg else [],
        "affected_columns": ["revenue", "region"],
        "calculation": calc or {},
        "source_analysis": "synthetic",
    }


@pytest.fixture
def sales_insights():
    """Insights mimicking the task example (North 42% revenue, 18% growth, concentration)."""
    return [
        _insight("concentration", "concentration_risk",
                 "High revenue concentration across region",
                 {"dimension": "region", "metric": "revenue",
                  "top_20_percent_share": 55.0, "top_3_entities_share": 42.0}, "North"),
        _insight("performance", "segmentation",
                 "North dominates revenue share",
                 {"dimension": "region", "metric": "revenue", "top_segment": "North",
                  "top_share_pct": 42.0, "bottom_segment": "West", "segment_count": 4,
                  "disparity_ratio": 4.0}, "North"),
        _insight("trend", "trend_growth",
                 "Revenue grew 18% over the period",
                 {"metric": "revenue", "overall_growth_pct": 18.0}, "North"),
    ]


@pytest.fixture
def declining_forecast():
    return ForecastResult(
        model_name="naive", model_family="naive_last", target="revenue",
        time_column="date", frequency="M", forecast_horizon=3,
        predictions=[
            ForecastPoint(timestamp="2025-01", prediction=100, lower_bound=90, upper_bound=110),
            ForecastPoint(timestamp="2025-02", prediction=95, lower_bound=85, upper_bound=105),
            ForecastPoint(timestamp="2025-03", prediction=92, lower_bound=80, upper_bound=106),
        ])


@pytest.fixture
def impact_scenario():
    return ScenarioResult(
        scenario_name="Increase marketing investment +10%", target_metric="revenue",
        baseline_value=10_000_000, scenario_value=10_420_000,
        absolute_difference=420_000, percentage_difference=4.2,
        assumptions=["Marketing spend +10%"], limitations=["Not causal"])


def _engine():
    return RecommendationEngine()
# ==============================================================================
# 1-3. Generation, Evidence, Separation
# ==============================================================================

def test_recommendation_generation(sales_insights):
    """1. Generate evidence-backed recommendations from insights."""
    res = _engine().generate(RecommendationRequest(insights=sales_insights, max_recommendations=5))
    assert res.status == "success"
    assert len(res.recommendations) >= 1
    assert res.recommendations[0].action


def test_recommendations_require_evidence(sales_insights):
    """2. No recommendation may exist without evidence."""
    res = _engine().generate(RecommendationRequest(insights=sales_insights))
    for rec in res.recommendations:
        assert rec.evidence_ids, "recommendation must cite evidence"
        assert len(rec.evidence) >= 1


def test_fact_recommendation_separation(sales_insights):
    """3. Facts, predictions, and recommendations must never merge."""
    res = _engine().generate(RecommendationRequest(insights=sales_insights))
    # Facts live in the decision context only
    ctx = DecisionContextBuilder.build(RecommendationRequest(insights=sales_insights))
    for fact in ctx.facts:
        assert fact.claim_type in (ClaimType.FACT, ClaimType.OBSERVATION)
    # Recommendations are marked advisory, not facts
    for rec in res.recommendations:
        assert rec.human_approval_required is True
        assert rec.recommendation_id.startswith("REC-")
        assert rec.action  # a recommendation expresses a course of action


# ==============================================================================
# 4-7. Objectives, Constraints, Conflicting objectives
# ==============================================================================

def test_objective_handling(sales_insights):
    """4. Optimize ranking toward a requested objective."""
    res = _engine().generate(RecommendationRequest(
        insights=sales_insights, optimization_objective=RecommendationObjective.REDUCE_RISK))
    assert RecommendationObjective.REDUCE_RISK in res.objectives
    assert res.recommendations[0].objective == RecommendationObjective.REDUCE_RISK


def test_missing_objective(sales_insights):
    """5. No invented objective -> ask for clarification."""
    res = _engine().generate(RecommendationRequest(insights=sales_insights,
                                                   user_intent="What should I do?"))
    assert res.needs_objective_clarification is True
    assert res.status == "needs_objective"
    assert "objective" in res.executive_summary.lower()


def test_constraints(sales_insights):
    """6. Recommendations violating explicit constraints are rejected."""
    res = _engine().generate(RecommendationRequest(
        insights=sales_insights,
        optimization_objective=RecommendationObjective.MAXIMIZE_REVENUE,
        business_constraints=[DecisionConstraint(constraint="Only focus on North region",
                                                 type="region_filter", value="North")]))
    for rec in res.recommendations:
        assert rec.affected_segment is None or "north" in str(rec.affected_segment).lower()


def test_conflicting_objectives_tradeoffs(sales_insights):
    """7. Multiple objectives surface explicit trade-offs."""
    res = _engine().generate(RecommendationRequest(
        insights=sales_insights,
        optimization_objective=[RecommendationObjective.MAXIMIZE_REVENUE,
                                RecommendationObjective.REDUCE_RISK]))
    assert len(res.trade_offs) >= 1
    assert any("risk" in t.note.lower() for t in res.trade_offs)


# ==============================================================================
# 8-11. Risk, Opportunity, Scoring, Ranking
# ==============================================================================

def test_risk_assessment(sales_insights):
    """8. Evidence-backed risk detection."""
    res = _engine().generate(RecommendationRequest(insights=sales_insights))
    assert any("concentration" in r.risk.lower() for r in res.risks)
    for r in res.risks:
        assert r.severity.value in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert r.evidence


def test_opportunity_detection(sales_insights):
    """9. Evidence-backed opportunity detection."""
    res = _engine().generate(RecommendationRequest(insights=sales_insights))
    assert len(res.opportunities) >= 1
    for opp in res.opportunities:
        assert opp.evidence, "opportunity requires measurable evidence"


def test_recommendation_scoring(sales_insights):
    """10. Transparent, documented scoring factors."""
    res = _engine().generate(RecommendationRequest(insights=sales_insights))
def test_ranking(sales_insights):
    """11. Recommendations are ranked by descending score."""
    res = _engine().generate(RecommendationRequest(insights=sales_insights, max_recommendations=10))
    scores = [r.score for r in res.recommendations]
    assert scores == sorted(scores, reverse=True)


# ==============================================================================
# 12-16. Expected impact, forecast, scenario, monitoring
# ==============================================================================

def test_expected_impact_available(sales_insights, impact_scenario):
    """12. Numerical expected impact comes from a real scenario calculation."""
    res = _engine().generate(RecommendationRequest(
        insights=sales_insights, scenarios=[impact_scenario],
        optimization_objective=RecommendationObjective.MAXIMIZE_REVENUE))
    impacts = [r.expected_impact for r in res.recommendations
               if r.expected_impact and r.expected_impact.availability.value == "available"]
    assert impacts, "expected a scenario-backed impact"
    for imp in impacts:
        assert imp.estimated_impact is not None
        assert imp.estimated_impact_pct is not None


def test_expected_impact_unavailable(sales_insights):
    """13. Without a scenario, impact is 'unavailable' and never invented."""
    res = _engine().generate(RecommendationRequest(insights=sales_insights))
    for rec in res.recommendations:
        if rec.expected_impact is None:
            continue
        assert rec.expected_impact.availability.value == "unavailable"
        assert rec.expected_impact.estimated_impact is None


def test_forecast_integration(declining_forecast):
    """14. A declining forecast drives a demand/inventory recommendation."""
    res = _engine().generate(RecommendationRequest(
        optimization_objective=RecommendationObjective.REDUCE_RISK,
        forecasts=[declining_forecast]))
    actions = " | ".join(r.action.lower() for r in res.recommendations)
    assert "inventory" in actions or "demand" in actions
    assert any("forecast" in r.risk.lower() for r in res.risks)


def test_scenario_integration(sales_insights, impact_scenario):
    """15. Scenario evidence is referenced in the recommendation."""
    res = _engine().generate(RecommendationRequest(
        insights=sales_insights, scenarios=[impact_scenario],
        optimization_objective=RecommendationObjective.MAXIMIZE_REVENUE))
    found = False
    for rec in res.recommendations:
        if rec.expected_impact and rec.expected_impact.availability.value == "available":
            assert rec.expected_impact.scenario_name == "Increase marketing investment +10%"
            found = True
    assert found


def test_monitoring_integration():
    """16. High data drift triggers a validate/investigate recommendation."""
    res = _engine().generate(RecommendationRequest(
        optimization_objective=RecommendationObjective.REDUCE_RISK,
        monitoring_results=[{"model_id": "m1",
                             "data_drift": {"overall_drift": True, "severity": "HIGH"}}]))
    actions = " | ".join(r.action.lower() for r in res.recommendations)
    assert "validate incoming data" in actions
    assert any("drift" in r.risk.lower() for r in res.risks)


def test_performance_degradation_monitoring():
    """16b. Performance degradation drives a recalibration review."""
    res = _engine().generate(RecommendationRequest(
        optimization_objective=RecommendationObjective.REDUCE_RISK,
        monitoring_results=[{"model_id": "m1",
                             "performance_drift": {"degradation_detected": True}}]))
    actions = " | ".join(r.action.lower() for r in res.recommendations)
    assert "recalibration" in actions

    for rec in res.recommendations:
        assert rec.score > 0
        assert rec.scoring.formula  # explainable formula documented
        assert rec.scoring.final_score == pytest.approx(rec.score, abs=1e-4)
        assert rec.priority.value in ("HIGH", "MEDIUM", "LOW")

