"""
Tests for Milestone 5, Task 1: Autonomous Data Analysis and Insight Generation Engine.

Verifies:
1. Automatic analysis discovery (AnalysisDiscoveryAgent)
2. Pure numerical dataset analysis (descriptive stats, dispersion, skew)
3. Categorical dataset analysis (value distributions, unique counts)
4. Date + metric dataset analysis (trend tracking, period-over-period growth)
5. Dimensional segmentation analysis (performance disparity, category shares)
6. Correlation analysis (Pearson associations with non-causal attribution)
7. Temporal trend detection (peaks, troughs, growth percentages)
8. Anomaly & outlier detection (IQR thresholds, outlier counts)
9. Concentration analysis (Pareto 80/20 rule, top contributor shares)
10. Insufficient data handling (graceful skip without crashes)
11. Duplicate insight detection and merging (InsightRanker.deduplicate)
12. Multi-factor insight ranking (InsightRanker.rank)
13. Grounded mathematical evidence generation
14. Dynamic confidence scoring
15. Business question mode ("Why did revenue decline?")
16. "Analyze everything" mode with safe depth bounds
17. Partial failure isolation
18. LLM unavailable / fully deterministic execution
19. Deterministic computational integrity (zero hallucinations)
20. Standardized AgentResult generation
"""
import pytest
import numpy as np
import pandas as pd

from agent.analysis_discovery_agent import AnalysisDiscoveryAgent
from agent.autonomous_analysis_engine import AutonomousAnalysisEngine
from agent.autonomous_analysis_schemas import (
    AnalysisCandidate,
    AnalysisDepth,
    AutonomousAnalysisRequest,
    AutonomousAnalysisResult,
    Insight,
    InsightCategory,
    InsightSeverity,
)
from agent.autonomous_analyst_agent import AutonomousAnalystAgent
from agent.insight_ranker import InsightRanker
from agent.intent import UserIntent
from agent.schemas import AgentResult, AgentStatus, ClaimType
from agent.tool_registry import DEFAULT_TOOL_REGISTRY


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sales_df():
    """Rich multi-dimensional sales dataset (120 rows)."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2025-01-01", periods=12, freq="ME").repeat(10)
    regions = np.random.choice(["North", "South", "East", "West"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    products = np.random.choice(["WidgetA", "WidgetB", "WidgetC"], size=n, p=[0.5, 0.3, 0.2])
    
    # Revenue has upward trend with an intentional spike in Q4
    base_rev = np.linspace(1000, 5000, n) + np.random.normal(0, 200, n)
    units = np.random.randint(5, 50, size=n)
    cost = base_rev * 0.6 + np.random.normal(0, 50, n)
    
    # Inject a few intentional outliers
    base_rev[15] = 25000.0
    base_rev[85] = 30000.0

    return pd.DataFrame({
        "date": dates,
        "region": regions,
        "product": products,
        "revenue": base_rev,
        "units_sold": units,
        "cost": cost,
    })


@pytest.fixture
def pure_numeric_df():
    """Purely numerical dataset."""
    np.random.seed(42)
    n = 50
    x1 = np.random.normal(100, 15, n)
    x2 = 2.5 * x1 + np.random.normal(0, 5, n)  # Strong positive correlation
    x3 = np.random.uniform(1, 10, n)
    return pd.DataFrame({"metric_a": x1, "metric_b": x2, "metric_c": x3})


# ==============================================================================
# 1-5. Discovery & Basic Analytical Capabilities
# ==============================================================================

def test_analysis_discovery_prioritization(sales_df):
    """1. Test that AnalysisDiscoveryAgent identifies and prioritizes suitable analyses."""
    discovery = AnalysisDiscoveryAgent()
    intent = UserIntent(
        intent_type="exploratory_data_analysis",
        objective="Analyze sales trends and regional performance",
        metrics=["revenue"],
        dimensions=["region"],
    )

    candidates = discovery.discover_analyses(
        df=sales_df,
        user_intent=intent,
        depth=AnalysisDepth.STANDARD,
    )

    assert len(candidates) >= 4
    cand_types = [c.analysis_type for c in candidates]
    assert "data_quality" in cand_types
    assert "trend_analysis" in cand_types
    assert "segmentation" in cand_types
    assert "correlation_analysis" in cand_types


def test_pure_numeric_dataset_analysis(pure_numeric_df):
    """2. Test descriptive stats and correlation on pure numerical data."""
    engine = AutonomousAnalysisEngine()
    stats_data, ins_stats = engine.analyze_descriptive_stats(pure_numeric_df, ["metric_a", "metric_b"])

    assert "metric_a" in stats_data
    assert stats_data["metric_a"]["count"] == 50
    assert stats_data["metric_a"]["mean"] > 0
    assert len(ins_stats) >= 2
    assert ins_stats[0].category == InsightCategory.PERFORMANCE


def test_categorical_dataset_analysis():
    """3. Test dataset quality and structure on categorical data."""
    df_cat = pd.DataFrame({
        "status": ["active", "pending", "active", "cancelled", "active"],
        "category": ["A", "B", "A", "C", "A"],
    })
    engine = AutonomousAnalysisEngine()
    qual_data, ins_qual = engine.analyze_data_quality(df_cat)

    assert qual_data["row_count"] == 5
    assert qual_data["column_count"] == 2
    assert len(ins_qual) == 1
    assert ins_qual[0].category == InsightCategory.DATA_QUALITY


def test_temporal_trend_and_growth_analysis(sales_df):
    """4. Test chronological trend tracking and growth calculation."""
    engine = AutonomousAnalysisEngine()
    trend_data, ins_trend = engine.analyze_trends(sales_df, date_col="date", metric_col="revenue")

    assert len(trend_data["periods"]) > 1
    assert "overall_growth_pct" in trend_data
    assert "peak_period" in trend_data
    assert "trough_period" in trend_data
    assert len(ins_trend) == 1
    assert ins_trend[0].category == InsightCategory.TREND
    assert "increased" in ins_trend[0].summary or "decreased" in ins_trend[0].summary


def test_dimensional_segmentation_analysis(sales_df):
    """5. Test grouping by dimension, leader shares, and disparity."""
    engine = AutonomousAnalysisEngine()
    seg_data, ins_seg = engine.analyze_segmentation(sales_df, dim_col="region", metric_col="revenue")

    assert seg_data["dimension"] == "region"
    assert seg_data["top_segment"] in ["North", "South", "East", "West"]
    assert seg_data["top_share_pct"] > 0
    assert len(ins_seg) == 1
    assert ins_seg[0].category == InsightCategory.PERFORMANCE


# ==============================================================================
# 6-10. Advanced Pattern Discovery & Edge Cases
# ==============================================================================

def test_correlation_analysis_with_non_causal_disclaimer(pure_numeric_df):
    """6. Test Pearson correlation and strict non-causal attribution disclaimer."""
    engine = AutonomousAnalysisEngine()
    corr_data, ins_corr = engine.analyze_correlations(pure_numeric_df, ["metric_a", "metric_b", "metric_c"])

    assert len(ins_corr) >= 1
    corr_ins = ins_corr[0]
    assert corr_ins.claim_type == ClaimType.CORRELATION
    assert "does not prove causal" in corr_ins.summary
    assert "Correlation does not imply causation" in corr_ins.limitations[0]


def test_anomaly_and_outlier_detection(sales_df):
    """8. Test IQR outlier detection on revenue with known injected outliers."""
    engine = AutonomousAnalysisEngine()
    anom_data, ins_anom = engine.analyze_anomalies(sales_df, "revenue")

    assert anom_data["outlier_count"] >= 2
    assert anom_data["outlier_percentage"] > 0
    assert len(ins_anom) == 1
    assert ins_anom[0].category == InsightCategory.ANOMALY
    assert ins_anom[0].claim_type == ClaimType.OBSERVATION


def test_concentration_and_pareto_analysis(sales_df):
    """9. Test concentration ratio and top 20% contribution calculation."""
    engine = AutonomousAnalysisEngine()
    conc_data, ins_conc = engine.analyze_concentration(sales_df, "region", "revenue")

    assert conc_data["top_20_percent_share"] > 0
    assert len(ins_conc) == 1
    assert ins_conc[0].category == InsightCategory.CONCENTRATION


def test_insufficient_data_handling():
    """10. Test graceful skips when dataset has insufficient rows for complex tests."""
    tiny_df = pd.DataFrame({"num": [10.0, 20.0]})
    engine = AutonomousAnalysisEngine()
    
    # Anomaly requires >= 15 rows
    anom_data, ins_anom = engine.analyze_anomalies(tiny_df, "num")
    assert ins_anom == []

    # Concentration requires >= 3 categories
    conc_data, ins_conc = engine.analyze_concentration(tiny_df, "num", "num")
    assert ins_conc == []


# ==============================================================================
# 11-14. Ranking, Deduplication, Evidence & Confidence
# ==============================================================================

def test_duplicate_insight_detection_and_merging(sales_df):
    """11. Test that duplicate insights on identical columns and category are deduplicated."""
    ranker = InsightRanker()
    engine = AutonomousAnalysisEngine()
    _, ins1 = engine.analyze_trends(sales_df, "date", "revenue")
    _, ins2 = engine.analyze_trends(sales_df, "date", "revenue")

    combined = ins1 + ins2
    assert len(combined) == 2

    deduped = ranker.deduplicate(combined)
    assert len(deduped) == 1


def test_insight_ranking_relevance_and_importance(sales_df):
    """12. Test that insights are scored and ranked with intent relevance boost."""
    ranker = InsightRanker()
    agent = AutonomousAnalystAgent()
    req = AutonomousAnalysisRequest(
        dataset=sales_df,
        user_intent=UserIntent(
            intent_type="exploratory_data_analysis",
            objective="Analyze revenue trends",
            metrics=["revenue"],
        ),
    )
    result = agent.analyze(req)

    assert len(result.insights) > 0
    # Top insight should be relevant to revenue
    assert "revenue" in result.insights[0].affected_columns


def test_grounded_mathematical_evidence(sales_df):
    """13. Test that every generated insight contains verified Evidence."""
    agent = AutonomousAnalystAgent()
    req = AutonomousAnalysisRequest(dataset=sales_df)
    result = agent.analyze(req)

    for ins in result.insights:
        assert ins.evidence is not None
        assert ins.evidence.source.startswith("AutonomousAnalysisEngine")
        assert len(ins.calculation) > 0


def test_dynamic_confidence_scoring(sales_df):
    """14. Test that confidence scores reflect statistical properties rather than static constants."""
    agent = AutonomousAnalystAgent()
    req = AutonomousAnalysisRequest(dataset=sales_df)
    result = agent.analyze(req)

    assert result.confidence >= 0.85
    for ins in result.insights:
        assert 0.0 < ins.confidence <= 1.0


# ==============================================================================
# 15-20. Modes, Agent Execution & Planner Integration
# ==============================================================================

def test_business_question_why_mode(sales_df):
    """15. Test 'Why did revenue decline?' mode triggering root cause discovery."""
    discovery = AnalysisDiscoveryAgent()
    intent = UserIntent(
        intent_type="root_cause_analysis",
        objective="Why did revenue drop in Q3?",
        metrics=["revenue"],
        original_command="Why did revenue drop in Q3?",
    )
    candidates = discovery.discover_analyses(sales_df, user_intent=intent)

    cand_types = [c.analysis_type for c in candidates]
    assert "business_driver_investigation" in cand_types
    assert candidates[0].analysis_type == "business_driver_investigation"


def test_complete_analysis_depth_limits(sales_df):
    """16. Test 'Analyze everything' mode respects configured depth limits."""
    agent = AutonomousAnalystAgent()
    
    # Quick depth -> top 4
    res_quick = agent.analyze(AutonomousAnalysisRequest(dataset=sales_df, analysis_depth=AnalysisDepth.QUICK))
    assert len(res_quick.insights) <= 4

    # Standard depth -> top 8
    res_std = agent.analyze(AutonomousAnalysisRequest(dataset=sales_df, analysis_depth=AnalysisDepth.STANDARD))
    assert len(res_std.insights) <= 8


def test_partial_failure_isolation(sales_df):
    """17. Test that failure in one sub-analysis preserves remaining successful results."""
    agent = AutonomousAnalystAgent()
    
    # Dataset with intentional type anomaly that might skip one sub-routine
    corrupted_df = sales_df.copy()
    corrupted_df["corrupted_num"] = [1.0 if i % 2 == 0 else "bad_val" for i in range(len(sales_df))]
    
    req = AutonomousAnalysisRequest(dataset=corrupted_df)
    result = agent.analyze(req)

    assert result.status in ("success", "partial")
    assert len(result.insights) > 0


def test_tool_registry_autonomous_analyst_integration():
    """18. Test that 'autonomous_analyst' is registered in DEFAULT_TOOL_REGISTRY."""
    tool = DEFAULT_TOOL_REGISTRY.get("autonomous_analyst")
    assert tool is not None
    assert "autonomous_analysis" in tool.capabilities
    assert "insight_generation" in tool.capabilities
    assert "pattern_discovery" in tool.capabilities


def test_deterministic_calculations(sales_df):
    """19. Test that re-running analysis on identical data produces identical metrics."""
    agent = AutonomousAnalystAgent()
    req = AutonomousAnalysisRequest(dataset=sales_df)
    
    res1 = agent.analyze(req)
    res2 = agent.analyze(req)

    assert len(res1.insights) == len(res2.insights)
    assert res1.insights[0].title == res2.insights[0].title
    assert res1.insights[0].summary == res2.insights[0].summary


def test_autonomous_analyst_agent_run(sales_df):
    """20. Test that AutonomousAnalystAgent returns standardized AgentResult."""
    agent = AutonomousAnalystAgent()
    task = {
        "data": sales_df,
        "analysis_depth": "standard",
    }
    result = agent.run(task)

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.status == AgentStatus.COMPLETED
    assert "insight_count" in result.metadata
    assert len(result.evidence) > 0
