"""Tests for PHASE H: evidence-backed AI insight engine."""
import pandas as pd

from backend.app.insights import InsightEngine
from backend.app.insights.interpreter import (
    InsightInterpreter,
    _extract_json_object,
    _looks_like_new_facts,
)
from backend.app.insights.rules import InsightRules


def _sample_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
        "region": ["North", "North", "North", "South"],
        "revenue": [100, 120, 150, 180],
        "profit": [10, 12, 15, 18],
    })


def test_insight_engine_returns_evidence_backed_structured_insights():
    result = InsightEngine().generate(_sample_df())
    assert result["facts"]["growth"][0]["growth_percentage"] == 80.0
    assert any(item["type"] == "trend" for item in result["insights"])
    for item in result["insights"]:
        assert {"type", "title", "description", "severity", "confidence", "evidence"} <= item.keys()
        # Every insight must be evidence-backed (each carries evidence dict)
        assert isinstance(item["evidence"], dict)


def test_high_missing_values_create_data_quality_risk():
    dataframe = pd.DataFrame({"value": [1, None, None, None, None]})
    insights = InsightEngine().generate(dataframe)["insights"]
    assert any(item["title"] == "Data Quality Warning" for item in insights)


def test_synthesize_covers_all_categories():
    result = InsightEngine().synthesize(_sample_df())
    categories = result["categories"]
    expected = {"key_finding", "trend", "anomaly", "risk", "opportunity", "recommendation"}
    assert set(categories.keys()) == expected
    for category, items in categories.items():
        # Every category must have at least one entry (rules or explicit "none detected")
        assert len(items) >= 1, f"{category} should have at least one entry"


def test_recommendations_derived_from_facts():
    rules = InsightRules()
    facts = {
        "missing_percentage": 25.0,
        "growth": [{"column": "revenue", "growth_percentage": -50.0}],
        "correlations": [{"left": "revenue", "right": "profit", "correlation": 0.9}],
        "anomalies": [{"column": "revenue", "anomaly_count": 2}],
        "category_shares": [],
    }
    recommendations = rules.recommendations(facts)
    assert any("missing" in r.lower() for r in recommendations)
    assert any("revenue" in r.lower() for r in recommendations)
    assert any("unusual" in r.lower() for r in recommendations)


def test_llm_without_key_falls_back_to_deterministic():
    # With no API key configured, the interpreter must fall back safely
    interpreter = InsightInterpreter()
    assert interpreter.enabled is False
    insights = rules_insights()
    enriched = interpreter.enrich("dataset", {"missing_percentage": 0.0}, insights)
    # Fallback keeps rule-derived insights and marks source as rule
    assert len(enriched) == len(insights)
    assert all(i.source == "rule" for i in enriched)


def rules_insights():
    from backend.app.insights.schemas import Insight
    return [Insight(
        type="trend", title="Revenue Growth",
        description="Revenue increased by 80% during the analyzed period.",
        evidence={"metric": "revenue_growth", "value": 80.0},
        source="rule",
    )]


def test_extract_json_object_handles_fences():
    content = '```json\n{"Revenue Growth": "Sales rose over the period."}\n```'
    parsed = _extract_json_object(content)
    assert parsed["Revenue Growth"] == "Sales rose over the period."


def test_looks_like_new_facts_guards_unverified_figures():
    assert _looks_like_new_facts("Revenue increased by 45% this year.")
    assert not _looks_like_new_facts("Revenue showed an upward trend over the analyzed period.")