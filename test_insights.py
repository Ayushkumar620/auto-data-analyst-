import pandas as pd

from backend.app.insights import InsightEngine


def test_insight_engine_returns_evidence_backed_structured_insights():
    dataframe = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
        "region": ["North", "North", "North", "South"],
        "revenue": [100, 120, 150, 180],
        "profit": [10, 12, 15, 18],
    })
    result = InsightEngine().generate(dataframe)
    assert result["facts"]["growth"][0]["growth_percentage"] == 80.0
    assert any(item["type"] == "trend" for item in result["insights"])
    assert all({"type", "title", "description", "severity", "confidence", "evidence"} <= item.keys()
               for item in result["insights"])


def test_high_missing_values_create_data_quality_risk():
    dataframe = pd.DataFrame({"value": [1, None, None, None, None]})
    insights = InsightEngine().generate(dataframe)["insights"]
    assert any(item["title"] == "Data Quality Warning" for item in insights)
