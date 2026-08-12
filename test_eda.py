import pandas as pd

from backend.app.eda.orchestrator import EDAOrchestrator


def test_eda_orchestrator_returns_sections():
    dataframe = pd.DataFrame(
        {
            "age": [25, 30, 35, 40],
            "income": [30000, 40000, 50000, 60000],
            "segment": ["A", "B", "A", "C"],
            "signup_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
        }
    )

    result = EDAOrchestrator().analyze(dataframe)

    assert result["summary"]["row_count"] == 4
    assert result["statistics"]["numeric"]["age"]["mean"] == 32.5
    assert result["correlations"]
    assert result["categorical"]["summary"]
    assert result["recommended_charts"]
