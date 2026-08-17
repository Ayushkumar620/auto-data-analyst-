from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.app.services.analysis_pipeline import AnalysisPipeline


def _write_test_csv(path: Path) -> pd.DataFrame:
    dataframe = pd.DataFrame(
        {
            "sales": [100, 200, None, 250],
            "region": ["North", "South", "North", "West"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        }
    )
    dataframe.to_csv(path, index=False)
    return dataframe


def test_analysis_pipeline_generates_dataset_metadata(tmp_path):
    file_path = tmp_path / "sales.csv"
    _write_test_csv(file_path)

    result = AnalysisPipeline(base_dir=str(tmp_path)).run_file(file_path)

    assert result["dataset_id"]
    assert result["metadata"]["rows"] == 4
    assert result["metadata"]["columns"] == 3
    assert result["metadata"]["quality_score"] >= 0
    assert result["profile"]["quality_score"] >= 0


def test_analysis_pipeline_preserves_original_and_keeps_cleaned_copy_separate(tmp_path):
    file_path = tmp_path / "sales.csv"
    _write_test_csv(file_path)

    result = AnalysisPipeline(base_dir=str(tmp_path)).run_file(file_path)

    assert Path(result["artifacts"]["original_path"]).exists()
    assert Path(result["artifacts"]["cleaned_path"]).exists()
    assert result["artifacts"]["original_path"] != result["artifacts"]["cleaned_path"]


def test_analysis_pipeline_returns_eda_visualizations_and_insights(tmp_path):
    file_path = tmp_path / "sales.csv"
    _write_test_csv(file_path)

    result = AnalysisPipeline(base_dir=str(tmp_path)).run_file(file_path)

    assert result["eda"]["summary"]["row_count"] == 4
    assert result["visualizations"]
    assert result["visualizations"][0]["data"]
    assert result["insights"]["insights"]
