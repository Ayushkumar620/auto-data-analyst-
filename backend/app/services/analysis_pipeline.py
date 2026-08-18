from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.cleaning.cleaner import DataCleaner
from backend.app.eda.orchestrator import EDAOrchestrator
from backend.app.insights.engine import InsightEngine
from backend.app.profilers.dataset_profiler import DatasetProfiler
from backend.app.visualization.engine import VisualizationEngine
from backend.app.visualization.serializers import figure_to_json


class AnalysisPipeline:
    def __init__(self, base_dir: str | Path = "uploads") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def run_file(self, file_path: str | Path) -> dict[str, Any]:
        source = Path(file_path)
        if not source.exists():
            raise ValueError(f"File not found: {source}")

        dataframe = self._read_dataframe(source)
        dataset_id = f"dataset_{uuid.uuid4().hex[:8]}"
        run_dir = self.base_dir / dataset_id
        run_dir.mkdir(parents=True, exist_ok=True)

        original_path = run_dir / f"original{source.suffix.lower()}"
        shutil.copy2(source, original_path)

        profile = DatasetProfiler().profile(
            dataframe=dataframe,
            filename=source.name,
            file_type=source.suffix.lower().lstrip("."),
            file_size=f"{source.stat().st_size} B",
        )

        cleaner = DataCleaner(dataframe)
        cleaned_df, cleaning_report = cleaner.transform()

        cleaned_path = run_dir / "cleaned.csv"
        cleaned_df.to_csv(cleaned_path, index=False)

        eda = EDAOrchestrator().analyze(cleaned_df)
        visualizations = self._build_visualizations(cleaned_df)
        insight_result = InsightEngine().generate(cleaned_df, eda)

        metadata = {
            "dataset_id": dataset_id,
            "name": source.name,
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
            "file_type": source.suffix.lower().lstrip("."),
            "quality_score": int(profile["profile"]["quality_score"]),
            "missing_values": int(profile["profile"]["missing_values"]),
            "duplicate_rows": int(profile["profile"]["duplicate_rows"]),
            "memory_usage": profile["profile"]["memory_usage"],
            "data_types": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
            "column_names": list(dataframe.columns),
            "preview": dataframe.head(20).to_dict(orient="records"),
        }

        result = {
            "dataset_id": dataset_id,
            "dataset": {
                "id": dataset_id,
                "name": source.name,
                "file_type": source.suffix.lower().lstrip("."),
                "rows": int(dataframe.shape[0]),
                "columns": int(dataframe.shape[1]),
                "created_at": None,
            },
            "metadata": metadata,
            "profile": profile["profile"],
            "cleaning": {
                "status": "success",
                "quality_before": int(cleaner._quality_score(dataframe)),
                "quality_after": int(cleaner._quality_score(cleaned_df)),
                "rows_removed": int(len(dataframe) - len(cleaned_df)),
                "missing_values_fixed": int(cleaning_report.get("missing_values_fixed", 0)),
                "datatype_conversions": int(cleaning_report.get("datatype_conversions", 0)),
                "outliers_detected": int(cleaning_report.get("outliers_detected", 0)),
                "cleaning_report": cleaning_report.get("messages", []),
            },
            "eda": eda,
            "visualizations": visualizations,
            "insights": insight_result,
            "artifacts": {
                "original_path": str(original_path),
                "cleaned_path": str(cleaned_path),
            },
        }
        return result

    def run_upload(self, uploaded_file: Any) -> dict[str, Any]:
        filename = getattr(uploaded_file, "filename", None) or getattr(uploaded_file, "name", "")
        if not filename:
            raise ValueError("Please upload a data file.")

        file_path = self.base_dir / filename

        if hasattr(uploaded_file, "file"):
            source_file = uploaded_file.file
            source_file.seek(0)
            content = source_file.read()
            source_file.seek(0)
        elif hasattr(uploaded_file, "stream"):
            stream = uploaded_file.stream
            try:
                stream.seek(0)
            except Exception:
                pass
            content = stream.read()
        elif isinstance(uploaded_file, (bytes, bytearray)):
            content = uploaded_file
        else:
            content = uploaded_file.read()

        with open(file_path, "wb") as file_handle:
            file_handle.write(content)

        return self.run_file(file_path)

    def _read_dataframe(self, file_path: Path) -> pd.DataFrame:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(file_path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)
        raise ValueError("Unsupported file type. Only CSV and Excel files are allowed.")

    def _build_visualizations(self, dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        visuals: list[dict[str, Any]] = []
        engine = VisualizationEngine()
        for index, recommendation in enumerate(engine.recommend(dataframe), start=1):
            payload = engine.generate(dataframe, recommendation, f"chart_{index:03d}")
            chart = payload["figure"]
            visuals.append({
                "id": payload["id"],
                "type": payload["type"],
                "title": payload["title"],
                "x_column": payload["x_column"],
                "y_column": payload["y_column"],
                "data": chart.get("data", []),
                "layout": chart.get("layout", {}),
            })
        return visuals
