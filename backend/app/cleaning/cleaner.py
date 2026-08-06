from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from backend.app.cleaning.duplicates import DuplicateCleaner
from backend.app.cleaning.missing import MissingValueCleaner
from backend.app.cleaning.datatypes import DataTypeCleaner
from backend.app.cleaning.outliers import OutlierDetector
from backend.app.cleaning.pipeline import CleaningPipeline


class DataCleaner:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.copy()

    def clean(self) -> Dict[str, Any]:
        pipeline = CleaningPipeline(
            step1=self._standardize_columns,
            step2=self._handle_missing_values,
            step3=self._remove_duplicates,
            step4=self._correct_dtypes,
            step5=self._detect_outliers,
        )
        cleaned_df, report = pipeline.run(self.dataframe)
        quality_before = self._quality_score(self.dataframe)
        quality_after = self._quality_score(cleaned_df)
        return {
            "status": "success",
            "quality_before": quality_before,
            "quality_after": quality_after,
            "rows_removed": int(len(self.dataframe) - len(cleaned_df)),
            "missing_values_fixed": report.get("missing_values_fixed", 0),
            "datatype_conversions": report.get("datatype_conversions", 0),
            "outliers_detected": report.get("outliers_detected", 0),
            "cleaning_report": report.get("messages", []),
            "cleaned_data": cleaned_df.head(20).to_dict(orient="records"),
        }

    def _standardize_columns(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        new_columns = []
        for column in dataframe.columns:
            normalized = column.strip().lower().replace(" ", "_")
            normalized = "".join(ch for ch in normalized if ch.isalnum() or ch == "_")
            new_columns.append(normalized)
        dataframe = dataframe.copy()
        dataframe.columns = new_columns
        return dataframe, [f"Standardized {len(new_columns)} column names"]

    def _handle_missing_values(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        return MissingValueCleaner().clean(dataframe)

    def _remove_duplicates(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        return DuplicateCleaner().clean(dataframe)

    def _correct_dtypes(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        return DataTypeCleaner().clean(dataframe)

    def _detect_outliers(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        return OutlierDetector().clean(dataframe)

    def _quality_score(self, dataframe: pd.DataFrame) -> int:
        missing_ratio = dataframe.isna().mean().mean() * 100
        duplicate_ratio = dataframe.duplicated().mean() * 100
        score = 100 - int(missing_ratio + duplicate_ratio)
        return max(0, min(100, score))
