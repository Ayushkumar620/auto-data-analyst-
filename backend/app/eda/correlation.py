from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


class CorrelationAnalyzer:
    def analyze(self, dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
        numeric_columns = [column for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column])]
        correlations: List[Dict[str, Any]] = []
        if len(numeric_columns) < 2:
            return correlations

        corr_matrix = dataframe[numeric_columns].corr(numeric_only=True)
        for i, left in enumerate(numeric_columns):
            for right in numeric_columns[i + 1:]:
                value = corr_matrix.loc[left, right]
                if pd.isna(value):
                    continue
                strength = self._classify_strength(value)
                correlations.append({
                    "left": left,
                    "right": right,
                    "correlation": round(float(value), 4),
                    "interpretation": strength,
                })
        return correlations

    def _classify_strength(self, value: float) -> str:
        if abs(value) >= 0.8:
            return "strong positive" if value >= 0 else "strong negative"
        if abs(value) >= 0.5:
            return "moderate positive" if value >= 0 else "moderate negative"
        if abs(value) >= 0.2:
            return "weak positive" if value >= 0 else "weak negative"
        return "very weak or no clear relationship"
