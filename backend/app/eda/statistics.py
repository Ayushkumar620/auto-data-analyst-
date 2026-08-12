from __future__ import annotations

from typing import Any, Dict

import pandas as pd


class StatisticsAnalyzer:
    def analyze(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        numeric_columns = [column for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column])]
        categorical_columns = [column for column in dataframe.columns if column not in numeric_columns and not pd.api.types.is_datetime64_any_dtype(dataframe[column])]
        date_columns = [column for column in dataframe.columns if pd.api.types.is_datetime64_any_dtype(dataframe[column])]

        stats: Dict[str, Any] = {"numeric": {}, "categorical": {}, "date": {}}
        for column in numeric_columns:
            series = dataframe[column]
            stats["numeric"][column] = {
                "count": int(series.count()),
                "mean": round(float(series.mean()), 4) if not pd.isna(series.mean()) else None,
                "median": round(float(series.median()), 4) if not pd.isna(series.median()) else None,
                "mode": self._safe_value(series.mode(dropna=True).iloc[0]) if not series.mode(dropna=True).empty else None,
                "std": round(float(series.std()), 4) if not pd.isna(series.std()) else None,
                "variance": round(float(series.var()), 4) if not pd.isna(series.var()) else None,
                "min": round(float(series.min()), 4) if not pd.isna(series.min()) else None,
                "max": round(float(series.max()), 4) if not pd.isna(series.max()) else None,
                "quartiles": [round(float(value), 4) for value in series.quantile([0.25, 0.5, 0.75]).tolist()] if not series.empty else [],
            }

        for column in categorical_columns:
            series = dataframe[column].dropna()
            counts = series.value_counts()
            stats["categorical"][column] = {
                "categories": int(counts.shape[0]),
                "most_frequent": self._safe_value(counts.idxmax()) if not counts.empty else None,
                "frequency_table": {self._safe_value(key): int(value) for key, value in counts.head(5).items()},
                "distribution": {self._safe_value(key): round(float(value / len(series) * 100), 2) for key, value in counts.head(5).items()},
            }

        for column in date_columns:
            series = dataframe[column].dropna()
            stats["date"][column] = {
                "earliest": series.min().isoformat() if not series.empty else None,
                "latest": series.max().isoformat() if not series.empty else None,
                "timespan": None,
                "missing_dates": int(dataframe[column].isna().sum()),
            }

        return stats

    def _safe_value(self, value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value
