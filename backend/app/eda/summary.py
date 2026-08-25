from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


class SummaryAnalyzer:
    def analyze(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        date_columns = []
        for column in dataframe.columns:
            if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
                date_columns.append(column)
            elif pd.api.types.is_numeric_dtype(dataframe[column]):
                s = dataframe[column].dropna()
                c_low = column.lower()
                if not s.empty and s.between(1800, 2150).all() and any(t in c_low for t in ("year", "fy", "cy", "yr", "date", "period")):
                    date_columns.append(column)
            elif pd.api.types.is_string_dtype(dataframe[column]) or pd.api.types.is_object_dtype(dataframe[column]):
                s = dataframe[column].dropna().head(10)
                if len(s) >= 3 and any(t in column.lower() for t in ("date", "time", "month", "year", "quarter", "period")):
                    if pd.to_datetime(s, errors="coerce").notna().mean() >= 0.7:
                        date_columns.append(column)

        numeric_columns = [column for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column]) and column not in date_columns]
        categorical_columns = [column for column in dataframe.columns if column not in numeric_columns and column not in date_columns]

        missing_count = int(dataframe.isna().sum().sum())
        total_cells = max(1, dataframe.shape[0] * dataframe.shape[1])

        return {
            "row_count": int(dataframe.shape[0]),
            "column_count": int(dataframe.shape[1]),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "date_columns": date_columns,
            "temporal_columns": date_columns,
            "missing_values": missing_count,
            "missing_count": missing_count,
            "missing_percentage": round((missing_count / total_cells) * 100, 2),
            "duplicate_rows": int(dataframe.duplicated().sum()),
        }
