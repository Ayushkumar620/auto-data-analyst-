from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


class SummaryAnalyzer:
    def analyze(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        numeric_columns = [column for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column])]
        categorical_columns = [column for column in dataframe.columns if column not in numeric_columns and not pd.api.types.is_datetime64_any_dtype(dataframe[column])]
        date_columns = [column for column in dataframe.columns if pd.api.types.is_datetime64_any_dtype(dataframe[column])]

        return {
            "row_count": int(dataframe.shape[0]),
            "column_count": int(dataframe.shape[1]),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "date_columns": date_columns,
            "missing_values": int(dataframe.isna().sum().sum()),
            "duplicate_rows": int(dataframe.duplicated().sum()),
        }
