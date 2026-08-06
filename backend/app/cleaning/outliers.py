from __future__ import annotations

from typing import List, Tuple

import pandas as pd


class OutlierDetector:
    def clean(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        outlier_count = 0
        for column in dataframe.select_dtypes(include=['number']).columns:
            series = dataframe[column]
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count += int(((series < lower) | (series > upper)).sum())
        if outlier_count:
            return dataframe, [f"Flagged {outlier_count} potential outliers"]
        return dataframe, ["No outliers detected"]
