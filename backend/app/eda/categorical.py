from __future__ import annotations

from typing import Dict, List

import pandas as pd


class CategoricalAnalyzer:
    def analyze(self, dataframe: pd.DataFrame) -> Dict[str, List[Dict[str, object]]]:
        results: Dict[str, List[Dict[str, object]]] = {"summary": []}
        for column in dataframe.select_dtypes(exclude=["number"]).columns:
            series = dataframe[column].dropna()
            if series.empty:
                continue
            counts = series.value_counts()
            results["summary"].append({
                "column": column,
                "category_count": int(counts.shape[0]),
                "most_frequent": counts.idxmax(),
                "frequency": int(counts.iloc[0]),
            })
        return results
