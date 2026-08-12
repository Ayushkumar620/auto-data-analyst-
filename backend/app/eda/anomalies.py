from __future__ import annotations

from typing import Dict, List

import pandas as pd


class AnomalyDetector:
    def analyze(self, dataframe: pd.DataFrame) -> Dict[str, List[Dict[str, object]]]:
        results: List[Dict[str, object]] = []
        for column in dataframe.select_dtypes(include=["number"]).columns:
            series = dataframe[column]
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            anomalies = series[(series < lower) | (series > upper)]
            if not anomalies.empty:
                results.append({
                    "column": column,
                    "anomaly_count": int(anomalies.shape[0]),
                    "values": anomalies.head(5).tolist(),
                })
        return {"anomalies": results}
