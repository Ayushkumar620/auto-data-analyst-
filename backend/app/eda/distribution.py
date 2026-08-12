from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


class DistributionAnalyzer:
    def analyze(self, dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for column in dataframe.select_dtypes(include=["number"]).columns:
            series = dataframe[column]
            if series.empty:
                continue
            skewness = float(series.skew()) if not pd.isna(series.skew()) else None
            iqr = series.quantile(0.75) - series.quantile(0.25)
            findings.append({
                "column": column,
                "distribution": self._classify_distribution(skewness),
                "skewness": round(skewness, 4) if skewness is not None else None,
                "outlier_candidates": int(((series < (series.quantile(0.25) - 1.5 * iqr)) | (series > (series.quantile(0.75) + 1.5 * iqr))).sum()),
                "value_concentration": self._value_concentration(series),
            })
        return findings

    def _classify_distribution(self, skewness: float | None) -> str:
        if skewness is None:
            return "unknown"
        if abs(skewness) < 0.5:
            return "roughly normal"
        if skewness > 0:
            return "right-skewed"
        return "left-skewed"

    def _value_concentration(self, series: pd.Series) -> str:
        top_share = series.value_counts(normalize=True).head(1).iloc[0] if not series.empty else 0
        if top_share > 0.5:
            return "high concentration"
        if top_share > 0.2:
            return "moderate concentration"
        return "broad spread"
