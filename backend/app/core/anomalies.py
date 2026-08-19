"""Universal Anomaly Detection Engine.

Single source of truth for anomaly detection across UI, reports, chat,
insights and agents.  Supports multiple methods:

  - iqr              (inter-quartile range fences)
  - zscore           (classic z-score threshold)
  - modified_zscore  (MAD-based robust z-score)
  - isolation_forest (sklearn, for larger datasets)
  - lof              (local outlier factor, multivariate)
  - time_series      (rolling median + MAD on a temporal sequence)

The engine automatically selects a method when method='auto', based on
dataset size, distribution and whether temporal structure exists.

Every result contains: method, parameters, outlier_count,
outlier_indices, statistics, confidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.temporal import TemporalIntelligenceEngine


class AnomalyDetectionEngine:
    """Centralized, method-aware anomaly detection."""

    METHODS = ("iqr", "zscore", "modified_zscore", "isolation_forest", "lof", "time_series")

    def __init__(self) -> None:
        self.temporal = TemporalIntelligenceEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, dataframe: pd.DataFrame, columns: list[str] | None = None,
               method: str = "auto", parameters: dict[str, Any] | None = None,
               date_column: str | None = None) -> dict[str, Any]:
        """Detect anomalies across numeric columns.

        Returns a dict keyed by column with per-column results plus a summary.
        """
        method = method or "auto"
        if method != "auto" and method not in self.METHODS:
            raise ValueError(f"Unknown anomaly method '{method}'. "
                             f"Supported methods: {', '.join(self.METHODS)} or 'auto'.")

        selected = columns or list(dataframe.select_dtypes(include="number").columns)
        selected = [column for column in selected if column in dataframe.columns]

        if method == "time_series" or (method == "auto" and date_column):
            if not date_column:
                field = self.temporal.primary_time_field(dataframe)
                date_column = field["column"] if field else None
            if date_column and date_column in dataframe.columns:
                return self._time_series_detect(dataframe, date_column, selected, parameters or {})
            if method == "time_series":
                return {"summary": {"total_outliers": 0, "columns_with_outliers": [],
                                    "reason": "No temporal field found"}}

        per_column: dict[str, Any] = {}
        total = 0
        for column in selected:
            series = pd.to_numeric(dataframe[column], errors="coerce").dropna()
            if len(series) < 3:
                continue
            chosen = self._select_method(series, method, parameters or {})
            result = self._detect_column(series, column, chosen, parameters or {})
            if result["outlier_count"] > 0:
                total += result["outlier_count"]
            per_column[column] = result

        return {
            "method": method,
            "columns": per_column,
            "summary": {
                "total_outliers": int(total),
                "columns_with_outliers": [
                    column for column, result in per_column.items()
                    if result["outlier_count"] > 0
                ],
            },
        }

    # ------------------------------------------------------------------
    # Method selection
    # ------------------------------------------------------------------
    def _select_method(self, series: pd.Series, method: str,
                       parameters: dict[str, Any]) -> str:
        if method != "auto":
            return method
        n = len(series)
        skew = float(series.skew()) if n > 2 and not pd.isna(series.skew()) else 0.0
        if n >= 200 and parameters.get("allow_multivariate", False):
            return "isolation_forest"
        if abs(skew) > 1.5:
            return "modified_zscore"
        if n < 20:
            return "iqr"
        return "modified_zscore"

    def _detect_column(self, series: pd.Series, column: str, method: str,
                       parameters: dict[str, Any]) -> dict[str, Any]:
        values = series.to_numpy(dtype=float)
        if method == "iqr":
            return self._iqr(values, column, parameters)
        if method == "zscore":
            return self._zscore(values, column, parameters)
        if method == "modified_zscore":
            return self._modified_zscore(values, column, parameters)
        if method == "isolation_forest":
            return self._isolation_forest(values, column, parameters)
        if method == "lof":
            return self._lof(values, column, parameters)
        raise ValueError(f"Unsupported method '{method}' for univariate detection.")

    # ------------------------------------------------------------------
    # Individual methods
    # ------------------------------------------------------------------
    def _iqr(self, values: np.ndarray, column: str,
             parameters: dict[str, Any]) -> dict[str, Any]:
        factor = float(parameters.get("iqr_factor", 1.5))
        q1 = float(np.quantile(values, 0.25))
        q3 = float(np.quantile(values, 0.75))
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        mask = (values < lower) | (values > upper)
        indices = np.flatnonzero(mask)
        return {
            "column": column, "method": "iqr",
            "parameters": {"iqr_factor": factor},
            "outlier_count": int(len(indices)),
            "outlier_indices": indices.astype(int).tolist(),
            "statistics": {"q1": round(q1, 6), "q3": round(q3, 6),
                           "iqr": round(iqr, 6), "lower_bound": round(lower, 6),
                           "upper_bound": round(upper, 6),
                           "mean": round(float(np.mean(values)), 6),
                           "std": round(float(np.std(values)), 6)},
            "confidence": round(self._scale_confidence(len(indices), len(values)), 3),
        }

    def _zscore(self, values: np.ndarray, column: str,
                parameters: dict[str, Any]) -> dict[str, Any]:
        threshold = float(parameters.get("zscore_threshold", 3.0))
        mean, std = float(np.mean(values)), float(np.std(values))
        if std == 0:
            return self._empty(column, "zscore", {"zscore_threshold": threshold},
                               {"mean": mean, "std": 0.0}, 0.9)
        scores = np.abs((values - mean) / std)
        mask = scores > threshold
        indices = np.flatnonzero(mask)
        return {
            "column": column, "method": "zscore",
            "parameters": {"zscore_threshold": threshold},
            "outlier_count": int(len(indices)),
            "outlier_indices": indices.astype(int).tolist(),
            "statistics": {"mean": round(mean, 6), "std": round(std, 6),
                           "threshold": threshold,
                           "max_abs_zscore": round(float(scores.max()), 6)},
            "confidence": round(self._scale_confidence(len(indices), len(values)), 3),
        }

    def _modified_zscore(self, values: np.ndarray, column: str,
                         parameters: dict[str, Any]) -> dict[str, Any]:
        threshold = float(parameters.get("modified_zscore_threshold", 3.5))
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad == 0:
            mad = float(np.std(values)) or 1e-12
        modified = 0.6745 * (values - median) / mad
        mask = np.abs(modified) > threshold
        indices = np.flatnonzero(mask)
        return {
            "column": column, "method": "modified_zscore",
            "parameters": {"modified_zscore_threshold": threshold},
            "outlier_count": int(len(indices)),
            "outlier_indices": indices.astype(int).tolist(),
            "statistics": {"median": round(median, 6), "mad": round(float(mad), 6),
                           "threshold": threshold,
                           "max_abs_modified_zscore": round(float(np.abs(modified).max()), 6)},
            "confidence": round(self._scale_confidence(len(indices), len(values)), 3),
        }

    def _isolation_forest(self, values: np.ndarray, column: str,
                          parameters: dict[str, Any]) -> dict[str, Any]:
        from sklearn.ensemble import IsolationForest
        n = len(values)
        contamination = float(parameters.get("contamination", 0.1))
        model = IsolationForest(contamination=min(contamination, 0.5), random_state=42)
        model.fit(values.reshape(-1, 1))
        mask = model.predict(values.reshape(-1, 1)) == -1
        indices = np.flatnonzero(mask)
        return {
            "column": column, "method": "isolation_forest",
            "parameters": {"contamination": contamination},
            "outlier_count": int(len(indices)),
            "outlier_indices": indices.astype(int).tolist(),
            "statistics": {"n": n,
                           "mean": round(float(np.mean(values)), 6),
                           "std": round(float(np.std(values)), 6)},
            "confidence": round(self._scale_confidence(len(indices), n), 3),
        }

    def _lof(self, values: np.ndarray, column: str,
             parameters: dict[str, Any]) -> dict[str, Any]:
        from sklearn.neighbors import LocalOutlierFactor
        n = len(values)
        if n < 10:
            return self._iqr(values, column, {"iqr_factor": 1.5})
        contamination = float(parameters.get("contamination", 0.1))
        n_neighbors = int(parameters.get("n_neighbors", min(20, max(2, n // 2))))
        model = LocalOutlierFactor(n_neighbors=n_neighbors,
                                   contamination=min(contamination, 0.5))
        mask = model.fit_predict(values.reshape(-1, 1)) == -1
        indices = np.flatnonzero(mask)
        return {
            "column": column, "method": "lof",
            "parameters": {"contamination": contamination, "n_neighbors": n_neighbors},
            "outlier_count": int(len(indices)),
            "outlier_indices": indices.astype(int).tolist(),
            "statistics": {"n": n,
                           "negative_outlier_factor_min": round(float(model.negative_outlier_factor_.min()), 6)},
            "confidence": round(self._scale_confidence(len(indices), n), 3),
        }

    def _time_series_detect(self, dataframe: pd.DataFrame, date_column: str,
                            columns: list[str], parameters: dict[str, Any]) -> dict[str, Any]:
        window = int(parameters.get("rolling_window", 5))
        threshold = float(parameters.get("ts_threshold", 3.0))
        parsed = pd.to_datetime(dataframe[date_column], errors="coerce", format="mixed")
        per_column: dict[str, Any] = {}
        total = 0
        for column in columns:
            numeric = pd.to_numeric(dataframe[column], errors="coerce")
            data = pd.DataFrame({date_column: parsed, column: numeric}).dropna()
            if len(data) < window + 2:
                continue
            data = data.sort_values(date_column)
            values = data[column]
            rolling_median = values.rolling(window, center=True, min_periods=1).median()
            rolling_mad = (values - rolling_median).abs().rolling(
                window, center=True, min_periods=1).median()
            spread = rolling_mad.replace(0, np.nan)
            modified = 0.6745 * (values - rolling_median) / spread
            mask = modified.abs() > threshold
            mask = mask.fillna(False)
            indices = data.index[mask].tolist()
            total += len(indices)
            per_column[column] = {
                "column": column, "method": "time_series",
                "parameters": {"rolling_window": window, "ts_threshold": threshold,
                               "date_column": date_column},
                "outlier_count": int(len(indices)),
                "outlier_indices": [int(index) for index in indices],
                "statistics": {"n": int(len(data)),
                               "mean": round(float(values.mean()), 6),
                               "std": round(float(values.std()), 6),
                               "date_column": date_column},
                "confidence": round(self._scale_confidence(len(indices), len(data)), 3),
            }
        return {
            "method": "time_series",
            "columns": per_column,
            "summary": {
                "total_outliers": int(total),
                "columns_with_outliers": [column for column, result in per_column.items()
                                          if result["outlier_count"] > 0],
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _empty(column: str, method: str, parameters: dict[str, Any],
               statistics: dict[str, Any], confidence: float) -> dict[str, Any]:
        return {
            "column": column, "method": method,
            "parameters": parameters,
            "outlier_count": 0, "outlier_indices": [],
            "statistics": statistics, "confidence": round(confidence, 3),
        }

    @staticmethod
    def _scale_confidence(outlier_count: int, n: int) -> float:
        if n == 0:
            return 0.5
        ratio = outlier_count / n
        if ratio == 0:
            return 0.6
        if ratio > 0.5:
            return 0.4  # More than half flagged -> suspicious method fit
        return min(0.98, 0.6 + 0.15 * min(ratio * 10, 2.0))