"""
Time-Series Detection and Suitability Engine.

Inspects DataFrame columns, infers chronological frequency, assesses interval regularity,
detects trend and seasonality patterns, and computes deterministic suitability scores.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from agent.forecasting_schemas import ForecastSuitabilityResult


class TimeSeriesDetector:
    """
    Evaluates dataset readiness for time-series forecasting.
    """

    DATE_PATTERNS = [
        re.compile(r"^date$", re.I),
        re.compile(r"^timestamp$", re.I),
        re.compile(r"^time$", re.I),
        re.compile(r"^ds$", re.I),
        re.compile(r"^datetime$", re.I),
        re.compile(r"(date|time|year|month|day|quarter|period)", re.I),
    ]

    def detect_time_column(self, df: pd.DataFrame, hint: Optional[str] = None) -> Optional[str]:
        """Identify primary chronological column."""
        if hint and hint in df.columns:
            return hint

        # 1. Exact datetime dtypes
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        # 2. String/object columns matching date names
        for pattern in self.DATE_PATTERNS:
            for col in df.columns:
                if pattern.search(col):
                    # Verify parseability on sample
                    try:
                        sample = df[col].dropna().head(10)
                        if not sample.empty:
                            pd.to_datetime(sample)
                            return col
                    except Exception:
                        pass

        # 3. Fallback: inspect any object column
        for col in df.select_dtypes(include=["object", "string"]).columns:
            try:
                sample = df[col].dropna().head(10)
                if len(sample) >= 3:
                    pd.to_datetime(sample)
                    return col
            except Exception:
                pass

        return None

    def detect_target_column(self, df: pd.DataFrame, time_col: Optional[str] = None, hint: Optional[str] = None) -> Optional[str]:
        """Identify primary numeric forecasting target."""
        if hint and hint in df.columns and pd.api.types.is_numeric_dtype(df[hint]):
            return hint

        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != time_col]
        if not num_cols:
            return None

        # Prefer common metric keywords
        metric_keywords = ["sales", "revenue", "demand", "units", "quantity", "price", "profit", "value", "count"]
        for kw in metric_keywords:
            for c in num_cols:
                if kw in c.lower():
                    return c

        return num_cols[0]

    def infer_frequency(self, series: pd.Series) -> Tuple[str, bool]:
        """
        Infer frequency string ("D", "W", "M", "Q", "Y", "IRREGULAR") and regularity flag.
        """
        try:
            dates = pd.to_datetime(series.dropna()).sort_values()
            if len(dates) < 3:
                return "IRREGULAR", False

            diffs = dates.diff().dropna()
            median_days = diffs.dt.total_seconds().median() / 86400.0

            if 0.8 <= median_days <= 1.2:
                return "D", True
            elif 6.0 <= median_days <= 8.0:
                return "W", True
            elif 27.0 <= median_days <= 32.0:
                return "M", True
            elif 88.0 <= median_days <= 93.0:
                return "Q", True
            elif 360.0 <= median_days <= 370.0:
                return "Y", True
            else:
                return "IRREGULAR", False
        except Exception:
            return "IRREGULAR", False

    def assess_suitability(
        self,
        df: pd.DataFrame,
        time_column: Optional[str] = None,
        target_column: Optional[str] = None,
        horizon: int = 6,
    ) -> ForecastSuitabilityResult:
        """Complete evaluation of forecasting readiness."""
        reasons: List[str] = []
        warnings: List[str] = []
        limitations: List[str] = []

        time_col = self.detect_time_column(df, hint=time_column)
        target_col = self.detect_target_column(df, time_col=time_col, hint=target_column)

        if not time_col:
            return ForecastSuitabilityResult(
                suitable=False,
                score=0.0,
                reasons=["No datetime or timestamp column found in dataset."],
                warnings=["Forecasting requires a valid chronological dimension."],
                limitations=["Longitudinal forecasting is unsupported on static cross-sectional datasets."],
            )

        if not target_col:
            return ForecastSuitabilityResult(
                suitable=False,
                score=0.0,
                detected_time_column=time_col,
                reasons=["No numeric target metric found in dataset."],
                warnings=["Forecasting requires at least one continuous numeric metric."],
                limitations=["Cannot forecast non-numeric categorical variables."],
            )

        # Ingest and prepare series
        clean_df = df[[time_col, target_col]].dropna().copy()
        clean_df[time_col] = pd.to_datetime(clean_df[time_col])
        clean_df = clean_df.sort_values(time_col)
        n_obs = len(clean_df)

        freq_str, is_regular = self.infer_frequency(clean_df[time_col])
        time_range = {
            "start": str(clean_df[time_col].iloc[0]),
            "end": str(clean_df[time_col].iloc[-1]),
        }

        # Check sample size constraints
        if n_obs < 5:
            return ForecastSuitabilityResult(
                suitable=False,
                score=0.10,
                detected_time_column=time_col,
                detected_target=target_col,
                detected_frequency=freq_str,
                observation_count=n_obs,
                time_range=time_range,
                reasons=[f"Insufficient historical observations (N={n_obs} < 5)."],
                warnings=["Minimum 5 historical data points required for baseline forecasting."],
                limitations=["Insufficient sample size prevents chronological backtesting."],
            )

        # Assess Trend and Seasonality
        has_trend = False
        has_seasonality = False
        score = 0.70

        if n_obs >= 8:
            y = clean_df[target_col].to_numpy(dtype=float)
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            corr = np.corrcoef(x, y)[0, 1]
            if abs(corr) > 0.4:
                has_trend = True
                reasons.append(f"Significant temporal trend detected ({'upward' if slope > 0 else 'downward'}, r={corr:.2f}).")
                score += 0.10

            # Seasonality check
            if n_obs >= 12 and freq_str in ("M", "Q", "D", "W"):
                lag = 12 if freq_str == "M" else (4 if freq_str == "Q" else 7)
                if n_obs >= lag * 2:
                    autocorr = pd.Series(y).autocorr(lag=lag)
                    if not np.isnan(autocorr) and autocorr > 0.35:
                        has_seasonality = True
                        reasons.append(f"Seasonal cycle detected at lag {lag} (autocorr={autocorr:.2f}).")
                        score += 0.10

        if not is_regular:
            warnings.append("Irregular observation intervals detected; series will be resampled.")
            score -= 0.10

        if n_obs < 15:
            warnings.append(f"Modest historical sample size (N={n_obs}); short-range horizons recommended.")
            limitations.append("Prediction intervals will expand rapidly beyond near-term periods.")

        reasons.append(f"Time series ready: {n_obs} observations with {freq_str} frequency.")
        score = min(1.0, max(0.2, score))

        return ForecastSuitabilityResult(
            suitable=True,
            score=score,
            detected_time_column=time_col,
            detected_target=target_col,
            detected_frequency=freq_str,
            observation_count=n_obs,
            time_range=time_range,
            has_seasonality=has_seasonality,
            has_trend=has_trend,
            reasons=reasons,
            warnings=warnings,
            limitations=limitations,
        )
