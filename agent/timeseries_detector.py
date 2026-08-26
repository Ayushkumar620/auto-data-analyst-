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

        # 2. String/object or integer columns matching date/time/year names
        for pattern in self.DATE_PATTERNS:
            for col in df.columns:
                if pattern.search(col):
                    series = df[col].dropna()
                    if series.empty:
                        continue
                    # Integer year columns (e.g. FiscalYear, Year: 2020-2030)
                    if pd.api.types.is_numeric_dtype(df[col]):
                        if series.between(1800, 2150).all():
                            return col
                        if series.between(1, 4).all() and any(q in col.lower() for q in ("quarter", "qtr", "q")):
                            return col
                    else:
                        # Verify parseability on sample
                        try:
                            sample = series.head(10)
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

        # 4. Fallback: inspect any numeric column that has year-range values
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) >= 3 and series.between(1900, 2100).all() and pd.api.types.is_integer_dtype(df[col]):
                return col

        return None

    def rank_target_candidates(self, df: pd.DataFrame, time_col: Optional[str] = None, hint: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Rank candidate numeric targets based on statistical properties without hardcoded column names.
        Returns list of (column_name, suitability_score) sorted descending by score.
        """
        from agent.canonical_data_layer import CanonicalDataLayer

        # 1. User explicit target ALWAYS has highest priority
        if hint and hint in df.columns:
            coerced_hint = CanonicalDataLayer.coerce_numeric_series(df[hint])
            if coerced_hint.notna().sum() >= 3:
                return [(hint, 100.0)]

        num_cols = [
            c for c in df.columns
            if c != time_col and CanonicalDataLayer.coerce_numeric_series(df[c]).notna().sum() >= 3
        ]
        if not num_cols:
            return []

        scored_candidates: List[Tuple[str, float]] = []

        for col in num_cols:
            series = df[col].dropna()
            n = len(series)
            if n < 3:
                continue

            # Check 1: Exclude zero-variance / constant columns
            unique_count = series.nunique()
            if unique_count <= 1:
                continue

            # Check 2: Exclude temporal year columns (e.g., 1800-2150 with uniform/discrete steps)
            if pd.api.types.is_integer_dtype(df[col]) and series.between(1800, 2150).all():
                # If values are sorted year spans like 2020..2025, treat as temporal index
                if series.is_monotonic_increasing or series.std() < 50:
                    continue

            # Check 3: Exclude quarter/month integers (1..4 or 1..12 discrete codes)
            if pd.api.types.is_integer_dtype(df[col]) and unique_count <= 4 and series.between(1, 4).all():
                continue

            # Check 4: Exclude identifier columns (monotonic sequence or unique ratio ~ 1.0 on integer IDs)
            if pd.api.types.is_integer_dtype(df[col]) and unique_count / n > 0.95 and series.min() >= 0:
                diffs = series.diff().dropna()
                if not diffs.empty and (diffs == 1).mean() > 0.8:
                    continue

            # Compute Dataset-Agnostic Statistical Suitability Score
            score = 50.0

            # Completeness bonus (fewer nulls = higher reliability)
            non_null_ratio = len(series) / len(df)
            score += non_null_ratio * 15.0

            # Cardinality / Continuous measure bonus (distinguishes measures from categorical codes)
            cardinality_ratio = unique_count / n
            if unique_count >= 10:
                score += 15.0
            elif unique_count >= 5:
                score += 8.0

            # Distribution spread & Coefficient of Variation
            mean_val = abs(series.mean())
            std_val = series.std()
            if std_val > 0 and mean_val > 0:
                cv = std_val / mean_val
                # Healthy variation (neither near-zero nor infinite noise)
                if 0.01 <= cv <= 10.0:
                    score += 10.0

            # Temporal correlation check (if time_col is present, measures usually have longitudinal structure)
            if time_col and time_col in df.columns:
                try:
                    time_clean = pd.to_datetime(df.loc[series.index, time_col], errors="coerce")
                    if time_clean.notna().sum() >= 4:
                        time_order = np.arange(len(series))
                        corr = np.corrcoef(time_order, series.to_numpy())[0, 1]
                        if not np.isnan(corr) and abs(corr) > 0.1:
                            score += min(10.0, abs(corr) * 10.0)
                except Exception:
                    pass

            scored_candidates.append((col, round(score, 2)))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates

    def detect_target_column(self, df: pd.DataFrame, time_col: Optional[str] = None, hint: Optional[str] = None) -> Optional[str]:
        """Identify primary numeric forecasting target using generic statistical ranking."""
        ranked = self.rank_target_candidates(df, time_col=time_col, hint=hint)
        return ranked[0][0] if ranked else None

    def get_ambiguous_targets(self, df: pd.DataFrame, time_col: Optional[str] = None, threshold: float = 5.0) -> List[str]:
        """Return list of candidate targets if top candidates are close in score and ambiguity exists."""
        ranked = self.rank_target_candidates(df, time_col=time_col)
        if len(ranked) < 2:
            return []
        top_score = ranked[0][1]
        close_candidates = [col for col, sc in ranked if (top_score - sc) <= threshold]
        return close_candidates if len(close_candidates) > 1 else []

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
        from agent.canonical_data_layer import CanonicalDataLayer
        dt_series = pd.to_datetime(df[time_col], errors="coerce")
        num_series = CanonicalDataLayer.coerce_numeric_series(df[target_col])
        clean_df = pd.DataFrame({"__time": dt_series, "__target": num_series}).dropna().sort_values("__time")
        n_obs = len(clean_df)

        if n_obs == 0:
            return ForecastSuitabilityResult(
                suitable=False,
                score=0.0,
                detected_time_column=time_col,
                detected_target=target_col,
                reasons=["No valid timestamp and numeric target pairs found."],
                warnings=["Observations could not be parsed into chronological numbers."],
            )

        freq_str, is_regular = self.infer_frequency(clean_df["__time"])
        time_range = {
            "start": str(clean_df["__time"].iloc[0]),
            "end": str(clean_df["__time"].iloc[-1]),
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
                warnings=["Insufficient historical data to detect temporal autocorrelation or trend."],
                limitations=["Statistical forecasts with N < 5 are mathematically unidentifiable."],
            )

        # Statistical suitability scoring
        y = clean_df["__target"].to_numpy()
        var_y = float(np.var(y))
        mean_y = float(np.mean(y))
        has_trend = False
        has_seasonality = False
        score = 0.70

        if n_obs >= 8:
            y = clean_df["__target"].to_numpy(dtype=float)
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

