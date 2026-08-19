"""Temporal Intelligence Engine.

Automatically detects temporal fields (date, time, datetime, timestamp,
year, month, quarter) and computes validated trends.

A trend is ONLY reported when the full chain is satisfied:

    temporal field
    -> chronological ordering
    -> appropriate aggregation
    -> trend calculation
    -> validation

If no valid temporal structure exists, no trend is reported.
Growth is never inferred from "first row vs last row"; the input is first
sorted chronologically and aggregated at the detected frequency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Frequency labels aligned with pandas resample offsets.
FREQUENCY_LABELS: dict[str, str] = {
    "D": "daily", "W": "weekly", "ME": "monthly",
    "QE": "quarterly", "YE": "yearly", "H": "hourly",
    "min": "minutely", "s": "secondly",
}


class TemporalIntelligenceEngine:
    """Detects temporal fields and produces validated trend results."""

    MIN_TREND_OBSERVATIONS = 2
    MIN_CONFIDENT_TREND_OBSERVATIONS = 3

    # ------------------------------------------------------------------
    # Temporal field detection
    # ------------------------------------------------------------------
    def detect_fields(self, dataframe: pd.DataFrame,
                      semantic_roles: dict[str, str] | None = None) -> list[dict[str, Any]]:
        fields: list[dict[str, Any]] = []
        roles = semantic_roles or {}
        for column in dataframe.columns:
            series = dataframe[column]
            field = self._detect_column(series, str(column), roles.get(str(column)))
            if field is not None:
                fields.append(field)
        # Rank by confidence so the strongest temporal field comes first.
        fields.sort(key=lambda item: item["confidence"], reverse=True)
        return fields

    def _detect_column(self, series: pd.Series, column: str,
                       semantic_role: str | None) -> dict[str, Any] | None:
        normalized = str(column).strip().lower().replace(" ", "_")
        non_null = series.dropna()
        if non_null.empty:
            return None

        # 1) Already a datetime dtype.
        if pd.api.types.is_datetime64_any_dtype(series):
            has_time = bool(((non_null.dt.hour != 0) | (non_null.dt.minute != 0) |
                             (non_null.dt.second != 0) | (non_null.dt.microsecond != 0)).any())
            kind = "datetime" if has_time else "date"
            return self._field(column, kind, 0.98, series)

        # 2) Object / string columns that parse as dates.
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.8:
                has_time = bool(((parsed.dt.hour != 0) | (parsed.dt.minute != 0) |
                                 (parsed.dt.second != 0) | (parsed.dt.microsecond != 0)).any())
                if has_time:
                    kind, confidence = "datetime", 0.9
                else:
                    hint = any(token in normalized for token in ("date", "day", "dt", "when", "period"))
                    if not hint:
                        return None
                    kind, confidence = "date", 0.85
                return self._field(column, kind, confidence, series, parsed=parsed)

        # 3) Integer year / month / quarter columns.
        if pd.api.types.is_integer_dtype(series):
            values = non_null.astype(int)
            if any(token in normalized for token in ("year", "yr")) and values.nunique() <= 60:
                if ((values >= 1900) & (values <= 2100)).mean() >= 0.95:
                    return self._field(column, "year", 0.9, series)
            if any(token in normalized for token in ("month", "mth")) and values.nunique() <= 12:
                if ((values >= 1) & (values <= 12)).mean() >= 0.95:
                    return self._field(column, "month", 0.9, series)
            if any(token in normalized for token in ("quarter", "qtr")) and values.nunique() <= 4:
                if ((values >= 1) & (values <= 4)).mean() >= 0.95:
                    return self._field(column, "quarter", 0.9, series)

        # 4) Numeric unix-style timestamps.
        if pd.api.types.is_numeric_dtype(series):
            name_hint = any(token in normalized for token in ("timestamp", "ts", "epoch", "unix"))
            if name_hint:
                large = non_null.astype(float)
                mean_magnitude = float(np.abs(large).mean()) if len(large) else 0.0
                if mean_magnitude > 1e9:
                    unit = "s"
                elif mean_magnitude > 1e6:
                    unit = "ms"
                else:
                    unit = None
                if unit is not None:
                    parsed = pd.to_datetime(large, unit=unit, errors="coerce")
                    if parsed.notna().mean() >= 0.8:
                        return self._field(column, "timestamp", 0.85, series, parsed=parsed)
        return None

    def _field(self, column: str, kind: str, confidence: float, series: pd.Series,
               parsed: pd.Series | None = None) -> dict[str, Any]:
        values = parsed if parsed is not None else series
        values = pd.to_datetime(values, errors="coerce").dropna()
        field: dict[str, Any] = {
            "column": str(column),
            "temporal_kind": kind,
            "confidence": round(confidence, 3),
            "min": values.min().isoformat() if not values.empty else None,
            "max": values.max().isoformat() if not values.empty else None,
            "unique_count": int(values.nunique()) if not values.empty else 0,
            "frequency": None,
            "aggregation_required": False,
        }
        if len(values) >= 2:
            median_gap = float(values.sort_values().diff().dropna().median().total_seconds())
            field["span_seconds"] = round((values.max() - values.min()).total_seconds(), 3)
            if median_gap > 0:
                if median_gap < 3600:
                    field["frequency"] = "hourly"
                elif median_gap < 86400:
                    field["frequency"] = "daily"
                elif median_gap < 7 * 86400:
                    field["frequency"] = "weekly"
                elif median_gap < 31 * 86400:
                    field["frequency"] = "monthly"
                elif median_gap < 93 * 86400:
                    field["frequency"] = "quarterly"
                else:
                    field["frequency"] = "yearly"
                field["aggregation_required"] = field["frequency"] in {
                    "daily", "weekly", "monthly", "quarterly", "yearly",
                }
        return field

    # ------------------------------------------------------------------
    # Validated trend analysis
    # ------------------------------------------------------------------
    def analyze_trends(self, dataframe: pd.DataFrame,
                       temporal_columns: list[str] | None = None,
                       target_columns: list[str] | None = None) -> list[dict[str, Any]]:
        """Return validated trends for each (temporal, target) pair.

        Each trend follows: chronological ordering -> frequency inference ->
        aggregation -> trend calculation -> validation.  Pairs that fail any
        step are reported with status='insufficient' and no trend numbers.
        """
        date_fields = self.detect_fields(dataframe)
        if not date_fields:
            return []
        selected_dates = temporal_columns or [field["column"] for field in date_fields
                                              if field["temporal_kind"] in {"date", "datetime", "timestamp"}]
        if not selected_dates:
            return []

        if target_columns is None:
            target_columns = [column for column in dataframe.select_dtypes(include="number").columns]
        else:
            target_columns = [column for column in target_columns
                              if column in dataframe.columns]

        results: list[dict[str, Any]] = []
        for date_column in selected_dates:
            if date_column not in dataframe.columns:
                continue
            for target in target_columns:
                results.append(self._trend_pair(dataframe, date_column, target))
        return results

    def _trend_pair(self, dataframe: pd.DataFrame, date_column: str, target: str) -> dict[str, Any]:
        base: dict[str, Any] = {
            "date_column": date_column,
            "target_column": target,
            "status": "insufficient",
            "frequency": None,
            "aggregation": None,
            "periods": 0,
        }
        parsed = pd.to_datetime(dataframe[date_column], errors="coerce", format="mixed")
        numeric = pd.to_numeric(dataframe[target], errors="coerce")
        data = pd.DataFrame({date_column: parsed, target: numeric}).dropna()
        if len(data) < self.MIN_TREND_OBSERVATIONS:
            base["reason"] = "Fewer than two usable (date, value) observations"
            return base

        # Chronological ordering.
        data = data.sort_values(date_column)

        # Detect frequency from the sorted timeline.
        gaps = data[date_column].sort_values().diff().dropna()
        if gaps.empty:
            base["reason"] = "Only one distinct date"
            return base
        median_gap = float(gaps.median().total_seconds())
        frequency, offset = self._frequency_for_gap(median_gap)

        # Appropriate aggregation (roll up duplicate timestamps).
        aggregated = data.set_index(date_column).resample(offset)[target].agg(["sum", "count"]).reset_index()
        aggregated = aggregated.dropna(subset=["sum"])
        aggregated.columns = [date_column, target, "_count"]
        periods = len(aggregated)
        base["frequency"] = frequency
        base["aggregation"] = "sum"
        base["periods"] = int(periods)

        if periods < self.MIN_TREND_OBSERVATIONS:
            base["reason"] = f"Only {periods} aggregated period(s); at least {self.MIN_TREND_OBSERVATIONS} are required"
            return base

        values = aggregated[target].to_numpy(dtype=float)
        start_value = float(values[0])
        end_value = float(values[-1])
        if start_value == 0:
            growth_percentage = None
            absolute_change = float(end_value - start_value)
        else:
            growth_percentage = round((end_value - start_value) / abs(start_value) * 100, 2)
            absolute_change = round(end_value - start_value, 6)

        # Direction from period-over-period changes (validated by majority).
        changes = np.diff(values)
        rising = int((changes > 0).sum())
        falling = int((changes < 0).sum())
        flat = int((changes == 0).sum())
        if rising > 0 and rising > falling and rising > flat:
            direction = "up"
        elif falling > 0 and falling > rising and falling > flat:
            direction = "down"
        else:
            direction = "flat"

        volatility = round(float(np.std(values, ddof=1)), 6) if periods > 1 else 0.0
        confidence = self._trend_confidence(periods, growth_percentage, rising, falling)

        return {
            **base,
            "status": "valid",
            "start_value": round(start_value, 6),
            "end_value": round(end_value, 6),
            "growth_percentage": growth_percentage,
            "absolute_change": absolute_change,
            "direction": direction,
            "period_changes_rising": rising,
            "period_changes_falling": falling,
            "period_changes_flat": flat,
            "volatility": volatility,
            "confidence": confidence,
            "period_series": [
                {"period": str(row[date_column].to_period("D")),
                 "value": round(float(row[target]), 6)}
                for _, row in aggregated.iterrows()
            ][-12:],
        }

    def _frequency_for_gap(self, median_gap_seconds: float) -> tuple[str, str]:
        if median_gap_seconds < 3600:
            return "hourly", "h"
        if median_gap_seconds < 86400:
            return "daily", "D"
        if median_gap_seconds < 7 * 86400:
            return "weekly", "W"
        if median_gap_seconds < 31 * 86400:
            return "monthly", "ME"
        if median_gap_seconds < 93 * 86400:
            return "quarterly", "QE"
        return "yearly", "YE"

    def _trend_confidence(self, periods: int, growth: float | None,
                          rising: int, falling: int) -> float:
        if periods < self.MIN_CONFIDENT_TREND_OBSERVATIONS:
            return 0.35
        confidence = 0.6 + 0.05 * min(periods, 10)
        if growth is not None:
            confidence += 0.1 if abs(growth) >= 5 else 0.05
        if rising > 0 and falling == 0:
            confidence += 0.1
        elif falling > 0 and rising == 0:
            confidence += 0.1
        elif rising == falling:
            confidence -= 0.1
        return round(min(0.98, confidence), 3)

    # ------------------------------------------------------------------
    # Convenience helpers used by other engines
    # ------------------------------------------------------------------
    def primary_time_field(self, dataframe: pd.DataFrame) -> dict[str, Any] | None:
        fields = self.detect_fields(dataframe)
        for field in fields:
            if field["temporal_kind"] in {"date", "datetime", "timestamp"}:
                return field
        return None

    def growth_facts(self, dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        """Map trends into the compact fact shape consumed by the insight rules.

        Only 'valid' trends produce growth facts; nothing is reported when the
        temporal structure is insufficient.
        """
        facts: list[dict[str, Any]] = []
        for trend in self.analyze_trends(dataframe):
            if trend["status"] != "valid":
                continue
            facts.append({
                "date_column": trend["date_column"],
                "column": trend["target_column"],
                "growth_percentage": trend["growth_percentage"],
                "start_value": trend["start_value"],
                "end_value": trend["end_value"],
                "periods": trend["periods"],
                "frequency": trend["frequency"],
                "direction": trend["direction"],
                "confidence": trend["confidence"],
                "status": "valid",
            })
        return facts