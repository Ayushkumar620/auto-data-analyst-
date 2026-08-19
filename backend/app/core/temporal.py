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