"""Rules that turn dataframe shapes into chart recommendations."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


class ChartSelector:
    """Select charts from column types; this module never creates figures."""

    def recommend(self, dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
        numeric = self._numeric_columns(dataframe)
        dates = self._date_columns(dataframe, numeric)
        categories = [
            column for column in dataframe.columns
            if column not in numeric and column not in dates and dataframe[column].notna().any()
        ]
        recommendations: List[Dict[str, Any]] = []

        # Keep the order stable: it also gives dashboard tiles a predictable layout.
        if dates and numeric:
            recommendations.append(self._recommendation("line", dates[0], numeric[0],
                                                       f"{self._label(numeric[0])} Over Time"))
        if categories and numeric:
            recommendations.append(self._recommendation("bar", categories[0], numeric[0],
                                                       f"{self._label(numeric[0])} by {self._label(categories[0])}"))
        if len(numeric) >= 2:
            recommendations.append(self._recommendation("scatter", numeric[0], numeric[1],
                                                       f"{self._label(numeric[1])} vs {self._label(numeric[0])}"))
        if numeric:
            recommendations.append(self._recommendation("histogram", numeric[0], None,
                                                       f"Distribution of {self._label(numeric[0])}"))
            recommendations.append(self._recommendation("box", numeric[0], None,
                                                       f"{self._label(numeric[0])} Distribution"))
        if len(numeric) >= 2:
            recommendations.append(self._recommendation("heatmap", None, None, "Numeric Column Correlation",
                                                       columns=numeric))
        return recommendations

    @staticmethod
    def _recommendation(chart_type: str, x: str | None, y: str | None, title: str,
                        **extra: Any) -> Dict[str, Any]:
        return {"chart_type": chart_type, "x": x, "y": y, "title": title, **extra}

    @staticmethod
    def _numeric_columns(dataframe: pd.DataFrame) -> List[str]:
        return [column for column in dataframe.columns
                if pd.api.types.is_numeric_dtype(dataframe[column]) and dataframe[column].notna().any()]

    @staticmethod
    def _date_columns(dataframe: pd.DataFrame, numeric: List[str]) -> List[str]:
        dates: List[str] = []
        for column in dataframe.columns:
            series = dataframe[column]
            if column in numeric or not series.notna().any():
                continue
            if pd.api.types.is_datetime64_any_dtype(series):
                dates.append(column)
                continue
            # Only infer dates for clearly date-like field names. This avoids treating IDs as dates.
            name = str(column).lower()
            if any(token in name for token in ("date", "time", "month", "year", "day")):
                parsed = pd.to_datetime(series, errors="coerce")
                if parsed.notna().sum() >= max(1, series.notna().sum() * 0.8):
                    dates.append(column)
        return dates

    @staticmethod
    def _label(column: str) -> str:
        return str(column).replace("_", " ").strip().title()
