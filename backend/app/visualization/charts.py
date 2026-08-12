"""Plotly figure factories for the supported dashboard chart types."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class ChartFactory:
    def create(self, dataframe: pd.DataFrame, chart_type: str, x: str | None = None,
               y: str | None = None, title: str | None = None,
               columns: Iterable[str] | None = None) -> go.Figure:
        chart_type = chart_type.lower()
        if chart_type == "bar":
            self._require(dataframe, x, y)
            self._numeric(dataframe, y)
            data = dataframe[[x, y]].dropna().groupby(x, as_index=False)[y].sum()
            figure = go.Figure(go.Bar(x=data[x], y=data[y]))
        elif chart_type == "line":
            self._require(dataframe, x, y)
            self._numeric(dataframe, y)
            data = dataframe[[x, y]].dropna().copy()
            data[x] = pd.to_datetime(data[x], errors="coerce")
            data = data.dropna(subset=[x]).groupby(x, as_index=False)[y].sum().sort_values(x)
            figure = go.Figure(go.Scatter(x=data[x], y=data[y], mode="lines+markers"))
        elif chart_type == "scatter":
            self._require(dataframe, x, y)
            self._numeric(dataframe, x, y)
            data = dataframe[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            figure = go.Figure(go.Scatter(x=data[x], y=data[y], mode="markers"))
        elif chart_type in {"histogram", "box"}:
            value = x or y
            self._require(dataframe, value)
            self._numeric(dataframe, value)
            data = dataframe[value].replace([np.inf, -np.inf], np.nan).dropna()
            figure = go.Figure(go.Histogram(x=data) if chart_type == "histogram" else go.Box(y=data, boxpoints="outliers"))
        elif chart_type == "heatmap":
            selected = list(columns or dataframe.select_dtypes(include="number").columns)
            if len(selected) < 2:
                raise ValueError("A heatmap requires at least two numeric columns.")
            self._require(dataframe, *selected)
            self._numeric(dataframe, *selected)
            correlation = dataframe[selected].replace([np.inf, -np.inf], np.nan).corr().fillna(0)
            figure = go.Figure(go.Heatmap(z=correlation.values, x=selected, y=selected,
                                          colorscale="RdBu", zmin=-1, zmax=1))
        else:
            raise ValueError(f"Unsupported chart type '{chart_type}'. Supported types: bar, line, scatter, histogram, box, heatmap.")
        figure.update_layout(title=title or chart_type.title(), template="plotly_white")
        return figure

    @staticmethod
    def _require(dataframe: pd.DataFrame, *columns: str | None) -> None:
        available = ", ".join(map(str, dataframe.columns)) or "(none)"
        for column in columns:
            if not column:
                raise ValueError("This chart requires the requested column names.")
            if column not in dataframe.columns:
                raise ValueError(f"Column '{column}' does not exist. Available columns: {available}.")

    @staticmethod
    def _numeric(dataframe: pd.DataFrame, *columns: str | None) -> None:
        for column in columns:
            if column and not pd.api.types.is_numeric_dtype(dataframe[column]):
                raise ValueError(f"Column '{column}' must be numeric for this chart.")
