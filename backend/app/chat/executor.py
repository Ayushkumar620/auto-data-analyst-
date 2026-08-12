"""Deterministic implementations behind the approved chat tools."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from backend.app.eda.orchestrator import EDAOrchestrator
from backend.app.insights import InsightEngine
from backend.app.visualization.engine import VisualizationEngine
from backend.app.forecasting import Forecaster

class DataExecutor:
    """Executes whitelisted dataframe operations; it never evaluates user code."""
    def get_dataset_schema(self, dataframe: pd.DataFrame) -> dict[str, Any]:
        return {"columns": list(dataframe.columns), "dtypes": {c: str(t) for c, t in dataframe.dtypes.items()}, "rows": int(len(dataframe))}

    def get_column_statistics(self, dataframe: pd.DataFrame, column: str) -> dict[str, Any]:
        self._column(dataframe, column); values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if values.empty: raise ValueError(f"Column '{column}' has no numeric values.")
        return {"column": column, "count": int(values.count()), "sum": self._number(values.sum()), "mean": self._number(values.mean()), "min": self._number(values.min()), "max": self._number(values.max())}

    def filter_data(self, dataframe: pd.DataFrame, column: str, operator: str, value: Any) -> pd.DataFrame:
        self._column(dataframe, column)
        if operator != "equals": raise ValueError("Only the approved 'equals' filter is supported.")
        return dataframe.loc[dataframe[column].astype(str).str.casefold() == str(value).casefold()].copy()

    def aggregate_data(self, dataframe: pd.DataFrame, column: str, operation: str = "sum") -> dict[str, Any]:
        self._column(dataframe, column); values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if values.empty: raise ValueError(f"Column '{column}' has no numeric values.")
        operations = {"sum": values.sum, "mean": values.mean, "average": values.mean, "min": values.min, "max": values.max, "count": values.count}
        if operation not in operations: raise ValueError(f"Unsupported aggregation '{operation}'.")
        return {"metric": column, "operation": operation, "value": self._number(operations[operation]()), "rows_used": int(values.count())}

    def group_by(self, dataframe: pd.DataFrame, group_column: str, metric_column: str, operation: str = "sum") -> list[dict[str, Any]]:
        self._column(dataframe, group_column); self._column(dataframe, metric_column)
        data = pd.DataFrame({group_column: dataframe[group_column], metric_column: pd.to_numeric(dataframe[metric_column], errors="coerce")}).dropna()
        if data.empty: raise ValueError("No usable rows remain after removing missing values.")
        operations = {"sum": "sum", "mean": "mean", "average": "mean", "min": "min", "max": "max", "count": "count"}
        if operation not in operations: raise ValueError(f"Unsupported aggregation '{operation}'.")
        result = data.groupby(group_column, dropna=False)[metric_column].agg(operations[operation]).reset_index().sort_values(metric_column, ascending=False)
        return [{group_column: self._safe(row[group_column]), metric_column: self._number(row[metric_column])} for _, row in result.iterrows()]

    def calculate_growth(self, dataframe: pd.DataFrame, metric_column: str, date_column: str) -> list[dict[str, Any]]:
        self._column(dataframe, metric_column); self._column(dataframe, date_column)
        data = pd.DataFrame({date_column: pd.to_datetime(dataframe[date_column], errors="coerce"), metric_column: pd.to_numeric(dataframe[metric_column], errors="coerce")}).dropna()
        monthly = data.groupby(data[date_column].dt.to_period("M"))[metric_column].sum().sort_index()
        return [{"period": str(period), "value": self._number(value), "growth_percent": None if pd.isna(growth) else self._number(growth)} for (period, value), growth in zip(monthly.items(), monthly.pct_change().mul(100))]

    def detect_anomalies(self, dataframe: pd.DataFrame, column: str) -> dict[str, Any]:
        self._column(dataframe, column); values = pd.to_numeric(dataframe[column], errors="coerce").dropna(); q1, q3 = values.quantile(.25), values.quantile(.75); iqr = q3 - q1
        outliers = values[(values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)]
        return {"column": column, "anomaly_count": int(len(outliers)), "values": [self._number(v) for v in outliers.head(20)]}

    def calculate_correlation(self, dataframe: pd.DataFrame, left: str, right: str) -> dict[str, Any]:
        self._column(dataframe, left); self._column(dataframe, right); data = dataframe[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        return {"columns": [left, right], "correlation": self._number(data[left].corr(data[right])) if len(data) > 1 else None}

    def create_bar_chart(self, dataframe: pd.DataFrame, x: str, y: str) -> dict[str, Any]: return self._chart(dataframe, "bar", x, y)
    def create_line_chart(self, dataframe: pd.DataFrame, x: str, y: str) -> dict[str, Any]: return self._chart(dataframe, "line", x, y)
    def create_scatter_chart(self, dataframe: pd.DataFrame, x: str, y: str) -> dict[str, Any]: return self._chart(dataframe, "scatter", x, y)
    def create_histogram(self, dataframe: pd.DataFrame, x: str) -> dict[str, Any]: return self._chart(dataframe, "histogram", x, None)
    def run_eda(self, dataframe: pd.DataFrame) -> dict[str, Any]: return EDAOrchestrator().analyze(dataframe)
    def generate_insights(self, dataframe: pd.DataFrame) -> dict[str, Any]: return InsightEngine().generate(dataframe)
    def forecast(self, dataframe: pd.DataFrame, metric_column: str, date_column: str, periods: int = 1) -> dict[str, Any]:
        return Forecaster().forecast(dataframe, periods, metric_column, date_column).to_dict()

    def _chart(self, dataframe: pd.DataFrame, chart_type: str, x: str, y: str | None) -> dict[str, Any]:
        return VisualizationEngine().generate(dataframe, {"chart_type": chart_type, "x": x, "y": y, "title": f"{y or x} by {x}"})

    @staticmethod
    def _column(dataframe: pd.DataFrame, column: str) -> None:
        if column not in dataframe.columns: raise ValueError(f"Column '{column}' does not exist.")
    @staticmethod
    def _number(value: Any) -> int | float | None:
        if pd.isna(value) or not np.isfinite(value): return None
        return int(value) if float(value).is_integer() else round(float(value), 6)
    @staticmethod
    def _safe(value: Any) -> Any:
        return None if pd.isna(value) else str(value) if isinstance(value, (pd.Timestamp, pd.Timedelta)) else value
