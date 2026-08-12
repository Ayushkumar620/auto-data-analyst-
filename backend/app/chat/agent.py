"""Dataset-aware intent detection and safe analytical orchestration."""
from __future__ import annotations
import re
from typing import Any
import pandas as pd
from .schemas import ChatResponse
from .tools import ToolRegistry
from .validator import ResultValidator

class ChatAgent:
    def __init__(self, tools: ToolRegistry | None = None) -> None:
        self.tools = tools or ToolRegistry(); self.validator = ResultValidator()

    def respond(self, dataframe: pd.DataFrame, message: str, context: dict[str, Any] | None = None) -> ChatResponse:
        text = message.casefold().strip(); context = context or {}
        columns = list(dataframe.columns)
        mentioned = self._columns(text, columns)
        metric = self._metric(mentioned, dataframe, context)
        group = self._group(mentioned, dataframe, context)
        if self._ambiguous(text, metric, group):
            return ChatResponse("How should I define best—highest sales, highest profit, or most units sold?", "clarification", "needs_clarification", suggested_questions=self._metric_questions(dataframe))
        if any(word in text for word in ("schema", "columns", "fields")):
            evidence = self.tools.execute("get_dataset_schema", dataframe)
            return ChatResponse(f"This dataset has {evidence['rows']} rows and these columns: {', '.join(evidence['columns'])}.", "schema", "success", evidence, suggested_questions=self._metric_questions(dataframe))
        if not metric:
            return ChatResponse("I can't determine that from the current dataset because I couldn't identify the required metric column.", "unsupported", "unsupported", suggested_questions=self._metric_questions(dataframe))
        if any(word in text for word in ("forecast", "predict", "projection", "next quarter")):
            date = self._date_column(dataframe)
            if not date:
                return ChatResponse("I can't reliably forecast this dataset because no usable date column was found.", "forecast", "unsupported")
            horizon = 3 if "quarter" in text else self._horizon(text)
            try:
                evidence = self.tools.execute("forecast", dataframe, metric_column=metric, date_column=date, periods=horizon)
            except ValueError as exc:
                return ChatResponse(str(exc), "forecast", "unsupported")
            first = evidence["forecast"][0]
            answer = f"The {evidence['model']} model forecasts {metric} at approximately {self._format(first['prediction'])} for {first['date']}, with an estimated range of {self._format(first['lower'])}–{self._format(first['upper'])}. Forecasts are estimates, not guarantees."
            return ChatResponse(answer, "forecast", "success", evidence, evidence.get("visualization"), self._metric_questions(dataframe))
        if any(word in text for word in ("anomal", "outlier")):
            evidence = self.tools.execute("detect_anomalies", dataframe, column=metric)
            answer = f"I found {evidence['anomaly_count']} potential anomalies in {metric}."
            return ChatResponse(answer, "anomaly_detection", "success", evidence, suggested_questions=self._metric_questions(dataframe))
        if any(word in text for word in ("average", "mean", "total", "sum", "maximum", "minimum", "highest", "lowest")) and not group:
            operation = self._operation(text); evidence = self.tools.execute("aggregate_data", dataframe, column=metric, operation=operation)
            return ChatResponse(f"The {operation} of {metric} is {self._format(evidence['value'])}.", "aggregation", "success", evidence, suggested_questions=self._metric_questions(dataframe))
        chart = self._chart(text)
        if chart:
            if not group: group = self._date_column(dataframe) or self._categorical_column(dataframe)
            if not group: return ChatResponse("I need a category or date column to create that chart.", "visualization", "unsupported")
            tool = f"create_{chart}_chart" if chart != "histogram" else "create_histogram"
            args = {"x": group, "y": metric} if chart != "histogram" else {"x": metric}
            visualization = self.tools.execute(tool, dataframe, **args)
            evidence = {"x": group, "y": metric, "aggregation": "sum", "chart": chart}
            return ChatResponse(f"I prepared a {chart} chart of {metric} by {group}.", "visualization", "success", evidence, visualization, self._metric_questions(dataframe))
        if group:
            operation = "sum" if any(word in text for word in ("highest", "lowest", "most", "total")) else self._operation(text)
            rows = self.tools.execute("group_by", dataframe, group_column=group, metric_column=metric, operation=operation)
            winner = rows[0]; evidence = {"metric": metric, "operation": operation, "group_by": group, "winner": winner[group], "value": winner[metric]}
            qualifier = "highest" if operation in {"sum", "max"} else "leading"
            return ChatResponse(f"{winner[group]} has the {qualifier} {metric} at {self._format(winner[metric])}.", "aggregation", "success", evidence, suggested_questions=self._metric_questions(dataframe, group))
        evidence = self.tools.execute("get_column_statistics", dataframe, column=metric)
        return ChatResponse(f"{metric} ranges from {self._format(evidence['min'])} to {self._format(evidence['max'])}, with an average of {self._format(evidence['mean'])}.", "statistics", "success", evidence, suggested_questions=self._metric_questions(dataframe))

    @staticmethod
    def _columns(text: str, columns: list[str]) -> list[str]: return [c for c in columns if c.casefold() in text]
    def _metric(self, mentioned: list[str], dataframe: pd.DataFrame, context: dict[str, Any]) -> str | None:
        numeric = [c for c in mentioned if pd.api.types.is_numeric_dtype(dataframe[c])]
        return numeric[0] if numeric else context.get("metric") if context.get("metric") in dataframe.columns else None
    def _group(self, mentioned: list[str], dataframe: pd.DataFrame, context: dict[str, Any]) -> str | None:
        groups = [c for c in mentioned if not pd.api.types.is_numeric_dtype(dataframe[c])]
        return groups[0] if groups else context.get("group_by") if context.get("group_by") in dataframe.columns else None
    @staticmethod
    def _operation(text: str) -> str:
        if any(w in text for w in ("average", "mean")): return "mean"
        if "minimum" in text or "lowest" in text: return "min"
        if "maximum" in text or "highest" in text: return "max" if "by" not in text else "sum"
        return "sum"
    @staticmethod
    def _chart(text: str) -> str | None:
        for name in ("bar", "line", "scatter", "histogram"):
            if name in text: return name
        return "line" if any(w in text for w in ("show", "chart", "plot", "monthly")) else None
    @staticmethod
    def _horizon(text: str) -> int:
        match = re.search(r"next\s+(\d+)\s+(?:day|week|month|period)", text)
        return int(match.group(1)) if match else 1
    @staticmethod
    def _date_column(dataframe: pd.DataFrame) -> str | None:
        return next((c for c in dataframe.columns if "date" in c.casefold() or "month" in c.casefold()), None)
    @staticmethod
    def _categorical_column(dataframe: pd.DataFrame) -> str | None:
        return next((c for c in dataframe.columns if not pd.api.types.is_numeric_dtype(dataframe[c])), None)
    @staticmethod
    def _ambiguous(text: str, metric: str | None, group: str | None) -> bool: return "best" in text and not metric
    def _metric_questions(self, dataframe: pd.DataFrame, group: str | None = None) -> list[str]:
        metrics = list(dataframe.select_dtypes(include="number").columns); metric = metrics[0] if metrics else "data"
        category = group or self._categorical_column(dataframe)
        return [f"Which {category} has the highest {metric}?" if category else f"What is the total {metric}?", f"Show a chart of {metric}.", f"Are there anomalies in {metric}?"]
    @staticmethod
    def _format(value: Any) -> str: return f"{value:,.2f}" if isinstance(value, float) else f"{value:,}" if isinstance(value, int) else str(value)
