from __future__ import annotations
import uuid
from typing import Any
import pandas as pd
from .schemas import Report
from .templates import METHODOLOGY

class ReportBuilder:
    def build(self, dataset_id: str, analysis: dict[str, Any]) -> Report:
        frame: pd.DataFrame = analysis["dataframe"]; eda = analysis.get("eda", {}); summary = eda.get("summary", {})
        cleaning = analysis.get("cleaning", {}); insights = analysis.get("insights", []); forecast = analysis.get("forecast", {})
        numeric = list(frame.select_dtypes(include="number").columns)
        kpis = [{"name": column, "value": self._number(frame[column].sum()), "operation": "sum"} for column in numeric[:4]]
        quality = {"missing_values": summary.get("missing_values", int(frame.isna().sum().sum())), "duplicate_rows": summary.get("duplicate_rows", int(frame.duplicated().sum())), "outliers": cleaning.get("outliers_detected", 0), "quality_before": cleaning.get("quality_before"), "quality_after": cleaning.get("quality_after"), "actions": cleaning.get("cleaning_report", [])}
        dates = self._period(frame)
        overview = {"rows": int(len(frame)), "columns": int(len(frame.columns)), "column_names": list(frame.columns), "analysis_period": dates}
        recommendations = [item["recommendation"] for item in insights if item.get("recommendation")]
        summary_text = self._summary(kpis, insights, forecast)
        return Report(f"report_{uuid.uuid4().hex[:8]}", dataset_id, f"{analysis.get('dataset_name', 'Dataset')} Analysis", summary_text, overview, quality, kpis, analysis.get("charts", []), insights, recommendations, forecast, METHODOLOGY, {"eda": eda})
    @staticmethod
    def _number(value: Any) -> int | float: return int(value) if float(value).is_integer() else round(float(value), 2)
    @staticmethod
    def _period(frame: pd.DataFrame) -> str | None:
        for column in frame.columns:
            if any(word in column.casefold() for word in ("date", "time", "month", "year")):
                dates = pd.to_datetime(frame[column], errors="coerce").dropna()
                if not dates.empty: return f"{dates.min():%b %Y} – {dates.max():%b %Y}"
        return None
    @staticmethod
    def _summary(kpis: list[dict[str, Any]], insights: list[dict[str, Any]], forecast: dict[str, Any]) -> str:
        text = "; ".join(f"{item['name']} totals {item['value']:,}" for item in kpis[:3]) or "The dataset contains no numeric KPIs."
        if insights: text += f" Key finding: {insights[0].get('description', insights[0].get('title', 'No insight available.'))}"
        if forecast: text += f" A {forecast.get('horizon')}-period {forecast.get('model', 'selected-model')} forecast is included."
        return text
