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
        if dataframe is None or dataframe.empty:
            return ChatResponse("The dataset is empty. Please provide or upload data to analyze.", "unsupported", "unsupported", suggested_questions=[])
        text = message.casefold().strip(); context = context or {}
        columns = list(dataframe.columns)
        mentioned = self._columns(text, columns)
        metric = self._metric(mentioned, dataframe, context)
        group = self._group(mentioned, dataframe, context)
        if self._ambiguous(text, metric, group):
            return ChatResponse("How should I define best—highest sales, highest profit, or most units sold?", "clarification", "needs_clarification", suggested_questions=self._metric_questions(dataframe))
        
        from agent.filter_engine import FilterEngine
        plan = FilterEngine.parse_query_plan(message, columns, dataframe)
        if FilterEngine.has_filter_intent(text) or plan.filter is not None or (plan.group_by and (len(plan.group_by) > 1 or plan.ranking or plan.secondary_analysis)):
            filter_res = FilterEngine.execute(dataframe, message)
            evidence = {
                "filter": filter_res.filter_description,
                "matching_rows": filter_res.matching_rows,
                "total_rows": filter_res.total_rows,
                "aggregations": filter_res.aggregations,
                "columns": filter_res.columns,
                "group_by": filter_res.group_by,
                "grouped_records": filter_res.grouped_records,
                "highest_record": filter_res.highest_record,
                "lowest_record": filter_res.lowest_record,
                "secondary_results": filter_res.secondary_results,
            }
            return ChatResponse(
                filter_res.markdown_response,
                "grouping_and_ranking" if plan.group_by else "filtering",
                "success",
                evidence=evidence,
                command_result={
                    "filter": filter_res.filter_description,
                    "matching_rows": filter_res.matching_rows,
                    "total_rows": filter_res.total_rows,
                    "aggregations": filter_res.aggregations,
                    "filtered_data": filter_res.filtered_df.head(10).to_dict(orient="records"),
                    "group_by": filter_res.group_by,
                    "grouped_records": filter_res.grouped_records,
                    "highest_record": filter_res.highest_record,
                    "lowest_record": filter_res.lowest_record,
                    "secondary_results": filter_res.secondary_results,
                },
                suggested_questions=self._metric_questions(dataframe),
            )
            diag_resp = (
                f"\nFINAL_RESPONSE\n"
                f"type=ChatResponse\n"
                f"keys={list(resp_obj.__dict__.keys())}\n"
            )
            print(diag_resp)
            import logging
            logging.getLogger("diagnostic").info(diag_resp)
            return resp_obj

        if any(word in text for word in ("schema", "columns", "fields")):
            evidence = self.tools.execute("get_dataset_schema", dataframe)
            return ChatResponse(f"This dataset has {evidence['rows']} rows and these columns: {', '.join(evidence['columns'])}.", "schema", "success", evidence, suggested_questions=self._metric_questions(dataframe))
        if not metric:
            # Fallback for purely non-numeric datasets or general overview requests
            evidence = self.tools.execute("get_dataset_schema", dataframe)
            cat_cols = [c for c in dataframe.columns if not pd.api.types.is_numeric_dtype(dataframe[c])]
            cat_summary = f" Categorical fields: {', '.join(cat_cols[:5])}." if cat_cols else ""
            return ChatResponse(
                f"I analyzed this dataset ({evidence['rows']:,} rows, {len(columns)} columns).{cat_summary} You can ask me to analyze distributions, count categories, inspect values, or compare groups.",
                "schema",
                "success",
                evidence,
                suggested_questions=self._metric_questions(dataframe),
            )
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
        if any(word in text for word in ("why", "reason", "cause", "decrease", "decline", "drop", "increase", "rise", "grew", "fall", "fell", "trend")):
            date = self._date_column(dataframe)
            if date:
                try:
                    growth = self.tools.execute("calculate_growth", dataframe, metric_column=metric, date_column=date)
                except ValueError as exc:
                    return ChatResponse(f"{exc} Please check that '{metric}' and '{date}' are valid columns.", "trend", "unsupported")
                if growth:
                    most_recent = growth[-1]
                    change = most_recent.get("growth_percent")
                    evidence = {"metric": metric, "date_column": date, "periods": growth}
                    if change is None:
                        answer = f"Across the periods I could measure, {metric} shows {self._describe_trend(growth)}. I don't have enough earlier data to compute a precise period-over-period change."
                    elif change < 0:
                        answer = f"Looking at the data, {metric} {self._describe_trend(growth)}. The most recent period changed by {self._format(abs(change))}% compared with the prior period — that is what drives the decrease. Correlation is not causation, so this is an observed change, not a proven explanation."
                    else:
                        answer = f"Looking at the data, {metric} {self._describe_trend(growth)}. The most recent period changed by {self._format(abs(change))}% compared with the prior period. This describes the observed trend rather than proving a cause."
                    return ChatResponse(answer, "trend", "success", evidence, self._chart_visualization(dataframe, "line", date, metric), self._metric_questions(dataframe))
        if any(word in text for word in ("correlat", "related", "relationship", "association", "pearson", "spearman")) and len(self._numeric_columns(dataframe)) >= 2:
            from agent.statistical_analysis_engine import StatisticalAnalysisEngine
            engine = StatisticalAnalysisEngine()
            stats_res = engine.analyze(data=dataframe)
            rels = stats_res.get("relationships", [])
            top_rels = stats_res.get("top_relationships", []) or rels
            corr_matrix = stats_res.get("correlation_matrix", {})

            # Format comprehensive table & explanation
            lines = ["📊 **Statistical Relationship & Correlation Analysis**:\n"]
            if top_rels:
                lines.append("| Relationship | Pearson r | Spearman ρ | Raw p-value | FDR-adjusted p-value | Effect Strength | Valid N | Outlier Sensitivity |")
                lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
                for r in top_rels[:8]:
                    fx = r.get("feature_x") or r.get("variable_x")
                    fy = r.get("feature_y") or r.get("variable_y")
                    p_info = r.get("pearson", {}) or {}
                    s_info = r.get("spearman", {}) or {}
                    p_r = p_info.get("r", r.get("statistic", 0.0))
                    p_val = p_info.get("p_value", r.get("p_value", 0.0))
                    rho_val = s_info.get("rho", 0.0)
                    adj_p = r.get("adjusted_p_value", p_val)
                    strength = r.get("strength", "moderate").replace("_", " ").title()
                    outlier_sens = "Yes ⚠️" if r.get("outlier_sensitivity") else "No"
                    v_n = r.get("valid_rows", len(dataframe))
                    lines.append(
                        f"| **{fx}** ↔ **{fy}** | {p_r:.3f} | {rho_val:.3f} | {p_val:.4g} | {adj_p:.4g} | {strength} | {v_n} | {outlier_sens} |"
                    )

                lines.append("\n**Key Statistical Findings**:")
                for i, r in enumerate(top_rels[:3], 1):
                    fx = r.get("feature_x")
                    fy = r.get("feature_y")
                    p_r = r.get("pearson", {}).get("r", r.get("statistic", 0.0))
                    rho_val = r.get("spearman", {}).get("rho", 0.0)
                    p_val = r.get("p_value", 0.0)
                    adj_p = r.get("adjusted_p_value", p_val)
                    strength = r.get("strength", "moderate")
                    lines.append(
                        f"{i}. **{fx}** ↔ **{fy}**: {strength.replace('_', ' ').title()} association (Pearson r = {p_r:.3f}, Spearman ρ = {rho_val:.3f}, raw p = {p_val:.4g}, FDR-adjusted p = {adj_p:.4g}, valid N = {r.get('valid_rows')})."
                    )

            lines.append("\n> [!NOTE]\n> *All reported p-values are adjusted via Benjamini-Hochberg FDR control. Correlations describe mathematical associations without establishing causal mechanisms.*")
            answer = "\n".join(lines)

            # Preserve legacy fields in evidence while attaching detailed relationship records
            evidence = {
                "columns": list(dataframe.columns),
                "numeric_columns": self._numeric_columns(dataframe),
                "correlation_matrix": corr_matrix,
                "relationships": rels,
                "top_relationships": top_rels,
                "correlation": top_rels[0].get("statistic") if top_rels else None,
            }
            return ChatResponse(
                answer,
                "correlation",
                "success",
                evidence,
                command_result={"relationships": rels, "correlation_matrix": corr_matrix},
                suggested_questions=self._metric_questions(dataframe),
            )
        if any(word in text for word in ("show", "display", "visualize", "chart", "plot", "monthly", "over time", "trend")):
            date = self._date_column(dataframe)
            target = date or self._categorical_column(dataframe)
            if target:
                visualization = self._chart_visualization(dataframe, "line", target, metric)
                evidence = {"x": target, "y": metric, "chart": "line"}
                return ChatResponse(f"I prepared a line chart of {metric} over {target}.", "visualization", "success", evidence, visualization, self._metric_questions(dataframe))
        if any(word in text for word in ("anomal", "outlier", "unusual")):
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
    def _columns(text: str, columns: list[str]) -> list[str]:
        matched = []
        normalized_text = " " + re.sub(r"[^\w\s]", " ", text.casefold()) + " "
        synonyms = {
            "revenue": ["sales", "turnover", "income", "amount", "spend", "earning", "earnings", "price", "val", "value", "charges"],
            "profit": ["margin", "gain", "net", "earnings"],
            "cost": ["expense", "expenditure", "charges", "fee", "fees"],
            "user": ["customer", "client", "buyer", "account", "member"],
            "date": ["time", "period", "timestamp", "year", "month", "day"],
            "quantity": ["units", "volume", "count", "items", "number"],
        }
        for c in columns:
            c_low = c.casefold()
            c_spaced = c_low.replace("_", " ").replace("-", " ")
            if c_low in text or f" {c_spaced} " in normalized_text or f" {c_low} " in normalized_text:
                matched.append(c)
                continue
            parts = [p for p in c_low.split("_") if len(p) > 2]
            if parts and any(f" {p} " in normalized_text for p in parts):
                matched.append(c)
                continue
            for k, syn_list in synonyms.items():
                if k in c_low and any(f" {s} " in normalized_text for s in syn_list):
                    matched.append(c)
                    break
                elif any(s in c_low for s in syn_list) and f" {k} " in normalized_text:
                    matched.append(c)
                    break
        return list(dict.fromkeys(matched))

    def _metric(self, mentioned: list[str], dataframe: pd.DataFrame, context: dict[str, Any]) -> str | None:
        numeric = [c for c in mentioned if pd.api.types.is_numeric_dtype(dataframe[c])]
        if numeric:
            return numeric[0]
        ctx_m = context.get("metric")
        if ctx_m and ctx_m in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[ctx_m]):
            return ctx_m
        # Fallback to first numeric column in dataframe if present
        all_numeric = self._numeric_columns(dataframe)
        return all_numeric[0] if all_numeric else None

    def _group(self, mentioned: list[str], dataframe: pd.DataFrame, context: dict[str, Any]) -> str | None:
        groups = [c for c in mentioned if not pd.api.types.is_numeric_dtype(dataframe[c])]
        if groups:
            return groups[0]
        ctx_g = context.get("group_by")
        if ctx_g and ctx_g in dataframe.columns:
            return ctx_g
        return None
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
    def _numeric_columns(dataframe: pd.DataFrame) -> list[str]:
        return [c for c in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[c])]
    @staticmethod
    def _describe_trend(growth: list[dict[str, Any]]) -> str:
        changes = [g.get("growth_percent") for g in growth if g.get("growth_percent") is not None]
        if not changes:
            return "an essentially flat pattern over the measured periods"
        if all(c > 0 for c in changes):
            return "increased in every measured period"
        if all(c < 0 for c in changes):
            return "decreased in every measured period"
        rising = sum(1 for c in changes if c > 0)
        return f"a mixed pattern ({rising} rising of {len(changes)} measured periods)"
    def _chart_visualization(self, dataframe: pd.DataFrame, chart_type: str, x: str, y: str | None) -> dict[str, Any] | None:
        try:
            tool = f"create_{chart_type}_chart"
            args = {"x": x, "y": y} if y else {"x": x}
            return self.tools.execute(tool, dataframe, **args)
        except Exception:
            return None
    @staticmethod
    def _format(value: Any) -> str: return f"{value:,.2f}" if isinstance(value, float) else f"{value:,}" if isinstance(value, int) else str(value)
    @staticmethod
    def _ambiguous(text: str, metric: str | None, group: str | None) -> bool:
        return "best" in text and ("which is best" in text or not any(m in text for m in ("sales", "profit", "revenue", "cost", "margin", "units", "quantity", "price", "highest", "lowest")))
    def _metric_questions(self, dataframe: pd.DataFrame, group: str | None = None) -> list[str]:
        metrics = list(dataframe.select_dtypes(include="number").columns); metric = metrics[0] if metrics else "data"
        category = group or self._categorical_column(dataframe)
        return [f"Which {category} has the highest {metric}?" if category else f"What is the total {metric}?", f"Show a chart of {metric}.", f"Are there anomalies in {metric}?"]
