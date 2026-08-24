"""
Specialized Agents - Each agent handles a specific type of analysis task.
Every agent attaches `Evidence` describing how its output was computed so
downstream consumers (planner, validator, report) can trace every claim to
the underlying data and method.
"""
import pandas as pd

from .base import BaseAgent
from .analyzer import DataAnalyzer
from .visualizer import DataVisualizer
from .predictor import DataPredictor
from .insights import InsightsEngine
from .cleaner import DataCleaner
from .model_selection_agent import ModelSelectionAgent
from .ann_agent import ANNAgent
from .cnn_agent import CNNAgent
from .registry_agent import ModelRegistryAgent
from .validation_agent import DataValidationAgent
from .command_orchestrator import AutonomousCommandOrchestrator
from .schemas import ClaimType, Evidence, ErrorCategory


def _frames(data):
    """Return a list of (name, DataFrame) pairs for single or multi-table data."""
    if isinstance(data, dict):
        return list(data.items())
    if isinstance(data, pd.DataFrame):
        return [("data", data)]
    return []


class DataLoadingAgent(BaseAgent):
    """Loads and validates data before passing to other agents."""

    name = "Data Loading Agent"
    description = "Loads and validates data from any file format"
    role = "loader"

    def run(self, task):
        self._start()
        try:
            data = task.get("data")
            source = task.get("source", "uploaded file")
            if data is None:
                return self._error("No data provided to load.")
            result = {
                "loaded": True,
                "source": source,
                "type": type(data).__name__,
                "tables": self._summarize(data),
            }
            evidence = self._load_evidence(data)
            return self._finish(
                result,
                evidence=evidence,
                confidence=1.0,
                metadata={"source": source},
            )
        except Exception as e:
            return self._error(str(e), category=ErrorCategory.INPUT_VALIDATION)

    def _load_evidence(self, data):
        evidence = []
        for name, df in _frames(data):
            evidence.append(self.make_evidence(
                method="dataframe.load",
                data_ref={
                    "frame": name,
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "column_names": list(df.columns),
                },
                confidence=1.0,
                claim_type=ClaimType.FACT,
            ))
        if not evidence:
            evidence.append(self.make_evidence(
                method="structure.detect",
                data_ref={"note": "Non-tabular data structure."},
                confidence=0.4,
                claim_type=ClaimType.OBSERVATION,
            ))
        return evidence

    def _summarize(self, data):
        if isinstance(data, pd.DataFrame):
            return {
                "dataframe": {
                    "rows": int(data.shape[0]),
                    "columns": int(data.shape[1]),
                    "column_names": list(data.columns),
                }
            }
        if isinstance(data, dict):
            return {
                f"table_{k}": {
                    "rows": int(v.shape[0]),
                    "columns": int(v.shape[1]),
                    "column_names": list(v.columns),
                }
                for k, v in data.items()
                if isinstance(v, pd.DataFrame)
            }
        return {"note": "Non-tabular data structure."}


class AnalysisAgent(BaseAgent):
    """Performs statistical analysis: summaries, stats, nulls, correlations."""

    name = "Analysis Agent"
    description = "Performs statistical analysis and data profiling"
    role = "analysis"

    # Maps logical requests to the precise pandas method used so the evidence
    # chain is exact.
    REQUEST_METHOD = {
        "summary": "pandas.DataFrame.describe(include='all')",
        "describe": "pandas.DataFrame.describe()",
        "nulls": "pandas.DataFrame.isnull().sum()",
        "correlation": "pandas.DataFrame.corr()",
        "head": "pandas.DataFrame.head(n=10)",
        "unique": "pandas.Series.nunique()",
    }

    def run(self, task):
        self._start()
        try:
            data = task.get("data")
            request = task.get("request", "summary")
            analyzer = DataAnalyzer(data)
            response = {}

            if request in ("summary", "overview", "info"):
                request = "summary"
                response = analyzer.summary()
            elif request in ("describe", "stats"):
                request = "describe"
                response = analyzer.describe()
            elif request in ("nulls", "missing"):
                request = "nulls"
                response = analyzer.nulls()
            elif request in ("correlation", "corr"):
                request = "correlation"
                response = analyzer.correlation()
            elif request in ("head", "view"):
                request = "head"
                response = analyzer.head()
            elif request in ("unique", "uniques"):
                request = "unique"
                response = analyzer.unique_values()
            else:
                request = "summary"
                response = analyzer.summary()

            evidence = self._analysis_evidence(data, request)
            confidence = 0.85 if request == "correlation" else 0.95
            return self._finish(
                {"request": request, "reports": response},
                evidence=evidence,
                confidence=confidence,
            )
        except Exception as e:
            return self._error(str(e), category=ErrorCategory.COMPUTATION)

    def _analysis_evidence(self, data, request):
        method = self.REQUEST_METHOD.get(request, "pandas.DataFrame.describe()")
        claim_type = ClaimType.CORRELATION if request == "correlation" else ClaimType.FACT
        evidence = []
        for name, df in _frames(data):
            if df.empty:
                continue
            evidence.append(self.make_evidence(
                method=method,
                data_ref={"frame": name, "rows": int(df.shape[0]),
                          "columns": int(df.shape[1]), "request": request},
                confidence=0.95,
                claim_type=claim_type,
            ))
        return evidence


class VisualizationAgent(BaseAgent):
    """Generates charts and visualizations."""

    name = "Visualization Agent"
    description = "Generates charts and graphs from data"
    role = "visualization"

    def run(self, task):
        self._start()
        try:
            data = task.get("data")
            chart_type = task.get("chart_type", "auto")
            x = task.get("x")
            y = task.get("y")
            visualizer = DataVisualizer(data)
            charts = visualizer.chart(chart_type=chart_type, x=x, y=y)
            evidence = []
            for chart in charts:
                evidence.append(self.make_evidence(
                    method=f"matplotlib.{chart.get('chart_type', 'chart')}",
                    data_ref={"frame": chart.get("name"), "x": x, "y": y,
                              "chart_type": chart.get("chart_type")},
                    confidence=0.9,
                    claim_type=ClaimType.OBSERVATION,
                ))
            return self._finish(
                {"chart_type": chart_type, "charts": charts},
                evidence=evidence,
                confidence=0.9 if charts else 0.0,
            )
        except Exception as e:
            return self._error(str(e), category=ErrorCategory.COMPUTATION)


class PredictionAgent(BaseAgent):
    """Builds ML models and makes predictions."""

    name = "Prediction Agent"
    description = "Trains machine learning models for predictions"
    role = "prediction"

    def run(self, task):
        self._start()
        try:
            data = task.get("data")
            target = task.get("target")
            predictor = DataPredictor(data)
            result = predictor.predict(target=target)
            if "error" in result:
                return self._finish(
                    result,
                    confidence=0.0,
                    warnings=[f"Prediction failed: {result['error']}"],
                )
            evidence, confidence = self._prediction_evidence(result)
            return self._finish(result, evidence=evidence, confidence=confidence)
        except Exception as e:
            return self._error(str(e), category=ErrorCategory.COMPUTATION)

    def _prediction_evidence(self, result):
        metric = result.get("metric", {})
        evidence = [self.make_evidence(
            method=f"sklearn.{str(metric.get('model', 'model')).lower().replace(' ', '_')}",
            data_ref={"target": result.get("target"),
                      "features": result.get("features", []),
                      "train_size": result.get("train_size"),
                      "test_size": result.get("test_size"),
                      "model": metric.get("model"),
                      "type": metric.get("type")},
            confidence=0.8,
            claim_type=ClaimType.FACT,
            raw_value=metric,
        )]
        if metric.get("type") == "classification":
            confidence = 0.7 + 0.25 * float(metric.get("accuracy", 0) or 0)
        else:
            r2 = metric.get("r2_score")
            confidence = min(0.95, 0.5 + 0.45 * float(r2)) if r2 is not None else 0.8
        return evidence, round(confidence, 3)


class ForecastAgent(BaseAgent):
    """Forecasts future values of a numeric column over time."""

    name = "Forecast Agent"
    description = "Forecasts future values of a time-series numeric column"
    role = "forecast"

    def run(self, task):
        self._start()
        try:
            data = task.get("data")
            target = task.get("target")
            periods = task.get("periods", 5)
            predictor = DataPredictor(data)
            result = predictor.forecast(target=target, periods=periods)
            if "error" in result:
                return self._finish(
                    result,
                    confidence=0.0,
                    warnings=[f"Forecast failed: {result['error']}"],
                )
            evidence = [self.make_evidence(
                method="position_based_linear_regression",
                data_ref={"target": result.get("target"),
                          "history_points": result.get("history_points"),
                          "date_col": result.get("date_col"),
                          "forecast_periods": result.get("forecast_periods"),
                          "trend": result.get("trend")},
                confidence=0.7,
                claim_type=ClaimType.INFERENCE,
                raw_value={"trend": result.get("trend"),
                           "projected_change_percent": result.get("projected_change_percent")},
            )]
            # Simple forecasts are inherently less certain than measured facts.
            return self._finish(result, evidence=evidence, confidence=0.7,
                                warnings=["Forecasts are estimates, not guarantees."])
        except Exception as e:
            return self._error(str(e), category=ErrorCategory.COMPUTATION)


class CleaningAgent(BaseAgent):
    """Automatically cleans and prepares data."""

    name = "Cleaning Agent"
    description = "Automatically detects and fixes data quality issues"
    role = "cleaning"

    def run(self, task):
        self._start()
        try:
            data = task.get("data")
            cleaner = DataCleaner(data)
            result = cleaner.clean()
            evidence = []
            total_actions = 0
            cleaned_frame_count = 0
            for report in result:
                actions = report.get("actions", [])
                total_actions += len(actions)
                cleaned_frame_count += 1
                evidence.append(self.make_evidence(
                    method="data_cleaning_pipeline",
                    data_ref={"frame": report.get("name"),
                              "original_shape": report.get("original_shape"),
                              "cleaned_shape": report.get("cleaned_shape"),
                              "actions": actions},
                    confidence=0.85,
                    claim_type=ClaimType.FACT,
                    raw_value=report.get("cleaned_shape"),
                ))
            return self._finish(
                {"reports": result},
                evidence=evidence,
                confidence=0.85 if cleaned_frame_count else 0.0,
                metadata={"total_actions": total_actions},
            )
        except Exception as e:
            return self._error(str(e), category=ErrorCategory.COMPUTATION)


class InsightAgent(BaseAgent):
    """Generates insights and answers natural-language questions."""

    name = "Insight Agent"
    description = "Generates insights and natural-language answers"
    role = "insight"

    # Map request types to the evidence claim type. Insights that interpret
    # patterns (smart, anomalies, report) are OBSERVATION/INFERENCE; direct
    # aggregations are FACT.
    REQUEST_CLAIM = {
        "text": ClaimType.FACT,
        "smart": ClaimType.OBSERVATION,
        "anomalies": ClaimType.OBSERVATION,
        "report": ClaimType.INFERENCE,
        "aggregate": ClaimType.FACT,
        "summary": ClaimType.OBSERVATION,
    }

    def run(self, task):
        self._start()
        try:
            data = task.get("data")
            intent = task.get("intent")
            engine = InsightsEngine(data)
            request_type = task.get("type", "")

            if request_type in ("structured", "evidence_based"):
                catalog = engine.generate_structured_insights(model_result=task.get("model_result"))
                evidence_list = []
                for ins in catalog.get("insights", [])[:5]:
                    ct_val = ins.get("claim_type", "observation")
                    try:
                        ct_enum = ClaimType(ct_val)
                    except ValueError:
                        ct_enum = ClaimType.OBSERVATION
                    evidence_list.append(
                        self.make_evidence(
                            method="evidence_based_insight",
                            data_ref={
                                "text": ins.get("text"),
                                "claim_type": ct_val,
                                "metrics": ins.get("supporting_metrics"),
                            },
                            confidence=ins.get("confidence", 0.9),
                            claim_type=ct_enum,
                        )
                    )
                return self._finish(
                    {"type": "structured_insights", "result": catalog},
                    evidence=evidence_list,
                    confidence=0.95,
                    warnings=["All correlations are strictly non-causal associations."],
                )

            if request_type == "text":
                result = engine.text_analysis()
                evidence, confidence = self._generic_evidence(request_type, result)
                return self._finish({"type": "text", "result": result},
                                    evidence=evidence, confidence=confidence)

            if request_type == "smart":
                result = engine.generate_smart_insights()
                evidence, confidence = self._generic_evidence("smart", result)
                return self._finish({"type": "insights", "result": result},
                                    evidence=evidence, confidence=confidence, warnings=["Insights are patterns derived from the data, not causal proof."])

            if request_type == "anomalies":
                result = engine.detect_anomalies()
                evidence, confidence = self._generic_evidence("anomalies", result)
                return self._finish({"type": "anomalies", "result": result},
                                    evidence=evidence, confidence=confidence)

            if request_type == "report":
                result = engine.generate_report()
                evidence, confidence = self._generic_evidence("report", result)
                return self._finish({"type": "report", "result": result},
                                    evidence=evidence, confidence=confidence)

            if intent is not None:
                result = engine.aggregate(intent)
                evidence, confidence = self._generic_evidence("aggregate", result)
                return self._finish({"type": "insight", "result": result},
                                    evidence=evidence, confidence=confidence)

            result = engine.summary_insights()
            evidence, confidence = self._generic_evidence("summary", result)
            return self._finish({"type": "insight", "result": result},
                                evidence=evidence, confidence=confidence)
        except Exception as e:
            return self._error(str(e), category=ErrorCategory.COMPUTATION)

    def _evidence(self, request_type):
        """Return a fresh evidence item referencing the data this insight used."""
        return self.make_evidence(
            method=f"insights_engine.{request_type}",
            data_ref={"analysis_kind": request_type},
            confidence=0.8,
            claim_type=self.REQUEST_CLAIM.get(request_type, ClaimType.OBSERVATION),
        )

    def _generic_evidence(self, request_type, result):
        """Build evidence + confidence for an insight result dict."""
        evidence = []
        if isinstance(result, dict) and "error" in result:
            return evidence, 0.0
        evidence.append(self._evidence(request_type))
        confidence = 0.9
        if request_type in ("smart", "summary"):
            confidence = 0.8  # interpretive insights carry inherent uncertainty
        # Down-weight strongest patterns but keep claims conservative.
        return evidence, confidence


class ReportAgent(BaseAgent):
    """Generates a professional narrative report from agent outputs."""

    name = "Report Agent"
    description = "Generates a professional report from agent findings"
    role = "report"

    def run(self, task):
        self._start()
        try:
            agent_outputs = task.get("agent_outputs", [])
            request = task.get("request", "analysis")
            report = self._build_report(agent_outputs, request)
            evidence = self._report_evidence(agent_outputs)
            completed_count = sum(
                1 for out in agent_outputs if out.get("status") == "completed"
            )
            return self._finish(
                {"report": report},
                evidence=evidence,
                confidence=0.9 if completed_count else 0.0,
                metadata={"upstream_agents": len(agent_outputs),
                          "completed_agents": completed_count},
            )
        except Exception as e:
            return self._error(str(e), category=ErrorCategory.COMPUTATION)

    def _report_evidence(self, outputs):
        """Evidence referencing each upstream agent's output as the report source."""
        evidence = []
        for out in outputs:
            agent_id = getattr(out, "agent_id", None) or out.get("agent_id")
            status = out.get("status")
            evidence.append(self.make_evidence(
                method="report.compose",
                data_ref={"upstream_agent": out.get("agent"),
                          "upstream_agent_id": agent_id,
                          "upstream_status": status,
                          "upstream_confidence": getattr(out, "confidence", None) or out.get("confidence")},
                confidence=0.9 if status == "completed" else 0.4,
                claim_type=ClaimType.INFERENCE,
            ))
        return evidence

    def _build_report(self, outputs, request):
        lines = []
        lines.append(f"# Data Analysis Report")
        lines.append("")
        lines.append(f"**Request:** {request}")
        lines.append("")
        lines.append("## Agent Work Pipeline")
        lines.append("")

        for out in outputs:
            if out.get("status") == "error":
                lines.append(f"- ❌ **{out.get('agent')}**: {out.get('output', {}).get('error', 'failed')}")
                continue
            lines.append(f"- ✅ **{out.get('agent')}** completed in {out.get('duration_ms', 0)}ms")

        lines.append("")
        lines.append("## Findings")
        lines.append("")
        for out in outputs:
            if out.get("status") != "completed":
                continue
            output = out.get("output", {})
            agent_name = out.get("agent", "")
            if "reports" in output:
                rep = output["reports"]
                if isinstance(rep, list) and rep:
                    first = rep[0]
                    if "shape" in first:
                        lines.append(f"### {agent_name} - Data Shape")
                        lines.append(f"- Rows: {first['shape']['rows']}, Columns: {first['shape']['columns']}")
                        lines.append(f"- Columns: {', '.join(first.get('columns', []))}")
            elif "result" in output and isinstance(output["result"], dict):
                res = output["result"]
                if "value" in res:
                    lines.append(f"### {agent_name} - Key Metric")
                    lines.append(f"- Value: {res['value']}")
                if "error" in res:
                    lines.append(f"### {agent_name}")
                    lines.append(f"- {res['error']}")
            elif "report" in output:
                lines.append(f"### {agent_name}")
                lines.append(output["report"])

        lines.append("")
        lines.append("---")
        lines.append("*Generated by the Auto Data Analyst multi-agent system*")
        return "\n".join(lines)
