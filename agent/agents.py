"""
Specialized Agents - Each agent handles a specific type of analysis task.
"""
import pandas as pd

from .base import BaseAgent
from .analyzer import DataAnalyzer
from .visualizer import DataVisualizer
from .predictor import DataPredictor
from .insights import InsightsEngine
from .cleaner import DataCleaner


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
            return self._finish(result)
        except Exception as e:
            return self._error(str(e))

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

    def run(self, task):
        self._start()
        try:
            data = task.get("data")
            request = task.get("request", "summary")
            analyzer = DataAnalyzer(data)
            response = {}

            if request in ("summary", "overview", "info"):
                response = analyzer.summary()
            elif request in ("describe", "stats"):
                response = analyzer.describe()
            elif request in ("nulls", "missing"):
                response = analyzer.nulls()
            elif request in ("correlation", "corr"):
                response = analyzer.correlation()
            elif request in ("head", "view"):
                response = analyzer.head()
            elif request in ("unique", "uniques"):
                response = analyzer.unique_values()
            else:
                response = analyzer.summary()

            return self._finish({"request": request, "reports": response})
        except Exception as e:
            return self._error(str(e))


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
            return self._finish(
                {"chart_type": chart_type, "charts": charts}
            )
        except Exception as e:
            return self._error(str(e))


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
            return self._finish(result)
        except Exception as e:
            return self._error(str(e))


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
            return self._finish(result)
        except Exception as e:
            return self._error(str(e))


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
            return self._finish({"reports": result})
        except Exception as e:
            return self._error(str(e))


class InsightAgent(BaseAgent):
    """Generates insights and answers natural-language questions."""

    name = "Insight Agent"
    description = "Generates insights and natural-language answers"
    role = "insight"

    def run(self, task):
        self._start()
        try:
            data = task.get("data")
            intent = task.get("intent")
            engine = InsightsEngine(data)
            request_type = task.get("type", "")

            if request_type == "text":
                result = engine.text_analysis()
                return self._finish({"type": "text", "result": result})

            if request_type == "smart":
                result = engine.generate_smart_insights()
                return self._finish({"type": "insights", "result": result})

            if request_type == "anomalies":
                result = engine.detect_anomalies()
                return self._finish({"type": "anomalies", "result": result})

            if request_type == "report":
                result = engine.generate_report()
                return self._finish({"type": "report", "result": result})

            if intent is not None:
                result = engine.aggregate(intent)
                return self._finish({"type": "insight", "result": result})

            return self._finish({"type": "insight", "result": engine.summary_insights()})
        except Exception as e:
            return self._error(str(e))


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
            return self._finish({"report": report})
        except Exception as e:
            return self._error(str(e))

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
