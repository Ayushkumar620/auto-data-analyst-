"""
Planner Agent - Orchestrates the multi-agent workflow.

The Planner receives a user request and decides which specialized agents to
invoke and in what order. It mirrors the design of a startup product where a
central "Planner Agent" routes tasks across Cleaner, EDA, Insight, ML,
Forecast, and Report agents.

It can run a single atomic task or a full autonomous pipeline:
  upload -> clean -> EDA -> insights -> report
"""
import time
from typing import Any, Dict, Optional

import pandas as pd

from .result_validator import ResultValidator
from .agents import (
    DataLoadingAgent,
    AnalysisAgent,
    VisualizationAgent,
    PredictionAgent,
    ForecastAgent,
    CleaningAgent,
    InsightAgent,
    ReportAgent,
)


class PlannerAgent:
    """Coordinates specialized agents to fulfill a user request."""

    # Map logical request types to the agent that handles them
    REQUEST_MAP = {
        "clean": {
            "action": "clean",
            "agent": CleaningAgent,
            "task": lambda data, req: {"data": data},
        },
        "summary": {
            "action": "summary",
            "agent": AnalysisAgent,
            "task": lambda data, req: {"data": data, "request": "summary"},
        },
        "describe": {
            "action": "describe",
            "agent": AnalysisAgent,
            "task": lambda data, req: {"data": data, "request": "describe"},
        },
        "stats": {
            "action": "describe",
            "agent": AnalysisAgent,
            "task": lambda data, req: {"data": data, "request": "describe"},
        },
        "nulls": {
            "action": "nulls",
            "agent": AnalysisAgent,
            "task": lambda data, req: {"data": data, "request": "nulls"},
        },
        "correlation": {
            "action": "correlation",
            "agent": AnalysisAgent,
            "task": lambda data, req: {"data": data, "request": "correlation"},
        },
        "head": {
            "action": "head",
            "agent": AnalysisAgent,
            "task": lambda data, req: {"data": data, "request": "head"},
        },
        "unique": {
            "action": "unique",
            "agent": AnalysisAgent,
            "task": lambda data, req: {"data": data, "request": "unique"},
        },
        "chart": {
            "action": "chart",
            "agent": VisualizationAgent,
            "task": lambda data, req: {
                "data": data,
                "chart_type": req.get("chart_type", "auto"),
                "x": req.get("x"),
                "y": req.get("y"),
            },
        },
        "predict": {
            "action": "predict",
            "agent": PredictionAgent,
            "task": lambda data, req: {"data": data, "target": req.get("target")},
        },
        "forecast": {
            "action": "forecast",
            "agent": ForecastAgent,
            "task": lambda data, req: {
                "data": data,
                "target": req.get("target"),
                "periods": req.get("periods", 5),
            },
        },
        "insights": {
            "action": "insights",
            "agent": InsightAgent,
            "task": lambda data, req: {"data": data, "type": "smart"},
        },
        "anomalies": {
            "action": "anomalies",
            "agent": InsightAgent,
            "task": lambda data, req: {"data": data, "type": "anomalies"},
        },
        "report": {
            "action": "report",
            "agent": InsightAgent,
            "task": lambda data, req: {"data": data, "type": "report"},
        },
        "text": {
            "action": "text",
            "agent": InsightAgent,
            "task": lambda data, req: {"data": data, "type": "text"},
        },
    }

    def __init__(self, data=None, validator: ResultValidator = None):
        self.data = data
        self._validator = validator or ResultValidator()

    def run_agent(self, request, data=None):
        """Execute a single atomic request using the appropriate agent."""
        if data is None:
            data = self.data
        req = request or {}
        action = req.get("action", "summary")
        entry = self.REQUEST_MAP.get(action)

        if entry is None:
            # Fall back to summary for unknown actions
            entry = self.REQUEST_MAP["summary"]

        agent = entry["agent"]()
        task = entry["task"](data, req)
        agent._start()
        try:
            result = agent.run(task)
        except Exception as e:
            result = agent._error(str(e))
        context = self._validation_context(data)
        # Validate + repair every AgentResult before it leaves the planner so
        # downstream consumers never see an unvalidated/unrepaired result.
        self._validator.repair(result, context)
        return result

    @staticmethod
    def _validation_context(
        data: Any,
    ) -> Optional[Dict[str, Any]]:
        """Build the validation context the ResultValidator uses to cross-check
        evidence against the real dataset."""
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            return {
                "dataframe": data,
                "columns": [str(c) for c in data.columns],
                "row_count": len(data),
            }
        if isinstance(data, dict):
            frames = [f for f in data.values() if isinstance(f, pd.DataFrame)]
            columns = sorted({str(c) for f in frames for c in f.columns})
            row_count = max((len(f) for f in frames), default=0)
            return {"dataframe": data, "columns": columns, "row_count": row_count}
        return None

    def run_pipeline(self, data=None, steps=None):
        """Run an autonomous pipeline of agents in sequence.

        steps: list of action strings, e.g. ["clean", "summary", "insights", "report"].
        Defaults to a full pipeline: clean -> summary -> insights.
        """
        if data is None:
            data = self.data
        if steps is None:
            steps = ["clean", "summary", "insights"]

        outputs = []
        current_data = data
        for action in steps:
            req = {"action": action}
            out = self.run_agent(req, current_data)
            outputs.append(out)
            # If cleaning, carry the cleaned data forward for subsequent steps
            if action == "clean" and out.get("status") == "completed":
                reports = out.get("output", {}).get("reports", [])
                if reports and isinstance(reports, list) and "cleaned_data" in reports[0]:
                    import pandas as pd
                    current_data = pd.DataFrame(reports[0]["cleaned_data"])

        # Optionally generate a narrative report from outputs
        report_agent = ReportAgent()
        report_out = report_agent.run({"agent_outputs": outputs, "request": "pipeline"})
        return {
            "steps": steps,
            "agents": outputs,
            "report": report_out,
            "duration_ms": round(sum(o.get("duration_ms", 0) for o in outputs), 2),
        }

    def describe_pipeline(self):
        """Return a description of the available orchestration capabilities."""
        return {
            "orchestrator": "Planner Agent",
            "available_actions": list(self.REQUEST_MAP.keys()),
            "pipeline_format": '{"action": "chart", "chart_type": "bar", "x": "col"}',
            "example_pipeline": ["clean", "summary", "insights", "forecast"],
        }
