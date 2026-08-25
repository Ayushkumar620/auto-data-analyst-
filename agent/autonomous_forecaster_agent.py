"""
Autonomous Forecaster and Scenario Agent.

Orchestrates time-series forecasting, candidate benchmarking, probabilistic intervals,
and counterfactual What-If scenario simulations.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from agent.autonomous_forecast_engine import AutonomousForecastEngine
from agent.base import BaseAgent
from agent.forecasting_schemas import (
    ForecastRequest,
    ForecastResult,
    ScenarioComparison,
    ScenarioResult,
    WhatIfRequest,
)
from agent.schemas import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)
from agent.what_if_scenario_engine import WhatIfScenarioEngine


class AutonomousForecasterAgent(BaseAgent):
    """
    Autonomous Forecaster & Scenario Agent executing time-series forecasting,
    prediction intervals, and What-If counterfactual simulations.
    """
    name = "Autonomous Forecaster Agent"
    role = "time_series_forecaster"
    description = "Autonomous time-series forecasting, candidate model benchmarking, and counterfactual What-If scenario modeling."

    def __init__(self, data: Optional[Any] = None):
        super().__init__(data=data)
        self.forecast_engine = AutonomousForecastEngine()
        self.scenario_engine = WhatIfScenarioEngine()

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        """Run time-series forecasting pipeline."""
        return self.forecast_engine.run_forecast(request)

    def scenario(self, request: WhatIfRequest) -> ScenarioResult:
        """Run counterfactual scenario simulation."""
        return self.scenario_engine.simulate_scenario(request)

    def compare_scenarios(
        self,
        df: pd.DataFrame,
        target: str,
        scenarios_spec: Dict[str, Dict[str, Any]],
    ) -> ScenarioComparison:
        """Compare multiple What-If scenarios."""
        return self.scenario_engine.compare_scenarios(df, target, scenarios_spec)

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """Standardized BaseAgent execution interface."""
        self._start()
        data = task.get("data", self.data)
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame()

        command = task.get("command") or task.get("user_message") or "Forecast metric"
        target = task.get("target_column") or task.get("target")
        time_col = task.get("time_column") or task.get("date_column")
        horizon = task.get("horizon") or task.get("forecast_horizon") or 6

        # Check for What-If Scenario Command
        is_what_if = bool(
            re.search(r"\b(what if|what happens|scenario|best (and|&) worst|increases? by|decreases? by|drops? by|grows? by)\b", command, re.I)
            or task.get("mode") == "scenario"
            or task.get("changed_variables")
        )

        try:
            if is_what_if:
                # Multi-scenario check ("best and worst", "scenarios")
                if re.search(r"\b(best (and|&) worst|all scenarios|compare scenarios|scenarios)\b", command, re.I):
                    scenarios_spec = {
                        "Optimistic (+15%)": {"pct": 0.15},
                        "Expected (+5%)": {"pct": 0.05},
                        "Pessimistic (-10%)": {"pct": -0.10},
                    }
                    comp_res = self.compare_scenarios(df, target=target or "target", scenarios_spec=scenarios_spec)
                    return self._finish(
                        result=comp_res.to_dict(),
                        evidence=comp_res.evidence,
                        confidence=0.90,
                        metadata={"operation": "scenario_comparison", "scenarios_count": len(comp_res.scenarios)},
                    )

                # Single scenario extraction
                pct_match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", command)
                pct_val = (float(pct_match.group(1)) / 100.0) if pct_match else 0.10

                # Segment match
                seg_match = re.search(r"\b(North|South|East|West|Product\s*[A-Z]|Segment\s*[A-Z])\b", command, re.I)
                seg_name = seg_match.group(1) if seg_match else None

                changed_vars = task.get("changed_variables") or (
                    {"segment": seg_name, "pct": pct_val} if seg_name else {"pct": pct_val}
                )

                req = WhatIfRequest(
                    dataset=df,
                    target=target,
                    scenario_name=task.get("scenario_name", f"Scenario ({pct_val*100:+.1f}%)"),
                    changed_variables=changed_vars,
                )
                scen_res = self.scenario(req)
                return self._finish(
                    result=scen_res.to_dict(),
                    evidence=scen_res.evidence,
                    confidence=0.90,
                    metadata={"operation": "what_if_simulation", "target": scen_res.target_metric},
                )

            # Standard Time-Series Forecast
            req = ForecastRequest(
                dataset=df,
                time_column=time_col,
                target_column=target,
                forecast_horizon=int(horizon),
                confidence_level=task.get("confidence_level", 0.80),
                frequency=task.get("frequency"),
            )
            fc_res = self.forecast(req)

            if fc_res.status == "NOT_SUPPORTED":
                return self._finish(
                    result=fc_res.to_dict(),
                    evidence=[],
                    confidence=0.10,
                    metadata={"status": "NOT_SUPPORTED", "reasons": fc_res.reasons},
                )

            return self._finish(
                result=fc_res.to_dict(),
                evidence=fc_res.evidence,
                confidence=fc_res.confidence,
                metadata={"model_name": fc_res.model_name, "horizon": fc_res.forecast_horizon},
            )

        except Exception as exc:
            return self._error(f"Autonomous forecasting failed: {str(exc)}", category=ErrorCategory.COMPUTATION)
