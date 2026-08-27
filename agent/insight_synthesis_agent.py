"""
Insight Synthesis Agent - Autonomous orchestration wrapper for cross-agent insight synthesis.

Integrates:
- InsightSynthesisEngine
- CanonicalDataLayer & SemanticProfile
- PreExecutionValidator
- ConfidenceCalculator
- ResultValidator
- AgentResult & Evidence
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from agent.base import BaseAgent
from agent.canonical_data_layer import CanonicalDataLayer, SemanticProfile
from agent.confidence_calculator import ConfidenceCalculator
from agent.insight_synthesis_engine import InsightSynthesisEngine, SynthesisReport
from agent.pre_execution_validator import PreExecutionValidator
from agent.result_validator import ResultValidator
from agent.schemas import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)


class InsightSynthesisAgent(BaseAgent):
    """
    Autonomous agent for multi-agent analytical insight synthesis.
    Consumes validated outputs from multiple specialized agents and produces
    an evidence-backed executive narrative.
    """

    def __init__(self, name: str = "Insight Synthesis Agent"):
        super().__init__(name=name)
        self.engine = InsightSynthesisEngine()
        self.role = "insight_synthesis"

    def execute(self, inputs: Dict[str, Any]) -> AgentResult:
        """Execute multi-agent analytical insight synthesis."""
        start_time = datetime.now()
        self.state["status"] = AgentStatus.WORKING.value
        self.state["warnings"] = []
        self.state["errors"] = []

        orchestration_result = inputs.get("orchestration_result") or inputs.get("result") or inputs
        dataframe = inputs.get("data") if isinstance(inputs.get("data"), pd.DataFrame) else None
        command = inputs.get("command") or inputs.get("user_request")

        # Ingest SemanticProfile if dataframe provided
        profile = None
        if dataframe is not None:
            profile = CanonicalDataLayer.ingest(dataframe).profile

        try:
            report: SynthesisReport = self.engine.synthesize(
                orchestration_result=orchestration_result,
                dataframe=dataframe,
                profile=profile,
                command=command,
            )

            report_dict = report.to_dict()
            duration_ms = round((datetime.now() - start_time).total_seconds() * 1000, 2)

            res = AgentResult.success(
                output=report_dict,
                agent_name=self.name,
                metrics={
                    "total_insights": len(report.key_insights) + len(report.relationships) + len(report.data_quality_findings),
                    "contradictions_count": len(report.contradictions),
                    "overall_confidence": report.overall_confidence,
                },
                evidence=report.evidence,
                confidence=report.overall_confidence,
                task_type="insight_synthesis",
                assumptions=[
                    "Synthesized insights are strictly observational and derived from validated multi-agent outputs.",
                    "No causal relationships are inferred from observational correlations.",
                ],
                limitations=report.limitations,
                duration_ms=duration_ms,
            )

            # Repair and validate
            repaired = ResultValidator.repair(res)
            self.state["status"] = AgentStatus.COMPLETED.value
            return repaired

        except Exception as exc:
            duration_ms = round((datetime.now() - start_time).total_seconds() * 1000, 2)
            self.state["status"] = AgentStatus.ERROR.value
            err = AgentError.create(
                category=ErrorCategory.COMPUTATION,
                user_message=f"Insight synthesis failed: {str(exc)}",
                agent_name=self.name,
                code="SYNTHESIS_ERROR",
            )
            return AgentResult.error(
                error=err.user_message,
                code=err.code,
                category=err.category,
                agent_name=self.name,
                task_type="insight_synthesis",
                duration_ms=duration_ms,
                errors=[err],
            )