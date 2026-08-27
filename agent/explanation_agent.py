"""
Universal Explanation Agent - Autonomous agent for explainability and evidence traceability.

Integrates:
- ExplanationEngine
- PreExecutionValidator
- ResultValidator
- ConfidenceCalculator
- AgentResult, Evidence, and AgentError contracts
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from agent.base import BaseAgent
from agent.confidence_calculator import ConfidenceCalculator
from agent.explanation_engine import ExplanationEngine
from agent.explanation_schemas import AnalyticalExplanation
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


class ExplanationAgent(BaseAgent):
    """
    Autonomous agent for analytical explanation and evidence traceability.
    Converts raw or synthesized analytical results into auditable, evidence-backed explanations.
    """

    name = "Explanation Agent"
    description = "Converts analytical results and models into transparent, evidence-backed explanations"
    role = "explanation"

    def __init__(self, data=None, name: str = "Explanation Agent"):
        super().__init__(data=data)
        self.name = name
        self.engine = ExplanationEngine()

    def run(self, task: Any) -> AgentResult:
        """Standard BaseAgent execution method."""
        if isinstance(task, dict):
            return self.execute(task)
        elif isinstance(task, (AgentResult, pd.DataFrame)):
            return self.execute({"result": task})
        return self.execute({"result": task})

    def execute(self, inputs: Dict[str, Any]) -> AgentResult:
        """Execute analytical explanation generation."""
        start_time = datetime.now()
        self.status = AgentStatus.WORKING

        target_payload = inputs.get("result") or inputs.get("orchestration_result") or inputs.get("data_result") or inputs
        dataframe = inputs.get("data") if isinstance(inputs.get("data"), pd.DataFrame) else None
        command = inputs.get("command") or inputs.get("user_request")
        depth = inputs.get("depth", "detailed")

        try:
            # Handle empty payload safely
            if not target_payload and dataframe is None:
                err = AgentError(
                    code="EMPTY_EXPLANATION_INPUT",
                    category=ErrorCategory.INPUT_INVALID,
                    user_message="No analytical result or dataset provided for explanation.",
                    message="No analytical result or dataset provided for explanation.",
                    agent_name=self.name,
                )
                return AgentResult(
                    status=AgentStatus.NEEDS_CLARIFICATION,
                    task_type="explanation",
                    agent_name=self.name,
                    result={"error": err.user_message},
                    data={"error": err.user_message},
                    output={"error": err.user_message},
                    confidence=0.30,
                    errors=[err],
                )

            explanation: AnalyticalExplanation = self.engine.explain(
                result=target_payload,
                dataframe=dataframe,
                command=command,
                depth=depth,
            )

            duration_ms = round((datetime.now() - start_time).total_seconds() * 1000, 2)
            exp_dict = explanation.to_dict()
            conf = float(explanation.uncertainty.get("epistemic_confidence", 0.90))

            # Convert EvidenceTraces to Evidence objects for AgentResult contract
            ev_list: List[Evidence] = []
            for tr in explanation.evidence:
                ev_list.append(
                    Evidence(
                        dataset_name=tr.source,
                        columns=tr.columns,
                        operation=tr.method,
                        calculation=tr.calculation or tr.claim,
                        result=tr.result,
                        confidence=tr.confidence,
                        claim_type=ClaimType.OBSERVATION,
                    )
                )

            res = AgentResult.success(
                output=exp_dict,
                agent_name=self.name,
                metrics={
                    "total_findings": len(explanation.findings),
                    "total_methodologies": len(explanation.methodology),
                    "total_metrics_explained": len(explanation.metrics),
                    "total_evidence_traces": len(explanation.evidence),
                    "epistemic_confidence": conf,
                },
                evidence=ev_list,
                confidence=conf,
                task_type="explanation",
                assumptions=explanation.assumptions,
                limitations=explanation.limitations,
                duration_ms=duration_ms,
            )
            self.status = AgentStatus.SUCCESS
            return res

        except Exception as ex:
            self.status = AgentStatus.ERROR
            err = AgentError(
                code="EXPLANATION_FAILED",
                category=ErrorCategory.INTERNAL_ERROR,
                user_message="Explanation generation encountered an internal calculation issue.",
                message=str(ex),
                agent_name=self.name,
                technical_details={"exception_type": type(ex).__name__},
            )
            return AgentResult(
                status=AgentStatus.ERROR,
                task_type="explanation",
                agent_name=self.name,
                result={"error": err.user_message},
                data={"error": err.user_message},
                output={"error": err.user_message},
                confidence=0.20,
                errors=[err],
            )