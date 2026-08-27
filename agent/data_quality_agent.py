"""
Universal Data Quality Gate Agent.

Orchestrates PreExecutionValidator, DataQualityGate, ConfidenceCalculator,
and ResultValidator into the canonical AgentResult lifecycle.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

from agent.agent_result import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)
from agent.base import BaseAgent


class DataQualityAgent(BaseAgent):
    """
    Autonomous Data Quality Gate Agent.
    Evaluates dataset readiness, feature eligibility, target compatibility, and temporal requirements
    before analytical execution.
    """

    name = "Data Quality Agent"
    description = "Evaluates dataset structural validity, feature eligibility, target compatibility, and task-specific readiness."
    role = "data_quality_gate"

    def run(self, task: Dict[str, Any]) -> AgentResult:
        self._start()
        try:
            from agent.pre_execution_validator import PreExecutionValidator
            from agent.confidence_calculator import ConfidenceCalculator
            from agent.result_validator import ResultValidator
            from agent.data_quality_gate import DataQualityGate, QualityGateDecision

            data = task.get("data")
            task_type = task.get("task_type", "eda")
            target = task.get("target")
            features = task.get("features") or task.get("selected_columns")
            time_column = task.get("time_column")
            config = task.get("config") or {}

            # 1. Pre-execution validation
            pre_audit = PreExecutionValidator.validate(
                data,
                task_type="data_quality_gate",
                target=target,
                feature_columns=features,
                agent_name=self.name,
            )
            if not pre_audit.is_valid:
                err = pre_audit.error
                return self._error(
                    message=err.user_message if err else "Data quality pre-execution validation failed.",
                    code=err.code if err else "VALIDATION_FAILURE",
                    category=err.category if err else ErrorCategory.DATA_INVALID,
                    details=err.technical_details if err else {},
                )

            # 2. Execution via DataQualityGate
            gate = DataQualityGate()
            decision: QualityGateDecision = gate.validate(
                df=data,
                task_type=task_type,
                target=target,
                features=features,
                time_column=time_column,
                config=config,
            )

            # 3. Canonical Evidence Generation (ClaimType.OBSERVATION)
            evidence_list: List[Evidence] = [
                self.make_evidence(
                    method="data_quality_gate.decision",
                    data_ref={
                        "status": decision.status,
                        "task_type": decision.task_type,
                        "is_ready": decision.is_ready,
                        "reasons": decision.reasons,
                        "quality_score": decision.quality_score,
                    },
                    confidence=decision.confidence,
                    claim_type=ClaimType.OBSERVATION,
                    raw_value=decision.status,
                )
            ]

            # 4. Confidence Calculation
            n_rows = decision.row_accounting.get("original_rows", 0)
            n_usable_feat = decision.diagnostics.get("usable_features_count", 0)
            conf_rep = ConfidenceCalculator.calculate_quality_gate_confidence(
                status=decision.status,
                n_rows=n_rows,
                n_usable_features=n_usable_feat,
                quality_score=decision.quality_score,
            )

            raw_res = self._finish(
                decision.to_dict(),
                evidence=evidence_list,
                confidence=conf_rep.confidence,
                model_used="DataQualityGate",
            )

            # 5. Result Validation & Invariant Repair
            repaired_res, _ = ResultValidator().repair(raw_res, context={"data": data})
            return repaired_res

        except Exception as e:
            return self._error(str(e), category=ErrorCategory.COMPUTATION)