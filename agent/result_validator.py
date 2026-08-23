"""Agent Result Validator.

Validates every AgentResult produced by an agent before it is passed
downstream. Checks:

  1. Schema          - required fields present, confidence in [0, 1]
  2. Consistency     - completed results carry finished_at/duration, error
                       results carry at least one AgentError
  3. Evidence model  - every Evidence item is well formed and bounded
  4. Claim integrity - a CORRELATION evidence item must NEVER be worded (in
                       its metadata/description) as causation
  5. Data cross-check- when the input context is supplied, references to
                       columns / frames / row counts are verified against the
                       real data and null-statistics are spot-recomputed

The validator never silently guesses.  It reports WARNING for situations that
could be explained by another pipeline stage, and ERROR / CRITICAL only when a
claim is provably wrong or out of contract.  ``ResultValidator.repair`` tries
fixes that are safe (clamp confidence, default claim_type, stamp timestamps),
and marks a result as ``repaired`` with the list of repair_actions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .schemas import (
    AgentResult,
    AgentStatus,
    ClaimType,
    Evidence,
    ValidationResult,
    ValidationSeverity,
)


# If a CORRELATION evidence item carries any of these in its meta/description
# it over-claims causality and must be flagged.
_CAUSAL_WORDS = (
    "cause", "causes", "caused", "because of", "drives", "driven by",
    "leads to", "leading to", "results in", "resulting in", "attributed to",
    "is the reason", "duplicated by",
)


def _text(value: Any) -> str:
    return str(value or "").casefold()


def _looks_causal(*texts: Any) -> bool:
    blob = " ".join(_text(t) for t in texts)
    return any(word in blob for word in _CAUSAL_WORDS)


class ResultValidator:
    """Validate and, when possible, repair AgentResult instances."""

    def __init__(self, max_repair_attempts: int = 3):
        self.max_repair_attempts = max(max_repair_attempts, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate(
        self,
        result: AgentResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate a result.

        context supports the keys:
            - dataframe | data: input pd.DataFrame or dict[str, DataFrame]
            - columns: list[str] of the actual column names
            - row_count: int actual row count of the frame
        """
        if not isinstance(result, AgentResult):
            raise TypeError(
                "ResultValidator.validate() expects an AgentResult, "
                f"got {type(result).__name__}"
            )
        vr = ValidationResult(passed=True)
        self._schema_check(result, vr)
        self._consistency_check(result, vr)
        self._evidence_check(result, vr)
        self._claim_integrity_check(result, vr)
        self._cross_check(result, vr, context)
        result.validation = vr
        return vr

    def repair(
        self,
        result: AgentResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[AgentResult, ValidationResult]:
        """Run validation, apply safe repairs, and re-validate.

        Returns (result, validation_result).  The result object is mutated in
        place (clamped confidence, normalized claim types, stamped timestamps,
        stripped evidence that references columns that do not exist).
        """
        initial = self.validate(result, context)
        if initial.passed:
            return result, initial

        actions: List[str] = []
        for _ in range(self.max_repair_attempts):
            if self.validate(result, context).passed:
                break
            if not self._attempt_repair(result, context, actions):
                break

        final = self.validate(result, context)
        final.repaired = bool(actions)
        final.repair_actions = actions
        result.validation = final
        return result, final


    # ------------------------------------------------------------------
    # Validation steps
    # ------------------------------------------------------------------
    def _schema_check(self, result: AgentResult, vr: ValidationResult) -> None:
        if not isinstance(result.agent, str) or not result.agent.strip():
            vr.add_issue(ValidationSeverity.ERROR, "MISSING_AGENT_NAME",
                         "AgentResult.agent must be a non-empty string.")
        if not isinstance(result.role, str) or not result.role.strip():
            vr.add_issue(ValidationSeverity.ERROR, "MISSING_ROLE",
                         "AgentResult.role must be a non-empty string.")
        if not isinstance(result.agent_id, str) or not result.agent_id.strip():
            vr.add_issue(ValidationSeverity.ERROR, "MISSING_AGENT_ID",
                         "AgentResult.agent_id must be a non-empty string.")
        if not isinstance(result.status, AgentStatus):
            vr.add_issue(ValidationSeverity.ERROR, "INVALID_STATUS",
                         "AgentResult.status must be an AgentStatus.")
        if not isinstance(result.output, dict):
            vr.add_issue(ValidationSeverity.ERROR, "INVALID_OUTPUT",
                         "AgentResult.output must be a dict.",
                         field="output", actual=type(result.output).__name__)
        if not isinstance(result.confidence, (int, float)):
            vr.add_issue(ValidationSeverity.ERROR, "INVALID_CONFIDENCE_TYPE",
                         "AgentResult.confidence must be numeric.",
                         field="confidence")
        elif not 0.0 <= float(result.confidence) <= 1.0:
            vr.add_issue(ValidationSeverity.ERROR, "CONFIDENCE_OUT_OF_RANGE",
                         "AgentResult.confidence must be within [0, 1].",
                         field="confidence", actual=result.confidence)

    def _consistency_check(self, result: AgentResult, vr: ValidationResult) -> None:
        if result.status == AgentStatus.COMPLETED:
            if result.finished_at is None:
                vr.add_issue(ValidationSeverity.ERROR, "MISSING_FINISHED_AT",
                             "A completed result must have finished_at set.")
            if result.duration_ms < 0:
                vr.add_issue(ValidationSeverity.ERROR, "NEGATIVE_DURATION",
                             "A completed result cannot have a negative duration_ms.",
                             field="duration_ms", actual=result.duration_ms)
            if not result.output:
                vr.add_issue(ValidationSeverity.WARNING, "EMPTY_COMPLETED",
                             "A completed result carries no output.")
        elif result.status == AgentStatus.ERROR:
            if not result.errors:
                vr.add_issue(ValidationSeverity.WARNING, "ERROR_WITHOUT_ERRORS",
                             "An error result should carry at least one AgentError.",
                             field="errors")
def _evidence_check(self, result: AgentResult, vr: ValidationResult) -> None:
        for index, evidence in enumerate(result.evidence):
            field = f"evidence[{index}]"
            if not isinstance(evidence, Evidence):
                vr.add_issue(ValidationSeverity.ERROR, "INVALID_EVIDENCE_TYPE",
                             "Every evidence item must be an Evidence instance.",
                             field=field)
                continue
            if not isinstance(evidence.method, str) or not evidence.method.strip():
                vr.add_issue(ValidationSeverity.ERROR, "EVIDENCE_NO_METHOD",
                             "Evidence must name the method used.",
                             field=field)
            if not isinstance(evidence.source, str) or not evidence.source.strip():
                vr.add_issue(ValidationSeverity.ERROR, "EVIDENCE_NO_SOURCE",
                             "Evidence must name its source agent.",
                             field=field)
            if not isinstance(evidence.data_ref, dict):
                vr.add_issue(ValidationSeverity.WARNING, "EVIDENCE_NO_DATA_REF",
                             "Evidence.data_ref should describe the data used.",
                             field=field)
            if not isinstance(evidence.confidence, (int, float)):
                vr.add_issue(ValidationSeverity.ERROR, "EVIDENCE_BAD_CONFIDENCE",
                             "Evidence confidence must be numeric.",
                             field=field)
            elif not 0.0 <= float(evidence.confidence) <= 1.0:
                vr.add_issue(ValidationSeverity.ERROR, "EVIDENCE_CONFIDENCE_RANGE",
                             "Evidence confidence must be within [0, 1].",
                             field=field, actual=evidence.confidence,
                             repair_hint="Clamp to [0, 1].")
            if not isinstance(evidence.claim_type, ClaimType):
                vr.add_issue(ValidationSeverity.ERROR, "INVALID_CLAIM_TYPE",
                             "Evidence.claim_type must be a ClaimType.",
                             field=field, actual=evidence.claim_type)

    def _claim_integrity_check(self, result: AgentResult, vr: ValidationResult) -> None:
        for index, evidence in enumerate(result.evidence):
            if evidence.claim_type != ClaimType.CORRELATION:
                continue
            meta = evidence.metadata or {}
            probe = [meta.get("description"), meta.get("interpretation"),
                     meta.get("claim")]
            if _looks_causal(*probe):
                vr.add_issue(
                    ValidationSeverity.ERROR, "CORRELATION_AS_CAUSATION",
                    "A CORRELATION evidence item must not imply causation.",
                    field=f"evidence[{index}]",
                    actual=" ".join(_text(p) for p in probe),
                    repair_hint="Rewrite as an INFERENCE with explicit causal "
                                "framing and lower confidence.",
                )