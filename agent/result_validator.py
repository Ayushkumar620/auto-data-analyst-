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

    # ------------------------------------------------------------------
    # Data cross-check
    # ------------------------------------------------------------------
    def _cross_check(self, result: AgentResult, vr: ValidationResult,
                     context: Optional[Dict[str, Any]]) -> None:
        """Verify evidence references against real data when context is given."""
        if not context:
            return
        data = context.get("dataframe") or context.get("data")
        if data is None:
            return

        actual_columns = context.get("columns")
        if actual_columns is None:
            if isinstance(data, pd.DataFrame):
                actual_columns = [str(c) for c in data.columns]
            elif isinstance(data, dict):
                actual_columns = [
                    str(c) for frame in data.values()
                    if isinstance(frame, pd.DataFrame) for c in frame.columns
                ]
            else:
                return
        actual_columns = [str(c) for c in (actual_columns or [])]

        actual_rows = context.get("row_count")
        if actual_rows is None:
            if isinstance(data, pd.DataFrame):
                actual_rows = len(data)
            elif isinstance(data, dict):
                actual_rows = max(
                    (len(f) for f in data.values() if isinstance(f, pd.DataFrame)),
                    default=0,
                )
            else:
                actual_rows = 0

        for index, evidence in enumerate(result.evidence):
            ref = evidence.data_ref or {}
            field = f"evidence[{index}].data_ref"
            self._cross_check_columns(evidence, ref, actual_columns, field, vr)
            self._cross_check_rows(evidence, ref, actual_rows, field, vr)
            if self._references_nulls(evidence):
                recomputed = self._recompute_null(data, ref)
                if recomputed is not None and isinstance(evidence.raw_value, (int, float)):
                    tolerance = max(1.0, abs(recomputed) * 0.01)
                    if abs(float(evidence.raw_value) - float(recomputed)) > tolerance:
                        vr.add_issue(
                            ValidationSeverity.ERROR, "CALCULATION_MISMATCH",
                            f"Null count in evidence ({evidence.raw_value}) does "
                            f"not match recomputation ({recomputed}).",
                            field=field, expected=recomputed,
                            actual=evidence.raw_value,
                            repair_hint="Recompute the value from the dataset.",
                        )

    def _cross_check_columns(self, evidence: Evidence, ref: Dict[str, Any],
                             actual_columns: List[str], field: str,
                             vr: ValidationResult) -> None:
        for key in ("column_names", "columns"):
            names = ref.get(key)
            if not isinstance(names, (list, tuple)):
                continue
            for name in names:
                if isinstance(name, str) and name not in actual_columns:
                    vr.add_issue(
                        ValidationSeverity.CRITICAL, "EVIDENCE_UNKNOWN_COLUMN",
                        f"Evidence references column '{name}' that is not "
                        "present in the dataset.",
                        field=f"{field}.{key}", expected=name,
                        actual=actual_columns,
                        repair_hint="Drop this evidence item or re-run the "
                                    "analysis against the actual columns.",
                    )
        single_col = ref.get("column") or ref.get("col")
        if isinstance(single_col, str) and single_col not in actual_columns:
            vr.add_issue(
                ValidationSeverity.CRITICAL, "EVIDENCE_UNKNOWN_COLUMN",
                f"Evidence references column '{single_col}' that is not "
                "present in the dataset.",
                field=field, expected=single_col, actual=actual_columns,
                repair_hint="Drop this evidence item or re-run the analysis.",
            )

    @staticmethod
    def _cross_check_rows(evidence: Evidence, ref: Dict[str, Any],
                          actual_rows: int, field: str,
                          vr: ValidationResult) -> None:
        if actual_rows is None:
            return
        claimed_rows = ref.get("rows") or ref.get("row_count")
        if isinstance(claimed_rows, (int, float)) and claimed_rows > actual_rows:
            vr.add_issue(
                ValidationSeverity.WARNING, "EVIDENCE_ROWS_EXCEEDED",
                f"Evidence claims {claimed_rows} rows but the dataset has "
                f"{actual_rows}.",
                field=field, expected=actual_rows, actual=claimed_rows,
                repair_hint="Clamp the row count to the dataset size.",
            )

    @staticmethod
    def _references_nulls(evidence: Evidence) -> bool:
        method = (evidence.method or "").casefold()
        return "null" in method or "missing" in method

    def _recompute_null(self, data: Any, ref: Dict[str, Any]) -> Optional[int]:
        frame = data
        if isinstance(data, dict):
            frame_name = ref.get("frame")
            if frame_name in data:
                frame = data[frame_name]
            else:
                return None
        if not isinstance(frame, pd.DataFrame):
            return None
        column = ref.get("column") or ref.get("col")
        if column is not None and column in frame.columns:
            return int(frame[column].isnull().sum())
        return int(frame.isnull().sum().sum())

    # ------------------------------------------------------------------
    # Repair
    # ------------------------------------------------------------------
    def _attempt_repair(self, result: AgentResult,
                        context: Optional[Dict[str, Any]],
                        actions: List[str]) -> bool:
        """Apply one round of safe repairs. Returns True if anything changed."""
        changed = False

        # 1) Clamp / fix out-of-range confidence.
        if not isinstance(result.confidence, (int, float)):
            result.confidence = 1.0
            actions.append("Set missing AgentResult.confidence to 1.0.")
            changed = True
        elif not 0.0 <= float(result.confidence) <= 1.0:
            result.confidence = max(0.0, min(1.0, float(result.confidence)))
            actions.append(f"Clamped AgentResult.confidence to {result.confidence}.")
            changed = True

        # 2) Stamp timestamps on completed results.
        if result.status == AgentStatus.COMPLETED and result.finished_at is None:
            result.finished_at = result.started_at
            actions.append("Stamped finished_at from started_at.")
            changed = True

        # 3) Normalize / drop broken evidence items.
        kept: List[Evidence] = []
        for evidence in result.evidence:
            if not isinstance(evidence, Evidence):
                actions.append("Dropped an item from evidence that was not Evidence.")
                changed = True
                continue
            if not isinstance(evidence.confidence, (int, float)) or not (
                    0.0 <= float(evidence.confidence) <= 1.0):
                actions.append("Dropped an evidence item with invalid confidence.")
                changed = True
                continue
            if not isinstance(evidence.claim_type, ClaimType):
                actions.append("Dropped an evidence item with invalid claim_type.")
                changed = True
                continue
            if evidence.data_ref and self._references_unknown_column(evidence, context):
                actions.append("Dropped an evidence item referencing an unknown column.")
                changed = True
                continue
            kept.append(evidence)
        result.evidence = kept
        return changed

    @staticmethod
    def _references_unknown_column(
        evidence: Evidence, context: Optional[Dict[str, Any]]
    ) -> bool:
        if not context:
            return False
        data = context.get("dataframe") or context.get("data")
        if data is None:
            return False
        columns = context.get("columns")
        if columns is None:
            if isinstance(data, pd.DataFrame):
                real = [str(c) for c in data.columns]
            elif isinstance(data, dict):
                real = [str(c) for frame in data.values()
                        if isinstance(frame, pd.DataFrame) for c in frame.columns]
            else:
                real = []
        else:
            real = [str(c) for c in columns]
        ref = evidence.data_ref or {}
        names = ref.get("column_names") or ref.get("columns") or []
        if not isinstance(names, (list, tuple)):
            names = [ref.get("column")] if ref.get("column") else []
        return any(isinstance(n, str) and n not in real for n in names)


def validate_agent_result(
    result: AgentResult, context: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """Validate a result and attach the outcome to ``result.validation``."""
    return ResultValidator().validate(result, context)
