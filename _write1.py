content = '''"""Agent Result Validator.

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

'''
open("agent/result_validator.py", "w").write(content)
print("wrote chunk 1:", len(content), "chars")
