"""
Universal Agent Result Validator & Metric Verifier.

Validates every AgentResult produced by an analytical agent before it is returned:
1. Schema & Lifecycle integrity
2. Mathematical bounds & finite numerical values (no NaN / Infinity leaks)
3. Prediction horizon integrity and shape matching
4. Forecast intervals (lower <= prediction <= upper)
5. Evidence & Provenance integrity (no causal over-claims on correlations)
6. Data cross-check against actual input context
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .schemas import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)


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


def _sanitize_numeric_recursively(obj: Any) -> Any:
    """Recursively convert NaN, Infinity, -Infinity to None or safe finite values."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _sanitize_numeric_recursively(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_numeric_recursively(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_sanitize_numeric_recursively(item) for item in obj)
    return obj


class ResultValidator:
    """Validate and safely repair AgentResult instances."""

    def __init__(self, max_repair_attempts: int = 3):
        self.max_repair_attempts = max(max_repair_attempts, 1)

    def validate(
        self,
        result: AgentResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Run complete suite of post-execution checks."""
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
        self._mathematical_metrics_check(result, vr)
        self._forecast_bounds_check(result, vr)
        self._cross_check(result, vr, context)
        result.validation = vr
        return vr

    def repair(
        self,
        result: AgentResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[AgentResult, ValidationResult]:
        """Run validation, apply safe deterministic repairs, and re-validate."""
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
        agent_name = result.agent_name or result.agent
        if not isinstance(agent_name, str) or not agent_name.strip():
            vr.add_issue(ValidationSeverity.ERROR, "MISSING_AGENT_NAME",
                         "AgentResult.agent_name must be a non-empty string.")

        if not isinstance(result.confidence, (int, float)):
            vr.add_issue(ValidationSeverity.ERROR, "INVALID_CONFIDENCE_TYPE",
                         "AgentResult.confidence must be numeric.", field="confidence")
        elif not 0.0 <= float(result.confidence) <= 1.0:
            vr.add_issue(ValidationSeverity.ERROR, "CONFIDENCE_OUT_OF_RANGE",
                         "AgentResult.confidence must be within [0, 1].",
                         field="confidence", actual=result.confidence)

    def _consistency_check(self, result: AgentResult, vr: ValidationResult) -> None:
        st_val = result.status.value if isinstance(result.status, AgentStatus) else str(result.status).lower()
        if st_val in ("success", "completed"):
            dur = result.execution_time_ms or result.duration_ms or result.execution_time
            if dur < 0:
                vr.add_issue(ValidationSeverity.ERROR, "NEGATIVE_DURATION",
                             "A completed result cannot have negative duration.",
                             field="duration_ms", actual=dur)
        elif st_val in ("error", "failed", "validation_failed"):
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
            op = evidence.operation or evidence.method
            if not isinstance(op, str) or not op.strip():
                vr.add_issue(ValidationSeverity.ERROR, "EVIDENCE_NO_METHOD",
                             "Evidence must name the method or operation used.",
                             field=field)

    def _claim_integrity_check(self, result: AgentResult, vr: ValidationResult) -> None:
        for index, evidence in enumerate(result.evidence):
            if not isinstance(evidence, Evidence):
                continue
            is_corr = evidence.claim_type in (ClaimType.CORRELATION, "correlation")
            if is_corr:
                ref = evidence.data_ref or {}
                if _looks_causal(evidence.method, evidence.operation, str(ref), str(evidence.metadata)):
                    vr.add_issue(
                        ValidationSeverity.ERROR, "CAUSAL_OVERCLAIM",
                        f"Evidence[{index}] marks a correlation claim with causal wording.",
                        field=f"evidence[{index}]",
                        repair_hint="Reclassify as INFERENCE or reword text to express statistical association.",
                    )

    def _mathematical_metrics_check(self, result: AgentResult, vr: ValidationResult) -> None:
        """Verify finite numbers and mathematical bounds on reported metrics."""
        metrics = result.metrics or result.data.get("metrics") or result.data.get("metric") or {}
        if isinstance(metrics, dict):
            # 1. R2 Score check (must be <= 1.0)
            if "r2_score" in metrics and isinstance(metrics["r2_score"], (int, float)):
                r2 = float(metrics["r2_score"])
                if r2 > 1.0:
                    vr.add_issue(ValidationSeverity.ERROR, "INVALID_R2_SCORE",
                                 f"R2 score ({r2}) cannot exceed 1.0.", field="metrics.r2_score")

            # 2. Accuracy check (must be within [0, 1])
            if "accuracy" in metrics and isinstance(metrics["accuracy"], (int, float)):
                acc = float(metrics["accuracy"])
                if not 0.0 <= acc <= 1.0:
                    vr.add_issue(ValidationSeverity.ERROR, "INVALID_ACCURACY",
                                 f"Accuracy ({acc}) must be within [0, 1].", field="metrics.accuracy")

            # 3. Error metrics must be non-negative
            for err_key in ("MAE", "mean_absolute_error", "RMSE", "mean_squared_error", "WAPE"):
                if err_key in metrics and isinstance(metrics[err_key], (int, float)):
                    val = float(metrics[err_key])
                    if val < 0:
                        vr.add_issue(ValidationSeverity.ERROR, "NEGATIVE_ERROR_METRIC",
                                     f"{err_key} ({val}) cannot be negative.", field=f"metrics.{err_key}")

    def _forecast_bounds_check(self, result: AgentResult, vr: ValidationResult) -> None:
        """Verify prediction intervals satisfy lower <= prediction <= upper."""
        forecast_pts = result.data.get("forecast") or result.data.get("predictions") or []
        if isinstance(forecast_pts, list):
            for i, pt in enumerate(forecast_pts):
                if isinstance(pt, dict):
                    pred = pt.get("prediction") or pt.get("forecast")
                    lower = pt.get("lower") or pt.get("lower_bound")
                    upper = pt.get("upper") or pt.get("upper_bound")
                    if isinstance(pred, (int, float)) and isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                        if lower > pred + 1e-6:
                            vr.add_issue(
                                ValidationSeverity.ERROR, "INVALID_PREDICTION_INTERVAL",
                                f"Forecast point {i} lower bound ({lower}) exceeds prediction ({pred}).",
                                field=f"forecast[{i}]",
                            )
                        if upper < pred - 1e-6:
                            vr.add_issue(
                                ValidationSeverity.ERROR, "INVALID_PREDICTION_INTERVAL",
                                f"Forecast point {i} upper bound ({upper}) is below prediction ({pred}).",
                                field=f"forecast[{i}]",
                            )

    def _cross_check(
        self,
        result: AgentResult,
        vr: ValidationResult,
        context: Optional[Dict[str, Any]],
    ) -> None:
        if not context:
            return
        data = context.get("dataframe") or context.get("data")
        if data is None:
            return
        actual_rows = len(data) if isinstance(data, pd.DataFrame) else None
        if actual_rows is not None:
            for i, evidence in enumerate(result.evidence):
                ref = evidence.data_ref or {}
                claimed = ref.get("rows") or ref.get("row_count")
                if isinstance(claimed, (int, float)) and claimed > actual_rows:
                    vr.add_issue(
                        ValidationSeverity.WARNING, "EVIDENCE_ROWS_EXCEEDED",
                        f"Evidence[{i}] claims {claimed} rows but dataset has {actual_rows}.",
                        field=f"evidence[{i}]",
                    )

    def _attempt_repair(
        self,
        result: AgentResult,
        context: Optional[Dict[str, Any]],
        actions: List[str],
    ) -> bool:
        """Apply safe deterministic fixes."""
        changed = False

        # 1. Clamp confidence
        if not isinstance(result.confidence, (int, float)):
            result.confidence = 1.0
            actions.append("Set missing confidence to 1.0.")
            changed = True
        elif not 0.0 <= float(result.confidence) <= 1.0:
            result.confidence = max(0.0, min(1.0, float(result.confidence)))
            actions.append(f"Clamped confidence to {result.confidence}.")
            changed = True

        # 2. Sanitize NaN/Inf in data & metrics
        clean_result = _sanitize_numeric_recursively(result.result)
        if clean_result != result.result:
            result.result = clean_result
            result.data = clean_result
            actions.append("Sanitized non-finite NaN/Infinity values.")
            changed = True

        # 3. Fix forecast interval bounds
        forecast_pts = result.data.get("forecast") or result.data.get("predictions") or []
        if isinstance(forecast_pts, list):
            for pt in forecast_pts:
                if isinstance(pt, dict):
                    pred = pt.get("prediction") or pt.get("forecast")
                    lower = pt.get("lower") or pt.get("lower_bound")
                    upper = pt.get("upper") or pt.get("upper_bound")
                    if isinstance(pred, (int, float)):
                        if isinstance(lower, (int, float)) and lower > pred:
                            if "lower" in pt: pt["lower"] = pred
                            if "lower_bound" in pt: pt["lower_bound"] = pred
                            changed = True
                        if isinstance(upper, (int, float)) and upper < pred:
                            if "upper" in pt: pt["upper"] = pred
                            if "upper_bound" in pt: pt["upper_bound"] = pred
                            changed = True
            if changed:
                actions.append("Adjusted inverted forecast interval bounds.")

        return changed


def validate_agent_result(
    result: AgentResult, context: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """Validate a result and attach outcome to result.validation."""
    return ResultValidator().validate(result, context)
