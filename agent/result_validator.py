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
            if result.finished_at is None:
                vr.add_issue(ValidationSeverity.ERROR, "MISSING_FINISHED_AT",
                             "A completed result must have finished_at set.")
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
            if not isinstance(evidence.confidence, (int, float)) or not (0.0 <= float(evidence.confidence) <= 1.0):
                vr.add_issue(ValidationSeverity.ERROR, "EVIDENCE_CONFIDENCE_RANGE",
                             "Evidence confidence must be within [0, 1].", field=field)
            if not isinstance(evidence.claim_type, ClaimType):
                vr.add_issue(ValidationSeverity.ERROR, "INVALID_CLAIM_TYPE",
                             "Evidence must carry a valid ClaimType enum.", field=field)
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
                        ValidationSeverity.ERROR, "CORRELATION_AS_CAUSATION",
                        f"Evidence[{index}] marks a correlation claim with causal wording.",
                        field=f"evidence[{index}]",
                        repair_hint="Reclassify as INFERENCE or reword text to express statistical association.",
                    )

    def _mathematical_metrics_check(self, result: AgentResult, vr: ValidationResult) -> None:
        """Verify finite numbers and mathematical bounds on reported metrics."""
        def _contains_non_finite(obj: Any) -> bool:
            if isinstance(obj, float):
                return math.isnan(obj) or math.isinf(obj)
            elif isinstance(obj, np.floating):
                val = float(obj)
                return math.isnan(val) or math.isinf(val)
            elif isinstance(obj, dict):
                return any(_contains_non_finite(v) for v in obj.values())
            elif isinstance(obj, (list, tuple)):
                return any(_contains_non_finite(item) for item in obj)
            return False

        if _contains_non_finite(result.result) or _contains_non_finite(result.data) or _contains_non_finite(result.metrics):
            vr.add_issue(
                ValidationSeverity.ERROR, "NON_FINITE_NUMERIC_VALUE",
                "Agent result or metrics contains non-finite NaN or Infinity value.",
                field="result",
                repair_hint="Sanitize non-finite values into None or bounded numbers.",
            )

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

            # 4. Clustering metrics
            if "silhouette_score" in metrics and isinstance(metrics["silhouette_score"], (int, float)):
                sil = float(metrics["silhouette_score"])
                if not -1.0 <= sil <= 1.0:
                    vr.add_issue(ValidationSeverity.ERROR, "INVALID_SILHOUETTE_SCORE",
                                 f"Silhouette score ({sil}) must be within [-1, 1].", field="metrics.silhouette_score")
            if "calinski_harabasz_score" in metrics and isinstance(metrics["calinski_harabasz_score"], (int, float)):
                ch = float(metrics["calinski_harabasz_score"])
                if ch < 0:
                    vr.add_issue(ValidationSeverity.ERROR, "INVALID_CH_SCORE",
                                 f"Calinski-Harabasz score ({ch}) must be non-negative.", field="metrics.calinski_harabasz_score")
            if "davies_bouldin_score" in metrics and isinstance(metrics["davies_bouldin_score"], (int, float)):
                db = float(metrics["davies_bouldin_score"])
                if db < 0:
                    vr.add_issue(ValidationSeverity.ERROR, "INVALID_DB_SCORE",
                                 f"Davies-Bouldin score ({db}) must be non-negative.", field="metrics.davies_bouldin_score")

            # 5. Statistical relationship checks in relationships list
            relationships = result.data.get("relationships") or result.result.get("relationships") or []
            if isinstance(relationships, list):
                for i, rel in enumerate(relationships):
                    if isinstance(rel, dict):
                        stat = rel.get("statistic")
                        if isinstance(stat, (int, float)) and rel.get("pair_type") == "numeric_numeric":
                            if not -1.0 <= float(stat) <= 1.0:
                                vr.add_issue(ValidationSeverity.ERROR, "INVALID_CORRELATION_STATISTIC",
                                             f"Correlation statistic ({stat}) at index {i} must be within [-1, 1].",
                                             field=f"relationships[{i}].statistic")
                        p_val = rel.get("p_value")
                        if isinstance(p_val, (int, float)):
                            if not 0.0 <= float(p_val) <= 1.0:
                                vr.add_issue(ValidationSeverity.ERROR, "INVALID_P_VALUE",
                                             f"P-value ({p_val}) at index {i} must be within [0, 1].",
                                             field=f"relationships[{i}].p_value")
                        adj_p = rel.get("adjusted_p_value")
                        if isinstance(adj_p, (int, float)):
                            if not 0.0 <= float(adj_p) <= 1.0:
                                vr.add_issue(ValidationSeverity.ERROR, "INVALID_ADJUSTED_P_VALUE",
                                             f"Adjusted p-value ({adj_p}) at index {i} must be within [0, 1].",
                                             field=f"relationships[{i}].adjusted_p_value")

            # 6. EDA & Data Quality Checks
            data_quality = result.data.get("data_quality") or result.result.get("data_quality")
            if isinstance(data_quality, dict):
                qs = data_quality.get("quality_score")
                if isinstance(qs, (int, float)):
                    if not 0.0 <= float(qs) <= 1.0:
                        vr.add_issue(ValidationSeverity.ERROR, "INVALID_QUALITY_SCORE",
                                     f"Quality score ({qs}) must be within [0, 1].",
                                     field="data_quality.quality_score")
                comps = data_quality.get("components")
                if isinstance(comps, dict):
                    for comp_name, comp_val in comps.items():
                        if isinstance(comp_val, (int, float)):
                            if not 0.0 <= float(comp_val) <= 1.0:
                                vr.add_issue(ValidationSeverity.ERROR, "INVALID_QUALITY_COMPONENT",
                                             f"Quality component {comp_name} ({comp_val}) must be within [0, 1].",
                                             field=f"data_quality.components.{comp_name}")

            # 7. Numeric stats quantile checks (min <= median <= max, q25 <= median <= q75)
            stats_sec = result.data.get("statistics") or result.result.get("statistics") or {}
            if isinstance(stats_sec, dict):
                num_stats = stats_sec.get("numeric") or {}
                if isinstance(num_stats, dict):
                    for col_name, st in num_stats.items():
                        if isinstance(st, dict):
                            c_min = st.get("min")
                            c_med = st.get("median")
                            c_max = st.get("max")
                            c_q25 = st.get("q25")
                            c_q75 = st.get("q75")
                            if all(isinstance(x, (int, float)) for x in (c_min, c_med, c_max)):
                                if c_min > c_med + 1e-6 or c_med > c_max + 1e-6:
                                    vr.add_issue(ValidationSeverity.ERROR, "INVALID_QUANTILE_ORDER",
                                                 f"Column '{col_name}' quantiles violate min ({c_min}) <= median ({c_med}) <= max ({c_max}).",
                                                 field=f"statistics.numeric.{col_name}")
                            if all(isinstance(x, (int, float)) for x in (c_q25, c_med, c_q75)):
                                if c_q25 > c_med + 1e-6 or c_med > c_q75 + 1e-6:
                                    vr.add_issue(ValidationSeverity.ERROR, "INVALID_IQR_ORDER",
                                                 f"Column '{col_name}' IQR violates q25 ({c_q25}) <= median ({c_med}) <= q75 ({c_q75}).",
                                                 field=f"statistics.numeric.{col_name}")

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
        data = context.get("dataframe") if context.get("dataframe") is not None else context.get("data")
        if data is None:
            return
        
        columns = context.get("columns")
        if columns is None:
            if isinstance(data, pd.DataFrame):
                real_cols = [str(c) for c in data.columns]
            elif isinstance(data, dict):
                real_cols = [str(c) for frame in data.values() if isinstance(frame, pd.DataFrame) for c in frame.columns]
            else:
                real_cols = []
        else:
            real_cols = [str(c) for c in columns]

        actual_rows = len(data) if isinstance(data, pd.DataFrame) else None

        for i, evidence in enumerate(result.evidence):
            if not isinstance(evidence, Evidence):
                continue
            ref = evidence.data_ref or {}
            field = f"evidence[{i}]"

            # Check unknown columns
            names = ref.get("column_names") or ref.get("columns") or []
            if isinstance(names, str):
                names = [names]
            elif not isinstance(names, (list, tuple)):
                names = []
            single_col = ref.get("column") or ref.get("col")
            if isinstance(single_col, str):
                names = list(names) + [single_col]
            
            if real_cols and any(isinstance(n, str) and n not in real_cols for n in names):
                vr.add_issue(
                    ValidationSeverity.ERROR, "EVIDENCE_UNKNOWN_COLUMN",
                    f"Evidence[{i}] references unknown column(s) not in dataset.",
                    field=field,
                )

            # Check row count
            claimed = ref.get("rows") or ref.get("row_count")
            if actual_rows is not None and isinstance(claimed, (int, float)) and claimed > actual_rows:
                vr.add_issue(
                    ValidationSeverity.WARNING, "EVIDENCE_ROWS_EXCEEDED",
                    f"Evidence[{i}] claims {claimed} rows but dataset has {actual_rows}.",
                    field=field,
                )

            # Null statistics spot check
            op_str = (evidence.operation or evidence.method or "").casefold()
            if "isnull" in op_str or "null" in op_str or "missing" in op_str:
                if isinstance(data, pd.DataFrame) and evidence.raw_value is not None:
                    col = ref.get("column") or ref.get("col")
                    if col and col in data.columns:
                        actual_nulls = int(data[col].isnull().sum())
                    else:
                        actual_nulls = int(data.isnull().sum().sum())

                    val_to_check = None
                    if isinstance(evidence.raw_value, (int, float)):
                        val_to_check = float(evidence.raw_value)
                    elif isinstance(evidence.raw_value, str):
                        try:
                            val_to_check = float(evidence.raw_value)
                        except Exception:
                            pass
                    elif isinstance(evidence.raw_value, dict):
                        mc = evidence.raw_value.get("total_missing_cells") or evidence.raw_value.get("missing_count") or evidence.raw_value.get("null_count")
                        if isinstance(mc, (int, float)):
                            val_to_check = float(mc)

                    if val_to_check is not None and val_to_check != float(actual_nulls):
                        vr.add_issue(
                            ValidationSeverity.ERROR, "CALCULATION_MISMATCH",
                            f"Evidence null count ({evidence.raw_value}) != actual ({actual_nulls}).",
                            field=field,
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

        # 2. Stamp finished_at if missing
        if result.finished_at is None and (result.started_at or result.timestamp):
            result.finished_at = result.started_at or result.timestamp
            actions.append("Stamped missing finished_at.")
            changed = True

        # 3. Sanitize NaN/Inf in data & metrics
        clean_result = _sanitize_numeric_recursively(result.result)
        if clean_result != result.result:
            result.result = clean_result
            result.data = clean_result
            actions.append("Sanitized non-finite NaN/Infinity values.")
            changed = True

        # 4. Clean broken evidence
        kept_ev: List[Evidence] = []
        for ev in result.evidence:
            if not isinstance(ev, Evidence):
                actions.append("Dropped non-Evidence item.")
                changed = True
                continue
            if not isinstance(ev.confidence, (int, float)) or not (0.0 <= float(ev.confidence) <= 1.0):
                actions.append("Dropped evidence with invalid confidence.")
                changed = True
                continue
            if not isinstance(ev.claim_type, ClaimType):
                actions.append("Dropped evidence with invalid claim_type.")
                changed = True
                continue
            if context:
                data = context.get("dataframe") if context.get("dataframe") is not None else context.get("data")
                if isinstance(data, pd.DataFrame):
                    ref = ev.data_ref or {}
                    names = ref.get("column_names")
                    if names is None:
                        cols_val = ref.get("columns")
                        if isinstance(cols_val, (list, tuple)):
                            names = list(cols_val)
                        elif isinstance(cols_val, str):
                            names = [cols_val]
                        else:
                            names = []
                    elif isinstance(names, str):
                        names = [names]
                    elif not isinstance(names, (list, tuple)):
                        names = []
                    single_col = ref.get("column") or ref.get("col")
                    if isinstance(single_col, str):
                        names = list(names) + [single_col]
                    if any(isinstance(n, str) and n not in data.columns for n in names):
                        actions.append("Dropped evidence referencing hallucinated column.")
                        changed = True
                        continue
            kept_ev.append(ev)
        if len(kept_ev) != len(result.evidence):
            result.evidence = kept_ev
            changed = True

        # 5. Fix forecast interval bounds
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
