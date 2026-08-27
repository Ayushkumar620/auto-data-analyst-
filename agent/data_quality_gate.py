"""
Universal Data Quality Gate & Pre-Analysis Validation Architecture.

Single source of truth for:
1. Task-aware dataset validation (Regression, Classification, Forecasting, Clustering, Anomaly, Statistical, EDA, Transformation)
2. QualityGateDecision model (READY, READY_WITH_WARNINGS, NEEDS_TRANSFORMATION, NEEDS_CLARIFICATION, BLOCKED)
3. Issue Severity Classification (INFO, WARNING, ERROR, BLOCKER)
4. Row Accounting (original_rows, parsed_rows, target_valid_rows, feature_valid_rows, time_valid_rows, analysis_rows, dropped_rows, drop_reasons)
5. Feature, Target, and Temporal Eligibility Matrices
6. Target Leakage & Duplicate Column Handling
7. Actionable Recommendations & Quality vs Task Suitability Separation
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from agent.canonical_data_layer import CanonicalDataLayer, CanonicalDataset, SemanticProfile
from agent.eda_engine import EDAEngine
from agent.transformation_engine import TransformationEngine, TransformationPlan, TransformationState


# ---------------------------------------------------------------------------
# Enums & Decision Data Structures
# ---------------------------------------------------------------------------

class QualityGateStatus(str, Enum):
    READY = "READY"
    READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
    NEEDS_TRANSFORMATION = "NEEDS_TRANSFORMATION"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    BLOCKED = "BLOCKED"


class IssueSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    BLOCKER = "BLOCKER"


@dataclass
class QualityIssue:
    code: str
    severity: str  # INFO, WARNING, ERROR, BLOCKER
    message: str
    affected_columns: List[str] = field(default_factory=list)
    suggested_action: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityGateDecision:
    """Universal, explainable pre-analysis quality gate decision."""
    status: str  # READY, READY_WITH_WARNINGS, NEEDS_TRANSFORMATION, NEEDS_CLARIFICATION, BLOCKED
    task_type: str
    is_ready: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    required_actions: List[str] = field(default_factory=list)
    optional_actions: List[str] = field(default_factory=list)
    row_accounting: Dict[str, Any] = field(default_factory=dict)
    feature_eligibility: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    target_eligibility: Optional[Dict[str, Any]] = None
    temporal_eligibility: Optional[Dict[str, Any]] = None
    leakage_risks: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    confidence: float = 0.95
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# DataQualityGate (Single Source of Truth)
# ---------------------------------------------------------------------------

class DataQualityGate:
    """
    Universal Data Quality Gate evaluating dataset suitability before analytical execution.
    Orchestrates CanonicalDataLayer, EDAEngine, and TransformationEngine without duplication.
    """

    SUPPORTED_TASKS = {
        "regression",
        "classification",
        "forecasting",
        "anomaly_detection",
        "clustering",
        "statistical_relationship",
        "correlation",
        "eda",
        "transformation",
    }

    def __init__(self):
        self.eda_engine = EDAEngine()
        self.transformation_engine = TransformationEngine()

    def validate(
        self,
        df: Union[pd.DataFrame, Any],
        task_type: str,
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
        time_column: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> QualityGateDecision:
        """
        Evaluate dataset compatibility and data quality for a specific analytical task.
        Guarantees non-mutation of input DataFrame and produces structured diagnostics.
        """
        raw_df = self._extract_dataframe(df)
        task_norm = (task_type or "eda").lower().strip()
        if task_norm in ("predict", "prediction"):
            task_norm = "regression"
        elif task_norm in ("forecast", "time_series"):
            task_norm = "forecasting"
        elif task_norm in ("anomaly", "anomalies", "outliers"):
            task_norm = "anomaly_detection"
        elif task_norm in ("cluster", "segmentation"):
            task_norm = "clustering"
        elif task_norm in ("stats", "hypothesis", "relationships"):
            task_norm = "statistical_relationship"

        cfg = config or {}

        # ----------------------------------------------------------------------
        # Phase 1: Structural Dataset Ingestion & Validation
        # ----------------------------------------------------------------------
        if raw_df is None or not isinstance(raw_df, pd.DataFrame):
            return self._blocked_decision(
                task_norm,
                code="INVALID_DATA_STRUCTURE",
                message="Input dataset must be a valid pandas DataFrame or tabular record structure.",
            )

        n_orig_rows, n_orig_cols = raw_df.shape
        if n_orig_rows == 0:
            return self._blocked_decision(
                task_norm,
                code="EMPTY_DATASET",
                message="Dataset contains 0 rows. Cannot perform analysis on an empty dataset.",
                row_accounting={"original_rows": 0, "parsed_rows": 0, "analysis_rows": 0, "dropped_rows": 0, "drop_reasons": {}},
            )

        if n_orig_cols == 0:
            return self._blocked_decision(
                task_norm,
                code="ZERO_COLUMNS",
                message="Dataset contains 0 columns. Analysis requires at least one attribute column.",
                row_accounting={"original_rows": n_orig_rows, "parsed_rows": n_orig_rows, "analysis_rows": 0, "dropped_rows": n_orig_rows, "drop_reasons": {"all_rows": "zero columns"}},
            )

        if raw_df.isna().all().all():
            return self._blocked_decision(
                task_norm,
                code="ALL_MISSING_DATA",
                message="Dataset contains 100% missing/null values across all columns and rows.",
                row_accounting={"original_rows": n_orig_rows, "parsed_rows": n_orig_rows, "analysis_rows": 0, "dropped_rows": n_orig_rows, "drop_reasons": {"all_rows": "100% missing"}},
            )

        # Handle Duplicate Column Names safely (Disambiguate internally with mapping)
        col_counts: Dict[str, int] = {}
        disambiguated_cols: List[str] = []
        col_provenance: Dict[str, str] = {}
        has_duplicate_cols = False

        for col in raw_df.columns:
            c_str = str(col)
            if c_str in col_counts:
                has_duplicate_cols = True
                col_counts[c_str] += 1
                new_col = f"{c_str}__duplicate_{col_counts[c_str]}"
                disambiguated_cols.append(new_col)
                col_provenance[new_col] = c_str
            else:
                col_counts[c_str] = 0
                disambiguated_cols.append(c_str)
                col_provenance[c_str] = c_str

        work_df = raw_df.copy()
        if has_duplicate_cols:
            work_df.columns = disambiguated_cols

        # ----------------------------------------------------------------------
        # Phase 2: Ingest into CanonicalDataLayer & SemanticProfile
        # ----------------------------------------------------------------------
        dataset: CanonicalDataset = CanonicalDataLayer.ingest(work_df)
        profile: SemanticProfile = dataset.profile

        # Run EDA Engine for overall dataset quality score
        eda_summary = self.eda_engine.profile(work_df)
        dataset_quality_score = eda_summary.get("data_quality", {}).get("quality_score", 1.0)

        issues: List[QualityIssue] = []
        reasons: List[str] = []
        warnings: List[str] = []
        required_actions: List[str] = []
        optional_actions: List[str] = []
        recommendations: List[str] = []
        leakage_risks: List[Dict[str, Any]] = []

        if has_duplicate_cols:
            dup_names = [c for c, count in col_counts.items() if count > 0]
            issues.append(QualityIssue(
                code="DUPLICATE_COLUMN_NAMES",
                severity=IssueSeverity.WARNING.value,
                message=f"Dataset contains duplicate column names: {dup_names}. Internally disambiguated with provenance preservation.",
                affected_columns=dup_names,
                suggested_action="Rename duplicate columns with unique descriptive identifiers.",
            ))
            warnings.append(f"Duplicate column names disambiguated: {dup_names}")
            optional_actions.append("Ensure column headers in data source are unique.")

        # ----------------------------------------------------------------------
        # Phase 3: Build Feature Eligibility Matrix
        # ----------------------------------------------------------------------
        feature_eligibility: Dict[str, Dict[str, Any]] = {}
        all_cols = list(work_df.columns)
        candidate_cols = [c for c in all_cols if c != target and c != time_column]

        for col in all_cols:
            s = work_df[col]
            n_valid = int(s.notna().sum())
            n_null = n_orig_rows - n_valid
            missing_rate = round(n_null / n_orig_rows, 4) if n_orig_rows > 0 else 1.0
            n_uniq = int(s.nunique(dropna=True))
            uniq_ratio = round(n_uniq / max(1, n_valid), 4)

            sem_role = "numeric" if col in profile.numeric_columns else ("datetime" if col in profile.datetime_candidates else "categorical")
            if col in profile.identifier_columns:
                sem_role = "identifier"
            elif col in profile.constant_columns:
                sem_role = "constant"

            coercion_rate = 1.0
            if pd.api.types.is_numeric_dtype(s):
                coercion_rate = 1.0
            elif sem_role == "datetime":
                dt_coerced = CanonicalDataLayer.coerce_datetime_series(s)
                coercion_rate = round(float(dt_coerced.notna().sum() / max(1, n_valid)), 4)
            else:
                num_coerced = CanonicalDataLayer.coerce_numeric_series(s)
                coercion_rate = round(float(num_coerced.notna().sum() / max(1, n_valid)), 4)

            # Determine usability
            is_usable = True
            excl_reason = None
            trans_req = False

            if n_valid == 0:
                is_usable = False
                excl_reason = "100% missing values"
            elif missing_rate > 0.60:
                is_usable = False
                excl_reason = f"High missing rate ({round(missing_rate*100, 1)}%) exceeds 60% threshold"
            elif n_uniq <= 1:
                is_usable = False
                excl_reason = "Constant feature with zero variance"
            elif col in profile.identifier_columns and n_orig_rows >= 10:
                is_usable = False
                excl_reason = "Excluded by default as unique database identifier / key"
            is_natively_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            if not is_natively_numeric or missing_rate > 0.0 or (sem_role in ("categorical", "datetime") and task_norm in ("regression", "clustering", "anomaly_detection")):
                trans_req = True

            # Leakage risk check
            leakage_flag = False
            if target and target in work_df.columns and col != target:
                # Check for exact duplicate values with target
                if s.equals(work_df[target]):
                    leakage_flag = True
                    leakage_risks.append({
                        "column": col,
                        "risk_type": "exact_target_duplicate",
                        "severity": IssueSeverity.ERROR.value,
                        "description": f"Column '{col}' has identical values to target '{target}' (potential target leakage).",
                    })

            feature_eligibility[col] = {
                "source_column": col_provenance.get(col, col),
                "usable": is_usable,
                "excluded": not is_usable,
                "reason": excl_reason,
                "semantic_role": sem_role,
                "missing_rate": missing_rate,
                "uniqueness_ratio": uniq_ratio,
                "coercion_rate": coercion_rate,
                "transformation_required": trans_req,
                "leakage_risk": leakage_flag,
                "downstream_compatibility": {
                    "regression": is_usable and sem_role in ("numeric", "categorical", "datetime"),
                    "clustering": is_usable and sem_role in ("numeric", "categorical"),
                    "anomaly_detection": is_usable and sem_role in ("numeric", "categorical"),
                    "forecasting": is_usable and sem_role in ("numeric", "datetime"),
                },
            }

        # ----------------------------------------------------------------------
        # Phase 4: Target & Temporal Eligibility
        # ----------------------------------------------------------------------
        target_eligibility: Optional[Dict[str, Any]] = None
        target_valid_rows = n_orig_rows

        if target:
            if target not in work_df.columns:
                target_eligibility = {
                    "target_name": target,
                    "usable": False,
                    "rejection_reason": f"Target column '{target}' does not exist in dataset.",
                }
            else:
                ts = work_df[target]
                t_valid = int(ts.notna().sum())
                t_missing = n_orig_rows - t_valid
                t_uniq = int(ts.nunique(dropna=True))
                target_valid_rows = t_valid

                t_type = "numeric" if (pd.api.types.is_numeric_dtype(ts) and not pd.api.types.is_bool_dtype(ts)) else "categorical"
                t_var = float(ts.dropna().astype(float).var()) if (t_type == "numeric" and t_valid > 1) else 0.0

                class_dist = None
                if t_type == "categorical" or t_uniq <= 10:
                    class_dist = {str(k): int(v) for k, v in ts.value_counts(dropna=True).to_dict().items()}

                t_usable = True
                t_rej = None
                if t_valid == 0:
                    t_usable = False
                    t_rej = "Target contains 100% missing values."
                elif t_uniq <= 1:
                    t_usable = False
                    t_rej = f"Target column '{target}' has only {t_uniq} distinct value (constant target)."

                target_eligibility = {
                    "target_name": target,
                    "detected_type": t_type,
                    "valid_count": t_valid,
                    "missing_count": t_missing,
                    "unique_count": t_uniq,
                    "variance": round(t_var, 6) if not math.isnan(t_var) else 0.0,
                    "class_distribution": class_dist,
                    "coercion_rate": 1.0,
                    "usable": t_usable,
                    "rejection_reason": t_rej,
                }

        temporal_eligibility: Optional[Dict[str, Any]] = None
        time_valid_rows = n_orig_rows

        if time_column or task_norm == "forecasting":
            chosen_time_col = time_column
            if not chosen_time_col and profile.datetime_candidates:
                chosen_time_col = profile.datetime_candidates[0]

            if chosen_time_col and chosen_time_col in work_df.columns:
                ts_col = work_df[chosen_time_col]
                coerced_dt = CanonicalDataLayer.coerce_datetime_series(ts_col)
                dt_valid = int(coerced_dt.notna().sum())
                time_valid_rows = dt_valid
                dt_rate = round(dt_valid / max(1, n_orig_rows), 4)

                dt_s_valid = coerced_dt.dropna()
                min_t = dt_s_valid.min().isoformat() if len(dt_s_valid) > 0 else None
                max_t = dt_s_valid.max().isoformat() if len(dt_s_valid) > 0 else None
                dup_t = int(dt_s_valid.duplicated().sum())

                is_chronological = bool(dt_s_valid.is_monotonic_increasing)
                inferred_freq = "unknown"
                if len(dt_s_valid) >= 3:
                    try:
                        inferred_freq = str(pd.infer_freq(dt_s_valid.drop_duplicates()))
                    except Exception:
                        inferred_freq = "irregular"

                time_usable = dt_rate >= 0.60 and dt_valid >= 5
                time_rej = None
                if not time_usable:
                    time_rej = f"Temporal column parseability rate ({round(dt_rate*100, 1)}%) or valid count ({dt_valid}) is insufficient."

                temporal_eligibility = {
                    "time_column": chosen_time_col,
                    "parse_success_rate": dt_rate,
                    "valid_count": dt_valid,
                    "min_time": min_t,
                    "max_time": max_t,
                    "duplicate_count": dup_t,
                    "inferred_frequency": inferred_freq,
                    "irregularity_score": 0.0 if inferred_freq not in ("irregular", "unknown", "None") else 0.5,
                    "chronological_order": is_chronological,
                    "missing_periods": 0,
                    "usable": time_usable,
                    "rejection_reason": time_rej,
                }
            elif chosen_time_col:
                temporal_eligibility = {
                    "time_column": chosen_time_col,
                    "usable": False,
                    "rejection_reason": f"Specified temporal column '{chosen_time_col}' not found in dataset.",
                }
            else:
                temporal_eligibility = {
                    "time_column": None,
                    "usable": False,
                    "rejection_reason": "No valid datetime/temporal column found or specified for time-series forecasting.",
                }

        # ----------------------------------------------------------------------
        # Phase 5: Task-Specific Eligibility Checks
        # ----------------------------------------------------------------------
        gate_status = QualityGateStatus.READY
        is_ready = True
        usable_features_list = [c for c, meta in feature_eligibility.items() if meta["usable"] and c != target and c != time_column]

        # 1. Regression
        if task_norm == "regression":
            if not target:
                gate_status = QualityGateStatus.NEEDS_CLARIFICATION
                is_ready = False
                reasons.append("Regression requires a target variable.")
                required_actions.append("Specify a continuous target column for regression.")
                issues.append(QualityIssue(
                    code="TARGET_MISSING",
                    severity=IssueSeverity.BLOCKER.value,
                    message="Target column was not specified for regression task.",
                    suggested_action="Select a numeric target variable.",
                ))
            elif target_eligibility and not target_eligibility["usable"]:
                gate_status = QualityGateStatus.BLOCKED
                is_ready = False
                reasons.append(f"Target '{target}' is not usable: {target_eligibility['rejection_reason']}")
                issues.append(QualityIssue(
                    code="TARGET_INVALID",
                    severity=IssueSeverity.BLOCKER.value,
                    message=target_eligibility["rejection_reason"] or "Target invalid",
                    affected_columns=[target],
                ))
            elif target_eligibility and target_eligibility["detected_type"] != "numeric":
                gate_status = QualityGateStatus.BLOCKED
                is_ready = False
                reasons.append(f"Target column '{target}' is categorical. Regression requires a continuous numeric target.")
                issues.append(QualityIssue(
                    code="TARGET_TYPE_MISMATCH",
                    severity=IssueSeverity.BLOCKER.value,
                    message=f"Target column '{target}' is categorical with {target_eligibility['unique_count']} classes. Consider classification.",
                    affected_columns=[target],
                    suggested_action="Use classification task instead of regression.",
                ))

            if len(usable_features_list) == 0:
                gate_status = QualityGateStatus.BLOCKED
                is_ready = False
                reasons.append("No usable explanatory features available for regression.")
                issues.append(QualityIssue(
                    code="NO_USABLE_FEATURES",
                    severity=IssueSeverity.BLOCKER.value,
                    message="All candidate features are identifiers, constants, or 100% missing.",
                ))

            if n_orig_rows < 5:
                gate_status = QualityGateStatus.BLOCKED
                is_ready = False
                reasons.append(f"Dataset has {n_orig_rows} rows. Regression requires at least 5 observations.")
                issues.append(QualityIssue(
                    code="INSUFFICIENT_ROWS",
                    severity=IssueSeverity.BLOCKER.value,
                    message=f"Sample size (N={n_orig_rows}) is too small for statistical regression modeling.",
                ))

        # 2. Classification
        elif task_norm == "classification":
            if not target:
                gate_status = QualityGateStatus.NEEDS_CLARIFICATION
                is_ready = False
                reasons.append("Classification requires a target class variable.")
                required_actions.append("Specify a categorical target column for classification.")
                issues.append(QualityIssue(
                    code="TARGET_MISSING",
                    severity=IssueSeverity.BLOCKER.value,
                    message="Target column was not specified for classification task.",
                ))
            elif target_eligibility and not target_eligibility["usable"]:
                gate_status = QualityGateStatus.BLOCKED
                is_ready = False
                reasons.append(f"Target '{target}' is not usable: {target_eligibility['rejection_reason']}")
                issues.append(QualityIssue(
                    code="TARGET_INVALID",
                    severity=IssueSeverity.BLOCKER.value,
                    message=target_eligibility["rejection_reason"] or "Target invalid",
                    affected_columns=[target],
                ))
            elif target_eligibility and target_eligibility["unique_count"] < 2:
                gate_status = QualityGateStatus.BLOCKED
                is_ready = False
                reasons.append(f"Target column '{target}' has only {target_eligibility['unique_count']} class. Classification requires at least 2 distinct classes.")
                issues.append(QualityIssue(
                    code="TARGET_CONSTANT_CLASS",
                    severity=IssueSeverity.BLOCKER.value,
                    message=f"Target has only {target_eligibility['unique_count']} class.",
                    affected_columns=[target],
                ))

            if len(usable_features_list) == 0:
                gate_status = QualityGateStatus.BLOCKED
                is_ready = False
                reasons.append("No usable explanatory features available for classification.")
                issues.append(QualityIssue(
                    code="NO_USABLE_FEATURES",
                    severity=IssueSeverity.BLOCKER.value,
                    message="All candidate features are identifiers, constants, or 100% missing.",
                ))

        # 3. Forecasting
        elif task_norm == "forecasting":
            if not temporal_eligibility or not temporal_eligibility["usable"]:
                gate_status = QualityGateStatus.BLOCKED
                is_ready = False
                reasons.append("Forecasting requires a valid temporal/datetime column with chronological observations.")
                issues.append(QualityIssue(
                    code="TEMPORAL_COLUMN_UNAVAILABLE",
                    severity=IssueSeverity.BLOCKER.value,
                    message=temporal_eligibility["rejection_reason"] if temporal_eligibility else "No temporal column detected.",
                    suggested_action="Specify a parseable datetime column.",
                ))
            elif target and target_eligibility and not target_eligibility["usable"]:
                gate_status = QualityGateStatus.BLOCKED
                is_ready = False
                reasons.append(f"Forecasting target '{target}' is invalid: {target_eligibility['rejection_reason']}")
                issues.append(QualityIssue(
                    code="TARGET_INVALID",
                    severity=IssueSeverity.BLOCKER.value,
                    message=target_eligibility["rejection_reason"] or "Target invalid",
                    affected_columns=[target],
                ))
            elif n_orig_rows < 5:
                gate_status = QualityGateStatus.BLOCKED
                is_ready = False
                reasons.append(f"Dataset has {n_orig_rows} periods. Time-series forecasting requires at least 5 chronological points.")
                issues.append(QualityIssue(
                    code="INSUFFICIENT_TIME_PERIODS",
                    severity=IssueSeverity.BLOCKER.value,
                    message="Insufficient time series observations.",
                ))

        # 4. Clustering & Anomaly Detection
        elif task_norm in ("clustering", "anomaly_detection"):
            if len(usable_features_list) == 0:
                gate_status = QualityGateStatus.BLOCKED
                is_ready = False
                reasons.append(f"{task_norm.capitalize()} requires at least one non-constant, non-identifier feature.")
                issues.append(QualityIssue(
                    code="NO_USABLE_FEATURES",
                    severity=IssueSeverity.BLOCKER.value,
                    message="Dataset contains only identifiers, constants, or empty columns.",
                ))
            elif task_norm == "clustering" and len(usable_features_list) < 2 and n_orig_rows < 10:
                gate_status = QualityGateStatus.READY_WITH_WARNINGS
                warnings.append("Low feature count (1 feature) limits cluster multidimensional geometry.")
            elif n_orig_rows < 4:
                gate_status = QualityGateStatus.BLOCKED
                is_ready = False
                reasons.append(f"{task_norm.capitalize()} requires at least 4 observations.")
                issues.append(QualityIssue(
                    code="INSUFFICIENT_ROWS",
                    severity=IssueSeverity.BLOCKER.value,
                    message="Too few observations for statistical clustering / anomaly detection.",
                ))

        # 5. Statistical Relationships
        elif task_norm == "statistical_relationship":
            if len(usable_features_list) < 2 and not target:
                gate_status = QualityGateStatus.BLOCKED
                is_ready = False
                reasons.append("Relationship analysis requires at least two distinct usable variables.")
                issues.append(QualityIssue(
                    code="INSUFFICIENT_VARIABLES",
                    severity=IssueSeverity.BLOCKER.value,
                    message="Need at least 2 non-constant features to evaluate statistical correlation/association.",
                ))

        # 6. EDA & Transformation
        elif task_norm in ("eda", "transformation"):
            # EDA is permissive: profiles even with warnings
            if n_orig_rows < 3:
                gate_status = QualityGateStatus.READY_WITH_WARNINGS
                warnings.append("Very small sample size (N < 3) limits descriptive distribution inferences.")

        # ----------------------------------------------------------------------
        # Phase 6: Check If Transformation Is Required
        # ----------------------------------------------------------------------
        any_trans_required = any(meta["transformation_required"] for meta in feature_eligibility.values() if meta["usable"])
        if is_ready and any_trans_required and task_norm in ("regression", "classification", "clustering", "anomaly_detection"):
            gate_status = QualityGateStatus.NEEDS_TRANSFORMATION
            reasons.append("Feature matrix contains missing values, dirty numeric strings, or categorical variables requiring preprocessing.")
            required_actions.append("Run TransformationEngine pre-flight pipeline prior to model training.")
            recommendations.append("Apply automated imputation, one-hot encoding, and robust scaling.")

        if is_ready and len(warnings) > 0 and gate_status == QualityGateStatus.READY:
            gate_status = QualityGateStatus.READY_WITH_WARNINGS

        # ----------------------------------------------------------------------
        # Phase 7: Row Accounting
        # ----------------------------------------------------------------------
        analysis_rows = min(n_orig_rows, target_valid_rows, time_valid_rows)
        dropped_rows = n_orig_rows - analysis_rows
        drop_reasons: Dict[str, int] = {}
        if n_orig_rows - target_valid_rows > 0:
            drop_reasons["missing_target"] = n_orig_rows - target_valid_rows
        if n_orig_rows - time_valid_rows > 0:
            drop_reasons["invalid_timestamps"] = n_orig_rows - time_valid_rows

        row_accounting = {
            "original_rows": n_orig_rows,
            "parsed_rows": n_orig_rows,
            "target_valid_rows": target_valid_rows,
            "feature_valid_rows": n_orig_rows,
            "time_valid_rows": time_valid_rows,
            "analysis_rows": analysis_rows,
            "dropped_rows": dropped_rows,
            "drop_reasons": drop_reasons,
        }

        # ----------------------------------------------------------------------
        # Phase 8: Recommendations & Action Synthesis
        # ----------------------------------------------------------------------
        if any_trans_required:
            recommendations.append("Automated imputation and scaling recommended for optimal numerical convergence.")
        if any(meta["uniqueness_ratio"] > 0.95 and meta["semantic_role"] == "identifier" for meta in feature_eligibility.values()):
            recommendations.append("High-cardinality identifier fields automatically excluded from predictive features.")
        if temporal_eligibility and temporal_eligibility.get("duplicate_count", 0) > 0:
            recommendations.append("Duplicate timestamps detected; consider aggregation or deduplication.")

        # Confidence Calculation
        conf = 0.95
        if gate_status == QualityGateStatus.BLOCKED:
            conf = 0.35
        elif gate_status == QualityGateStatus.NEEDS_CLARIFICATION:
            conf = 0.50
        elif gate_status == QualityGateStatus.NEEDS_TRANSFORMATION:
            conf = 0.85
        elif gate_status == QualityGateStatus.READY_WITH_WARNINGS:
            conf = 0.88

        return QualityGateDecision(
            status=gate_status.value,
            task_type=task_norm,
            is_ready=is_ready,
            reasons=reasons if reasons else ["Dataset passes pre-analysis quality gate checks."],
            warnings=warnings,
            issues=[iss.to_dict() for iss in issues],
            required_actions=required_actions,
            optional_actions=optional_actions,
            row_accounting=row_accounting,
            feature_eligibility=feature_eligibility,
            target_eligibility=target_eligibility,
            temporal_eligibility=temporal_eligibility,
            leakage_risks=leakage_risks,
            recommendations=recommendations,
            quality_score=round(float(dataset_quality_score), 4),
            confidence=round(float(conf), 4),
            diagnostics={
                "original_shape": [n_orig_rows, n_orig_cols],
                "usable_features_count": len(usable_features_list),
                "excluded_features_count": n_orig_cols - len(usable_features_list),
                "has_duplicate_columns": has_duplicate_cols,
            },
        )

    def _blocked_decision(
        self,
        task_type: str,
        code: str,
        message: str,
        row_accounting: Optional[Dict[str, Any]] = None,
    ) -> QualityGateDecision:
        return QualityGateDecision(
            status=QualityGateStatus.BLOCKED.value,
            task_type=task_type,
            is_ready=False,
            reasons=[message],
            warnings=[],
            issues=[QualityIssue(code=code, severity=IssueSeverity.BLOCKER.value, message=message).to_dict()],
            required_actions=["Provide a structurally valid tabular dataset."],
            optional_actions=[],
            row_accounting=row_accounting or {"original_rows": 0, "parsed_rows": 0, "analysis_rows": 0, "dropped_rows": 0, "drop_reasons": {}},
            feature_eligibility={},
            target_eligibility=None,
            temporal_eligibility=None,
            leakage_risks=[],
            recommendations=["Verify dataset encoding and structure."],
            quality_score=0.0,
            confidence=0.30,
            diagnostics={"error_code": code},
        )

    def _extract_dataframe(self, data: Any) -> Optional[pd.DataFrame]:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            for df in data.values():
                if isinstance(df, pd.DataFrame):
                    return df
        return None