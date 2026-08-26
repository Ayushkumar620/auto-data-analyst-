"""
Universal Pre-Execution Validator.

Performs dataset-agnostic, task-specific pre-flight audits before any
analytical agent or algorithm executes.

Ensures:
1. Dataset is readable and non-empty
2. Target exists and exhibits non-zero variance (for predictive tasks)
3. Requested columns exist
4. Task is compatible with dataset modality
5. Task-specific minimum row requirements are met (N>=5 for forecast, N>=10 for ML)
6. Non-destructive validation (missing values in unrelated columns do not invalidate valid rows)
7. Ambiguous requests produce structured NEEDS_CLARIFICATION reports rather than arbitrary assumptions
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from agent.agent_result import AgentError, AgentStatus, ErrorCategory
from agent.canonical_data_layer import CanonicalDataLayer, CanonicalDataset, DatasetRowAudit, SemanticProfile


@dataclass
class PreExecutionValidationReport:
    """Standardized pre-flight audit report."""
    is_valid: bool
    task_type: str
    target_column: Optional[str] = None
    time_column: Optional[str] = None
    audit: Optional[DatasetRowAudit] = None
    error: Optional[AgentError] = None
    needs_clarification: bool = False
    clarification_options: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "task_type": self.task_type,
            "target_column": self.target_column,
            "time_column": self.time_column,
            "needs_clarification": self.needs_clarification,
            "clarification_options": self.clarification_options,
            "warnings": self.warnings,
            "diagnostics": self.diagnostics,
            "error": self.error.to_dict() if self.error else None,
            "audit": self.audit.to_dict() if self.audit else None,
        }


class PreExecutionValidator:
    """Universal pre-execution audit engine across all analytical tasks."""

    @classmethod
    def validate(
        cls,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame], Any],
        task_type: str = "descriptive",
        target: Optional[str] = None,
        time_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        agent_name: str = "PreExecutionValidator",
    ) -> PreExecutionValidationReport:
        """
        Validate input dataset and parameters for a specific analytical task.
        Task types: "forecasting", "regression", "classification", "clustering", "anomaly_detection", "descriptive"
        """
        task_norm = str(task_type).lower().strip()

        # 1. Dataset Existence and Structure Check
        if data is None:
            err = AgentError.create(
                category=ErrorCategory.INPUT_INVALID,
                user_message="No dataset was provided for analysis.",
                suggested_action="Please upload a dataset or select an active table.",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(is_valid=False, task_type=task_norm, error=err)

        if isinstance(data, dict):
            for df in data.values():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    main_df = df
                    break
            else:
                main_df = pd.DataFrame()
        elif isinstance(data, pd.DataFrame):
            main_df = data
        else:
            err = AgentError.create(
                category=ErrorCategory.INPUT_INVALID,
                user_message=f"Unsupported data format '{type(data).__name__}'. Expected a tabular DataFrame.",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(is_valid=False, task_type=task_norm, error=err)

        if main_df.empty or len(main_df) == 0:
            err = AgentError.create(
                category=ErrorCategory.DATA_INVALID,
                user_message="The provided dataset is empty (0 rows).",
                suggested_action="Upload a dataset with valid records.",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(is_valid=False, task_type=task_norm, error=err)

        # 2. Semantic Profile Ingestion
        dataset: CanonicalDataset = CanonicalDataLayer.ingest(main_df)
        profile: SemanticProfile = dataset.profile
        n_orig = dataset.original_rows

        # 3. Explicit Column Existence Check
        if target and target not in main_df.columns:
            err = AgentError.create(
                category=ErrorCategory.TARGET_NOT_FOUND,
                user_message=f"Requested target column '{target}' was not found in dataset.",
                technical_details={"available_columns": list(main_df.columns)},
                suggested_action="Select one of the available columns.",
                field="target",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(is_valid=False, task_type=task_norm, error=err)

        if feature_columns:
            missing_cols = [c for c in feature_columns if c not in main_df.columns]
            if missing_cols:
                err = AgentError.create(
                    category=ErrorCategory.INPUT_INVALID,
                    user_message=f"Requested feature column(s) {missing_cols} do not exist in dataset.",
                    technical_details={"missing_columns": missing_cols, "available_columns": list(main_df.columns)},
                    agent_name=agent_name,
                )
                return PreExecutionValidationReport(is_valid=False, task_type=task_norm, error=err)

        # ----------------------------------------------------------------------
        # 4. Task-Specific Pre-Execution Audits
        # ----------------------------------------------------------------------

        # TASK A: TIME-SERIES FORECASTING
        if task_norm in ("forecast", "forecasting", "time_series", "time_series_forecast"):
            return cls._validate_forecasting(main_df, profile, target, time_column, agent_name)

        # TASK B: SUPERVISED REGRESSION
        elif task_norm in ("regression", "predict_regression", "continuous_prediction"):
            return cls._validate_regression(main_df, profile, target, feature_columns, agent_name)

        # TASK C: SUPERVISED CLASSIFICATION
        elif task_norm in ("classification", "predict_classification", "churn"):
            return cls._validate_classification(main_df, profile, target, feature_columns, agent_name)

        # TASK D: GENERAL SUPERVISED PREDICTION (Auto-detect task from target)
        elif task_norm in ("predict", "prediction", "model"):
            return cls._validate_general_prediction(main_df, profile, target, feature_columns, agent_name)

        # TASK E: CLUSTERING / SEGMENTATION
        elif task_norm in ("clustering", "cluster", "segmentation", "segment"):
            return cls._validate_clustering(main_df, profile, feature_columns, agent_name)

        # TASK F: ANOMALY DETECTION
        elif task_norm in ("anomaly", "anomalies", "anomaly_detection", "outliers"):
            return cls._validate_anomaly_detection(main_df, profile, target, agent_name)

        # TASK G: DESCRIPTIVE / EDA / GENERAL ANALYSIS
        else:
            return PreExecutionValidationReport(
                is_valid=True,
                task_type="descriptive",
                target_column=target,
                time_column=time_column,
                diagnostics={"rows": n_orig, "columns": list(main_df.columns)},
            )

    # --------------------------------------------------------------------------
    # Sub-Validators
    # --------------------------------------------------------------------------

    @classmethod
    def _validate_forecasting(
        cls,
        df: pd.DataFrame,
        profile: SemanticProfile,
        target: Optional[str],
        time_col: Optional[str],
        agent_name: str,
    ) -> PreExecutionValidationReport:
        # Time Column Resolution
        if time_col and time_col in df.columns:
            chosen_time = time_col
        elif profile.datetime_candidates:
            chosen_time = profile.datetime_candidates[0]
        else:
            err = AgentError.create(
                category=ErrorCategory.TIME_COLUMN_NOT_FOUND,
                user_message="Forecasting requires a valid datetime or timestamp column. None was found.",
                technical_details={"columns": list(df.columns)},
                suggested_action="Ensure your dataset contains dates or timestamps for temporal forecasting.",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(is_valid=False, task_type="forecasting", error=err)

        # Target Column Resolution
        if target and target in df.columns:
            chosen_target = target
        elif profile.target_candidates:
            # Pick highest scored regression target
            reg_targets = [t for t in profile.target_candidates if t["type"] == "regression"]
            if reg_targets:
                chosen_target = reg_targets[0]["column"]
            else:
                chosen_target = profile.target_candidates[0]["column"]
        elif profile.numeric_columns:
            chosen_target = profile.numeric_columns[0]
        else:
            err = AgentError.create(
                category=ErrorCategory.TARGET_NOT_FOUND,
                user_message="Forecasting requires at least one continuous numeric metric.",
                suggested_action="Ensure your dataset has a numeric metric to forecast.",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(is_valid=False, task_type="forecasting", error=err)

        # Audit (time, target) pair non-destructively
        audit, target_clean, time_clean = CanonicalDataLayer.audit_dataset_for_target(
            df, target_column=chosen_target, time_column=chosen_time, minimum_required_rows=5
        )

        if audit.valid_rows < 5:
            err = AgentError.create(
                category=ErrorCategory.INSUFFICIENT_DATA,
                user_message=f"Need at least 5 valid historical data points for forecasting. Found {audit.valid_rows}.",
                technical_details=audit.to_dict(),
                suggested_action="Provide a longer historical time series with at least 5 observations.",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(
                is_valid=False, task_type="forecasting", target_column=chosen_target, time_column=chosen_time, audit=audit, error=err
            )

        # Check for constant target
        valid_y = target_clean.dropna()
        if valid_y.nunique() <= 1:
            err = AgentError.create(
                category=ErrorCategory.DATA_INVALID,
                user_message=f"Target column '{chosen_target}' has constant values (0 variance). Cannot forecast flat series.",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(
                is_valid=False, task_type="forecasting", target_column=chosen_target, time_column=chosen_time, audit=audit, error=err
            )

        return PreExecutionValidationReport(
            is_valid=True,
            task_type="forecasting",
            target_column=chosen_target,
            time_column=chosen_time,
            audit=audit,
        )

    @classmethod
    def _validate_regression(
        cls,
        df: pd.DataFrame,
        profile: SemanticProfile,
        target: Optional[str],
        features: Optional[List[str]],
        agent_name: str,
    ) -> PreExecutionValidationReport:
        chosen_target = target or (profile.target_candidates[0]["column"] if profile.target_candidates else (profile.numeric_columns[0] if profile.numeric_columns else None))
        if not chosen_target:
            err = AgentError.create(
                category=ErrorCategory.TARGET_NOT_FOUND,
                user_message="No suitable numeric target column found for regression.",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(is_valid=False, task_type="regression", error=err)

        audit, target_clean, _ = CanonicalDataLayer.audit_dataset_for_target(
            df, target_column=chosen_target, minimum_required_rows=10
        )

        if audit.valid_rows < 10:
            err = AgentError.create(
                category=ErrorCategory.INSUFFICIENT_DATA,
                user_message=f"Need at least 10 valid rows for regression prediction. Found {audit.valid_rows}.",
                technical_details=audit.to_dict(),
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(
                is_valid=False, task_type="regression", target_column=chosen_target, audit=audit, error=err
            )

        # Constant target check
        if target_clean.dropna().nunique() <= 1:
            err = AgentError.create(
                category=ErrorCategory.DATA_INVALID,
                user_message=f"Target column '{chosen_target}' has 0 variance (constant value).",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(
                is_valid=False, task_type="regression", target_column=chosen_target, audit=audit, error=err
            )

        return PreExecutionValidationReport(
            is_valid=True, task_type="regression", target_column=chosen_target, audit=audit
        )

    @classmethod
    def _validate_classification(
        cls,
        df: pd.DataFrame,
        profile: SemanticProfile,
        target: Optional[str],
        features: Optional[List[str]],
        agent_name: str,
    ) -> PreExecutionValidationReport:
        chosen_target = target or (profile.target_candidates[0]["column"] if profile.target_candidates else (profile.categorical_columns[0] if profile.categorical_columns else None))
        if not chosen_target:
            err = AgentError.create(
                category=ErrorCategory.TARGET_NOT_FOUND,
                user_message="No suitable categorical/class target found for classification.",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(is_valid=False, task_type="classification", error=err)

        audit, target_clean, _ = CanonicalDataLayer.audit_dataset_for_target(
            df, target_column=chosen_target, minimum_required_rows=10
        )

        if audit.valid_rows < 10:
            err = AgentError.create(
                category=ErrorCategory.INSUFFICIENT_DATA,
                user_message=f"Need at least 10 valid rows for classification. Found {audit.valid_rows}.",
                technical_details=audit.to_dict(),
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(
                is_valid=False, task_type="classification", target_column=chosen_target, audit=audit, error=err
            )

        n_classes = target_clean.dropna().nunique()
        if n_classes < 2:
            err = AgentError.create(
                category=ErrorCategory.DATA_INVALID,
                user_message=f"Classification target '{chosen_target}' must contain at least 2 distinct classes. Found {n_classes}.",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(
                is_valid=False, task_type="classification", target_column=chosen_target, audit=audit, error=err
            )

        return PreExecutionValidationReport(
            is_valid=True, task_type="classification", target_column=chosen_target, audit=audit
        )

    @classmethod
    def _validate_general_prediction(
        cls,
        df: pd.DataFrame,
        profile: SemanticProfile,
        target: Optional[str],
        features: Optional[List[str]],
        agent_name: str,
    ) -> PreExecutionValidationReport:
        if not target and profile.target_candidates:
            # Check for ambiguity if top candidates are tied
            top_candidates = profile.target_candidates[:3]
            if len(top_candidates) >= 2 and abs(top_candidates[0]["score"] - top_candidates[1]["score"]) < 2.0:
                options = [{"column": t["column"], "type": t["type"], "score": t["score"]} for t in top_candidates]
                return PreExecutionValidationReport(
                    is_valid=False,
                    task_type="prediction",
                    needs_clarification=True,
                    clarification_options=options,
                    error=AgentError.create(
                        category=ErrorCategory.AMBIGUOUS_REQUEST,
                        user_message=f"Multiple target candidates ({[t['column'] for t in top_candidates]}) are equally plausible. Please specify which target to predict.",
                        technical_details={"options": options},
                        suggested_action="Specify target=<column_name> in your command.",
                        agent_name=agent_name,
                    ),
                )
            target = profile.target_candidates[0]["column"]

        chosen_target = target or (profile.numeric_columns[0] if profile.numeric_columns else df.columns[-1])
        is_cat = chosen_target in profile.categorical_columns or df[chosen_target].dtype == object or df[chosen_target].nunique() <= 10

        if is_cat:
            return cls._validate_classification(df, profile, chosen_target, features, agent_name)
        else:
            return cls._validate_regression(df, profile, chosen_target, features, agent_name)

    @classmethod
    def _validate_clustering(
        cls,
        df: pd.DataFrame,
        profile: SemanticProfile,
        features: Optional[List[str]],
        agent_name: str,
    ) -> PreExecutionValidationReport:
        n_features = len(features) if features else (len(profile.numeric_columns) + len(profile.categorical_columns))
        if n_features < 2:
            err = AgentError.create(
                category=ErrorCategory.INSUFFICIENT_DATA,
                user_message="Clustering requires at least 2 distinct feature columns.",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(is_valid=False, task_type="clustering", error=err)

        if len(df) < 5:
            err = AgentError.create(
                category=ErrorCategory.INSUFFICIENT_DATA,
                user_message=f"Clustering requires at least 5 sample observations. Found {len(df)}.",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(is_valid=False, task_type="clustering", error=err)

        return PreExecutionValidationReport(is_valid=True, task_type="clustering")

    @classmethod
    def _validate_anomaly_detection(
        cls,
        df: pd.DataFrame,
        profile: SemanticProfile,
        target: Optional[str],
        agent_name: str,
    ) -> PreExecutionValidationReport:
        if len(df) < 5:
            err = AgentError.create(
                category=ErrorCategory.INSUFFICIENT_DATA,
                user_message=f"Need at least 5 observations for statistical anomaly detection. Found {len(df)}.",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(is_valid=False, task_type="anomaly_detection", error=err)

        if not profile.numeric_columns:
            err = AgentError.create(
                category=ErrorCategory.DATA_INVALID,
                user_message="Anomaly detection requires at least one numeric measure to compute deviations.",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(is_valid=False, task_type="anomaly_detection", error=err)

        # Check if all candidate numeric columns have 0 variance
        non_constant_numeric = [
            c for c in profile.numeric_columns
            if c not in profile.constant_columns and df[c].nunique(dropna=True) > 1
        ]
        if not non_constant_numeric:
            err = AgentError.create(
                category=ErrorCategory.DATA_INVALID,
                user_message="All candidate feature columns have zero variance (constant values). Cannot detect anomalies in uniform data.",
                agent_name=agent_name,
            )
            return PreExecutionValidationReport(is_valid=False, task_type="anomaly_detection", error=err)

        return PreExecutionValidationReport(is_valid=True, task_type="anomaly_detection", target_column=target)
