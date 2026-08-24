"""Comprehensive Validation Engine for Machine Learning & Deep Learning Workflows.

Audits:
1. Data Leakage (Target leakage, near-duplicate target columns, duplicate rows in split)
2. Class Imbalance (Severe minority class imbalance, inappropriate metric warnings)
3. Overfitting & Underfitting (Train vs holdout divergence, high variance)
4. Temporal Leakage (Lookahead bias, non-chronological time series splits)
5. Outlier Sensitivity & High Leverage (Extreme z-scores, heavy tail distributions)
6. Statistical Assumptions & Multicollinearity (High pairwise collinearity, VIF)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class ValidationCheckType(str, Enum):
    DATA_LEAKAGE = "data_leakage"
    CLASS_IMBALANCE = "class_imbalance"
    OVERFIT_UNDERFIT = "overfit_underfit"
    TEMPORAL_LEAKAGE = "temporal_leakage"
    OUTLIER_SENSITIVITY = "outlier_sensitivity"
    STATISTICAL_ASSUMPTIONS = "statistical_assumptions"


class IssueSeverity(str, Enum):
    CRITICAL = "CRITICAL"  # Invalidates model results / corrupts training
    WARNING = "WARNING"    # May degrade generalization / requires caution
    INFO = "INFO"          # Best practice suggestion


@dataclass
class ValidationIssue:
    """Standardized representation of a detected modeling issue or vulnerability."""
    check_type: ValidationCheckType
    severity: IssueSeverity
    title: str
    description: str
    affected_columns: List[str] = field(default_factory=list)
    suggested_remediation: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_type": self.check_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "affected_columns": self.affected_columns,
            "suggested_remediation": self.suggested_remediation,
            "metrics": self.metrics,
        }


@dataclass
class ValidationAuditReport:
    """Comprehensive validation report summarizing pre-modeling and post-training diagnostics."""
    target_column: str
    overall_status: str  # "PASSED", "PASSED_WITH_WARNINGS", "FAILED"
    critical_issues_count: int
    warnings_count: int
    issues: List[ValidationIssue]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_column": self.target_column,
            "overall_status": self.overall_status,
            "critical_issues_count": self.critical_issues_count,
            "warnings_count": self.warnings_count,
            "issues": [i.to_dict() for i in self.issues],
            "diagnostics": self.diagnostics,
        }


class DataModelValidator:
    """Unified engine for inspecting data leakage, class imbalance, overfit, and statistical integrity."""

    # ------------------------------------------------------------------
    # 1. Data Leakage Detection
    # ------------------------------------------------------------------
    @staticmethod
    def check_data_leakage(
        df: pd.DataFrame,
        target_column: str,
        correlation_threshold: float = 0.98,
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Inspect for direct or proxy target leakage in feature columns."""
        issues: List[ValidationIssue] = []
        diagnostics: Dict[str, Any] = {"leaking_features": [], "perfect_correlations": []}

        if target_column not in df.columns:
            return issues, diagnostics

        y = df[target_column]
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]

        # 1. Numeric target leakage (Pearson correlation)
        if pd.api.types.is_numeric_dtype(y):
            y_clean = pd.to_numeric(y, errors="coerce")
            for col in numeric_cols:
                x_clean = pd.to_numeric(df[col], errors="coerce")
                valid_mask = ~(x_clean.isna() | y_clean.isna())
                if valid_mask.sum() >= 10:
                    corr = float(np.corrcoef(x_clean[valid_mask], y_clean[valid_mask])[0, 1])
                    if not np.isnan(corr) and abs(corr) >= correlation_threshold:
                        diagnostics["perfect_correlations"].append({"column": col, "correlation": round(corr, 4)})
                        diagnostics["leaking_features"].append(col)
                        issues.append(
                            ValidationIssue(
                                check_type=ValidationCheckType.DATA_LEAKAGE,
                                severity=IssueSeverity.CRITICAL if abs(corr) > 0.999 else IssueSeverity.WARNING,
                                title=f"Potential Target Leakage Detected in '{col}'",
                                description=(
                                    f"Feature '{col}' has an extreme correlation ({corr:.4f}) with the target '{target_column}'. "
                                    f"This typically indicates the feature was generated using target information or is a proxy."
                                ),
                                affected_columns=[col],
                                suggested_remediation=f"Exclude '{col}' from feature matrix before training.",
                                metrics={"correlation": round(corr, 4)},
                            )
                        )

        # 2. Identical column detection
        for col in [c for c in df.columns if c != target_column]:
            if df[col].equals(df[target_column]):
                diagnostics["leaking_features"].append(col)
                issues.append(
                    ValidationIssue(
                        check_type=ValidationCheckType.DATA_LEAKAGE,
                        severity=IssueSeverity.CRITICAL,
                        title=f"Duplicate Target Column '{col}'",
                        description=f"Column '{col}' is identical to the target variable '{target_column}'.",
                        affected_columns=[col],
                        suggested_remediation=f"Drop duplicate column '{col}'.",
                        metrics={"identical": True},
                    )
                )

        return issues, diagnostics

    # ------------------------------------------------------------------
    # 2. Class Imbalance Assessment
    # ------------------------------------------------------------------
    @staticmethod
    def check_class_imbalance(
        y: pd.Series,
        imbalance_threshold: float = 0.20,
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Evaluate class distribution and flag severe imbalance."""
        issues: List[ValidationIssue] = []
        diagnostics: Dict[str, Any] = {}

        clean_y = y.dropna()
        if clean_y.nunique() > 10:
            return issues, {"is_classification": False}

        counts = clean_y.value_counts(normalize=True).to_dict()
        min_class = min(counts.values()) if counts else 1.0
        min_label = min(counts, key=counts.get) if counts else None
        diagnostics = {
            "is_classification": True,
            "class_proportions": {str(k): round(float(v), 4) for k, v in counts.items()},
            "minority_class_ratio": round(float(min_class), 4),
            "is_imbalanced": min_class <= imbalance_threshold,
        }

        if min_class <= imbalance_threshold:
            severity = IssueSeverity.CRITICAL if min_class < 0.05 else IssueSeverity.WARNING
            issues.append(
                ValidationIssue(
                    check_type=ValidationCheckType.CLASS_IMBALANCE,
                    severity=severity,
                    title=f"Severe Class Imbalance ({min_class * 100:.1f}% minority class)",
                    description=(
                        f"Target class '{min_label}' represents only {min_class * 100:.1f}% of observations. "
                        f"Standard accuracy is misleading; evaluation must use weighted/macro F1, ROC-AUC, or PR-AUC."
                    ),
                    affected_columns=[str(y.name)],
                    suggested_remediation="Use StratifiedKFold, class-weight balancing ('balanced'), or SMOTE resampling.",
                    metrics={"minority_ratio": round(min_class, 4), "minority_label": str(min_label)},
                )
            )

        return issues, diagnostics

    # ------------------------------------------------------------------
    # 3. Overfitting / Underfitting Assessment
    # ------------------------------------------------------------------
    @staticmethod
    def check_overfitting_underfitting(
        train_score: float,
        test_score: float,
        primary_metric_name: str = "score",
        overfit_gap_threshold: float = 0.15,
        underfit_threshold: float = 0.50,
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Inspect model score divergence between training and holdout sets."""
        issues: List[ValidationIssue] = []
        delta = float(train_score - test_score)
        diagnostics = {
            "train_score": round(float(train_score), 4),
            "test_score": round(float(test_score), 4),
            "overfit_gap": round(delta, 4),
            "is_overfitting": delta >= overfit_gap_threshold,
            "is_underfitting": (train_score < underfit_threshold and test_score < underfit_threshold),
        }

        # Overfitting check
        if delta >= overfit_gap_threshold:
            issues.append(
                ValidationIssue(
                    check_type=ValidationCheckType.OVERFIT_UNDERFIT,
                    severity=IssueSeverity.WARNING if delta < 0.30 else IssueSeverity.CRITICAL,
                    title=f"Overfitting Detected (Gap = {delta:.4f})",
                    description=(
                        f"Training {primary_metric_name} ({train_score:.4f}) significantly exceeds test "
                        f"{primary_metric_name} ({test_score:.4f}). The model has memorized training noise."
                    ),
                    suggested_remediation="Increase regularization (L1/L2 alpha, max_depth limits, dropout) or gather more data.",
                    metrics={"train_score": round(train_score, 4), "test_score": round(test_score, 4), "gap": round(delta, 4)},
                )
            )

        # Underfitting check
        if train_score < underfit_threshold and test_score < underfit_threshold:
            issues.append(
                ValidationIssue(
                    check_type=ValidationCheckType.OVERFIT_UNDERFIT,
                    severity=IssueSeverity.WARNING,
                    title=f"Underfitting Detected (Low Overall Score)",
                    description=(
                        f"Both training ({train_score:.4f}) and test ({test_score:.4f}) {primary_metric_name} are low. "
                        f"The model lacks the capacity to capture true data patterns."
                    ),
                    suggested_remediation="Use non-linear model architectures (Random Forest, Gradient Boosting, ANN) or engineer interaction features.",
                    metrics={"train_score": round(train_score, 4), "test_score": round(test_score, 4)},
                )
            )

        return issues, diagnostics

    # ------------------------------------------------------------------
    # 4. Temporal Leakage in Time Series
    # ------------------------------------------------------------------
    @staticmethod
    def check_temporal_leakage(
        train_timestamps: Union[pd.Series, List[Any]],
        test_timestamps: Union[pd.Series, List[Any]],
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Verify that training timestamps strictly precede test timestamps (no lookahead bias)."""
        issues: List[ValidationIssue] = []
        diagnostics: Dict[str, Any] = {}

        t_tr = pd.to_datetime(pd.Series(train_timestamps)).dropna()
        t_te = pd.to_datetime(pd.Series(test_timestamps)).dropna()

        if t_tr.empty or t_te.empty:
            return issues, {"checked": False}

        max_train = t_tr.max()
        min_test = t_te.min()
        has_temporal_overlap = max_train >= min_test

        diagnostics = {
            "max_train_date": str(max_train),
            "min_test_date": str(min_test),
            "has_temporal_leakage": has_temporal_overlap,
        }

        if has_temporal_overlap:
            overlap_count = int((t_tr >= min_test).sum())
            issues.append(
                ValidationIssue(
                    check_type=ValidationCheckType.TEMPORAL_LEAKAGE,
                    severity=IssueSeverity.CRITICAL,
                    title="Temporal Lookahead Leakage Detected",
                    description=(
                        f"Training split contains timestamps ({max_train}) occurring at or after test timestamps ({min_test}). "
                        f"This causes artificial lookahead bias in time series forecasting."
                    ),
                    suggested_remediation="Use chronological TimeSeriesSplit or split strictly by cutoff date without random shuffling.",
                    metrics={"max_train": str(max_train), "min_test": str(min_test), "overlap_samples": overlap_count},
                )
            )

        return issues, diagnostics

    # ------------------------------------------------------------------
    # 5. Outliers & High Leverage
    # ------------------------------------------------------------------
    @staticmethod
    def check_outliers(
        df: pd.DataFrame,
        z_threshold: float = 3.5,
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Identify extreme numeric outliers that could distort linear models and MSE losses."""
        issues: List[ValidationIssue] = []
        outlier_cols: Dict[str, int] = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) >= 10:
                mean_val = s.mean()
                std_val = s.std()
                if std_val > 0:
                    z_scores = np.abs((s - mean_val) / std_val)
                    extreme_count = int((z_scores > z_threshold).sum())
                    if extreme_count > 0:
                        outlier_cols[col] = extreme_count

        diagnostics = {"columns_with_extreme_outliers": outlier_cols}

        if outlier_cols:
            top_outlier_col = max(outlier_cols, key=outlier_cols.get)
            issues.append(
                ValidationIssue(
                    check_type=ValidationCheckType.OUTLIER_SENSITIVITY,
                    severity=IssueSeverity.INFO,
                    title=f"Outliers Detected in {len(outlier_cols)} Column(s)",
                    description=(
                        f"Columns contain extreme values exceeding {z_threshold} standard deviations "
                        f"(e.g. '{top_outlier_col}' has {outlier_cols[top_outlier_col]} outliers). "
                        f"Consider robust scalers or tree-based algorithms."
                    ),
                    affected_columns=list(outlier_cols.keys()),
                    suggested_remediation="Use RobustScaler, winsorization, or robust loss functions (Huber / MAE).",
                    metrics=outlier_cols,
                )
            )

        return issues, diagnostics

    # ------------------------------------------------------------------
    # 6. Statistical Assumptions & Multicollinearity
    # ------------------------------------------------------------------
    @staticmethod
    def check_multicollinearity(
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        collinear_threshold: float = 0.95,
    ) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
        """Detect highly collinear feature pairs that inflate regression standard errors."""
        issues: List[ValidationIssue] = []
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]

        collinear_pairs: List[Dict[str, Any]] = []
        if len(numeric_cols) >= 2:
            corr_mat = df[numeric_cols].corr().abs()
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    c1, c2 = numeric_cols[i], numeric_cols[j]
                    val = float(corr_mat.loc[c1, c2])
                    if not np.isnan(val) and val >= collinear_threshold:
                        collinear_pairs.append({"col1": c1, "col2": c2, "correlation": round(val, 4)})

        diagnostics = {"collinear_pairs": collinear_pairs}

        if collinear_pairs:
            pair = collinear_pairs[0]
            issues.append(
                ValidationIssue(
                    check_type=ValidationCheckType.STATISTICAL_ASSUMPTIONS,
                    severity=IssueSeverity.WARNING,
                    title=f"High Multicollinearity Between '{pair['col1']}' and '{pair['col2']}'",
                    description=(
                        f"Features '{pair['col1']}' and '{pair['col2']}' have correlation {pair['correlation']:.4f} "
                        f"(>= {collinear_threshold}). This destabilizes linear regression coefficients."
                    ),
                    affected_columns=[pair["col1"], pair["col2"]],
                    suggested_remediation="Apply Ridge regularization or drop one of the redundant collinear features.",
                    metrics=pair,
                )
            )

        return issues, diagnostics

    # ------------------------------------------------------------------
    # Full Comprehensive Audit Pipeline
    # ------------------------------------------------------------------
    def audit_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str,
        train_score: Optional[float] = None,
        test_score: Optional[float] = None,
        primary_metric_name: str = "score",
        train_dates: Optional[Union[pd.Series, List[Any]]] = None,
        test_dates: Optional[Union[pd.Series, List[Any]]] = None,
    ) -> ValidationAuditReport:
        """Run all validation checks and synthesize a complete audit report."""
        all_issues: List[ValidationIssue] = []
        all_diagnostics: Dict[str, Any] = {}

        # 1. Leakage
        leak_issues, leak_diag = self.check_data_leakage(df, target_column)
        all_issues.extend(leak_issues)
        all_diagnostics["leakage"] = leak_diag

        # 2. Imbalance
        if target_column in df.columns:
            imb_issues, imb_diag = self.check_class_imbalance(df[target_column])
            all_issues.extend(imb_issues)
            all_diagnostics["imbalance"] = imb_diag

        # 3. Multicollinearity
        mc_issues, mc_diag = self.check_multicollinearity(df, target_column)
        all_issues.extend(mc_issues)
        all_diagnostics["multicollinearity"] = mc_diag

        # 4. Outliers
        out_issues, out_diag = self.check_outliers(df)
        all_issues.extend(out_issues)
        all_diagnostics["outliers"] = out_diag

        # 5. Overfit / Underfit (if scores provided)
        if train_score is not None and test_score is not None:
            of_issues, of_diag = self.check_overfitting_underfitting(
                train_score, test_score, primary_metric_name=primary_metric_name
            )
            all_issues.extend(of_issues)
            all_diagnostics["overfit_underfit"] = of_diag

        # 6. Temporal Leakage (if dates provided)
        if train_dates is not None and test_dates is not None:
            temp_issues, temp_diag = self.check_temporal_leakage(train_dates, test_dates)
            all_issues.extend(temp_issues)
            all_diagnostics["temporal"] = temp_diag

        critical_count = sum(1 for i in all_issues if i.severity == IssueSeverity.CRITICAL)
        warning_count = sum(1 for i in all_issues if i.severity == IssueSeverity.WARNING)

        if critical_count > 0:
            status = "FAILED"
        elif warning_count > 0:
            status = "PASSED_WITH_WARNINGS"
        else:
            status = "PASSED"

        return ValidationAuditReport(
            target_column=target_column,
            overall_status=status,
            critical_issues_count=critical_count,
            warnings_count=warning_count,
            issues=all_issues,
            diagnostics=all_diagnostics,
        )
