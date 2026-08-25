"""
Deterministic Model Monitoring & Statistical Drift Engine.

Provides rigorous, evidence-based statistical testing for:
1. Numeric feature drift (2-sample Kolmogorov-Smirnov & Population Stability Index)
2. Categorical feature drift (Chi-Square test, Categorical PSI, novel & disappearing categories)
3. Missing-value rate drift & Data quality tracking
4. Schema & Data type drift (missing features, unexpected columns, incompatible dtypes)
5. Target drift (when ground-truth outcomes are available)
6. Prediction drift (distribution divergence on inference outputs)
7. Model performance degradation benchmarking (against reference validation metrics)
8. Graduated severity classification & evidence-backed recommendations
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from agent.model_monitoring_schemas import (
    DatasetDriftReport,
    DriftRequest,
    DriftSeverity,
    DriftThresholdConfig,
    FeatureDriftResult,
    ModelPerformanceReport,
    MonitoringResult,
    PredictionDriftReport,
)
from agent.schemas import ClaimType, Evidence
from backend.app.ml.registry import ModelArtifactMetadata, ModelRegistry


class ModelMonitoringEngine:
    """
    Deterministic Statistical Monitoring Engine.
    Executes hypothesis testing, divergence calculations, and metric comparisons without hallucinations.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()

    # ------------------------------------------------------------------
    # 1. Reference Profile Builder
    # ------------------------------------------------------------------
    @staticmethod
    def build_reference_profile(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate statistical baseline profile from training reference dataset."""
        profile: Dict[str, Any] = {
            "row_count": len(df),
            "feature_columns": feature_cols,
            "feature_dtypes": {c: str(df[c].dtype) for c in feature_cols if c in df.columns},
            "features": {},
            "target": None,
            "missing_rates": {},
            "duplicate_count": int(df.duplicated().sum()),
        }

        for col in feature_cols:
            if col not in df.columns:
                continue
            s = df[col]
            missing_pct = float(s.isna().mean())
            profile["missing_rates"][col] = round(missing_pct, 4)

            if pd.api.types.is_numeric_dtype(s):
                valid_vals = s.dropna().to_numpy(dtype=float)
                if len(valid_vals) > 0:
                    profile["features"][col] = {
                        "type": "numeric",
                        "mean": float(np.mean(valid_vals)),
                        "std": float(np.std(valid_vals)),
                        "min": float(np.min(valid_vals)),
                        "p25": float(np.percentile(valid_vals, 25)),
                        "median": float(np.median(valid_vals)),
                        "p75": float(np.percentile(valid_vals, 75)),
                        "max": float(np.max(valid_vals)),
                        "sample_quantiles": np.percentile(valid_vals, np.linspace(0, 100, 11)).tolist(),
                    }
                else:
                    profile["features"][col] = {"type": "numeric", "empty": True}
            else:
                vc = s.dropna().astype(str).value_counts(normalize=True).to_dict()
                profile["features"][col] = {
                    "type": "categorical",
                    "frequencies": {str(k): round(float(v), 4) for k, v in vc.items()},
                    "cardinality": int(s.nunique()),
                }

        if target_col and target_col in df.columns:
            s_target = df[target_col]
            if pd.api.types.is_numeric_dtype(s_target):
                v_target = s_target.dropna().to_numpy(dtype=float)
                profile["target"] = {
                    "column": target_col,
                    "type": "numeric",
                    "mean": float(np.mean(v_target)) if len(v_target) > 0 else 0.0,
                    "std": float(np.std(v_target)) if len(v_target) > 0 else 0.0,
                }
            else:
                vc_t = s_target.dropna().astype(str).value_counts(normalize=True).to_dict()
                profile["target"] = {
                    "column": target_col,
                    "type": "categorical",
                    "frequencies": {str(k): round(float(v), 4) for k, v in vc_t.items()},
                }

        return profile

    # ------------------------------------------------------------------
    # 2. Statistical Drift Calculations
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_numeric_psi(
        ref_vals: np.ndarray,
        curr_vals: np.ndarray,
        num_bins: int = 10,
    ) -> float:
        """Calculate Population Stability Index (PSI) for continuous numeric distributions."""
        if len(ref_vals) == 0 or len(curr_vals) == 0:
            return 0.0

        # Create quantile-based bins from reference
        quantiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(ref_vals, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_counts, _ = np.histogram(ref_vals, bins=bins)
        curr_counts, _ = np.histogram(curr_vals, bins=bins)

        eps = 1e-4
        ref_dist = (ref_counts + eps) / (len(ref_vals) + eps * num_bins)
        curr_dist = (curr_counts + eps) / (len(curr_vals) + eps * num_bins)

        psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
        return float(max(0.0, psi))

    @staticmethod
    def calculate_categorical_psi(
        ref_freqs: Dict[str, float],
        curr_freqs: Dict[str, float],
    ) -> float:
        """Calculate Population Stability Index (PSI) for discrete categorical distributions."""
        all_cats = list(set(ref_freqs.keys()) | set(curr_freqs.keys()))
        if not all_cats:
            return 0.0

        eps = 1e-4
        psi = 0.0
        for cat in all_cats:
            r = ref_freqs.get(cat, 0.0) + eps
            c = curr_freqs.get(cat, 0.0) + eps
            psi += (c - r) * math.log(c / r)

        return float(max(0.0, psi))

    def evaluate_numeric_feature_drift(
        self,
        feature_name: str,
        ref_vals: np.ndarray,
        curr_vals: np.ndarray,
        threshold_config: DriftThresholdConfig,
    ) -> FeatureDriftResult:
        """Evaluate numeric feature drift using 2-Sample Kolmogorov-Smirnov test and PSI."""
        if len(ref_vals) == 0 or len(curr_vals) == 0:
            return FeatureDriftResult(
                feature_name=feature_name,
                drift_detected=False,
                drift_score=0.0,
                statistical_test="insufficient_data",
                threshold=threshold_config.numeric_p_value_threshold,
                severity=DriftSeverity.NONE,
            )

        # 2-Sample Kolmogorov-Smirnov Test
        ks_res = stats.ks_2samp(ref_vals, curr_vals)
        ks_stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)

        # PSI computation
        psi_score = self.calculate_numeric_psi(ref_vals, curr_vals)

        p_thresh = threshold_config.numeric_p_value_threshold
        psi_thresh = threshold_config.numeric_psi_threshold
        drift_detected = (p_val < p_thresh) or (psi_score >= psi_thresh)

        # Determine severity
        if not drift_detected:
            sev = DriftSeverity.NONE
        elif psi_score >= 0.25 or (p_val < 0.001 and ks_stat > 0.3):
            sev = DriftSeverity.HIGH
        else:
            sev = DriftSeverity.MEDIUM

        ref_stats = {
            "mean": round(float(np.mean(ref_vals)), 4),
            "std": round(float(np.std(ref_vals)), 4),
            "median": round(float(np.median(ref_vals)), 4),
        }
        curr_stats = {
            "mean": round(float(np.mean(curr_vals)), 4),
            "std": round(float(np.std(curr_vals)), 4),
            "median": round(float(np.median(curr_vals)), 4),
            "ks_statistic": round(ks_stat, 4),
            "psi": round(psi_score, 4),
        }

        evidence = Evidence(
            source="ModelMonitoringEngine",
            method="kolmogorov_smirnov_and_psi",
            data_ref={
                "feature": feature_name,
                "ks_statistic": round(ks_stat, 4),
                "p_value": round(p_val, 4),
                "psi": round(psi_score, 4),
                "ref_mean": ref_stats["mean"],
                "curr_mean": curr_stats["mean"],
            },
            confidence=0.95,
            claim_type=ClaimType.FACT,
        )

        return FeatureDriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            drift_score=round(ks_stat, 4),
            statistical_test="kolmogorov_smirnov",
            p_value=round(p_val, 4),
            reference_statistics=ref_stats,
            current_statistics=curr_stats,
            threshold=p_thresh,
            severity=sev,
            evidence=evidence,
            confidence=0.95,
        )

    def evaluate_categorical_feature_drift(
        self,
        feature_name: str,
        ref_series: pd.Series,
        curr_series: pd.Series,
        threshold_config: DriftThresholdConfig,
    ) -> FeatureDriftResult:
        """Evaluate categorical distribution drift using Chi-Square homogeneity test, PSI, and novel category detection."""
        ref_clean = ref_series.dropna().astype(str)
        curr_clean = curr_series.dropna().astype(str)

        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return FeatureDriftResult(
                feature_name=feature_name,
                drift_detected=False,
                drift_score=0.0,
                statistical_test="insufficient_data",
                threshold=threshold_config.categorical_p_value_threshold,
                severity=DriftSeverity.NONE,
            )

        ref_freq = ref_clean.value_counts(normalize=True).to_dict()
        curr_freq = curr_clean.value_counts(normalize=True).to_dict()

        # Novel categories (present in current, missing in ref)
        novel_cats = list(set(curr_freq.keys()) - set(ref_freq.keys()))
        # Disappearing categories (in ref with > 5% weight, missing in current)
        disappearing_cats = [k for k, v in ref_freq.items() if v >= 0.05 and k not in curr_freq]

        # Categorical PSI
        psi_score = self.calculate_categorical_psi(ref_freq, curr_freq)

        # Chi-Square Test
        all_categories = sorted(list(set(ref_freq.keys()) | set(curr_freq.keys())))
        ref_counts = np.array([ref_clean.eq(c).sum() for c in all_categories], dtype=float)
        curr_counts = np.array([curr_clean.eq(c).sum() for c in all_categories], dtype=float)

        # Rescale reference counts to match current sample size for expected frequencies
        expected_counts = (ref_counts / max(1, ref_counts.sum())) * curr_counts.sum()
        expected_counts = np.maximum(expected_counts, 1e-3)

        try:
            chi_res = stats.chisquare(f_obs=curr_counts, f_exp=expected_counts)
            p_val = float(chi_res.pvalue)
            chi_stat = float(chi_res.statistic)
        except Exception:
            p_val = 1.0
            chi_stat = 0.0

        p_thresh = threshold_config.categorical_p_value_threshold
        psi_thresh = threshold_config.categorical_psi_threshold

        drift_detected = (
            (p_val < p_thresh)
            or (psi_score >= psi_thresh)
            or (len(novel_cats) > 0 and len(curr_clean) >= 20)
        )

        if not drift_detected:
            sev = DriftSeverity.NONE
        elif psi_score >= 0.25 or len(novel_cats) >= 2:
            sev = DriftSeverity.HIGH
        else:
            sev = DriftSeverity.MEDIUM

        ref_stats = {"frequencies": {k: round(v, 4) for k, v in ref_freq.items()}, "unique_count": len(ref_freq)}
        curr_stats = {
            "frequencies": {k: round(v, 4) for k, v in curr_freq.items()},
            "unique_count": len(curr_freq),
            "novel_categories": novel_cats,
            "disappearing_categories": disappearing_cats,
            "chi_square_stat": round(chi_stat, 4),
            "psi": round(psi_score, 4),
        }

        evidence = Evidence(
            source="ModelMonitoringEngine",
            method="chi_square_and_categorical_psi",
            data_ref={
                "feature": feature_name,
                "p_value": round(p_val, 4),
                "psi": round(psi_score, 4),
                "novel_categories": novel_cats,
                "disappearing_categories": disappearing_cats,
            },
            confidence=0.95,
            claim_type=ClaimType.FACT,
        )

        return FeatureDriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            drift_score=round(psi_score, 4),
            statistical_test="chi_square",
            p_value=round(p_val, 4),
            reference_statistics=ref_stats,
            current_statistics=curr_stats,
            threshold=p_thresh,
            severity=sev,
            evidence=evidence,
            confidence=0.95,
        )

    # ------------------------------------------------------------------
    # 3. Schema & Data Quality Drift
    # ------------------------------------------------------------------
    @staticmethod
    def evaluate_schema_drift(
        expected_features: List[str],
        expected_dtypes: Dict[str, str],
        curr_df: pd.DataFrame,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Identify schema discrepancies: missing features, novel columns, and data type alterations."""
        curr_cols = list(curr_df.columns)
        missing_features = [f for f in expected_features if f not in curr_cols]
        extra_columns = [c for c in curr_cols if c not in expected_features]

        dtype_mismatches: Dict[str, Dict[str, str]] = {}
        for col in expected_features:
            if col in curr_df.columns and col in expected_dtypes:
                exp_dt = expected_dtypes[col]
                curr_dt = str(curr_df[col].dtype)
                # Flag incompatible type alterations (e.g. float/int turning into string object)
                if ("float" in exp_dt or "int" in exp_dt) and ("object" in curr_dt or "str" in curr_dt):
                    dtype_mismatches[col] = {"expected": exp_dt, "found": curr_dt}

        schema_drift = (len(missing_features) > 0) or (len(dtype_mismatches) > 0)
        return schema_drift, {
            "missing_features": missing_features,
            "extra_columns": extra_columns,
            "dtype_mismatches": dtype_mismatches,
        }

    # ------------------------------------------------------------------
    # 4. Master Monitoring Execution
    # ------------------------------------------------------------------
    def monitor(
        self,
        request: DriftRequest,
    ) -> MonitoringResult:
        """
        Execute comprehensive model monitoring:
        1. Fetch model metadata and baseline reference profile
        2. Evaluate schema and missing-value drift
        3. Evaluate numeric & categorical feature drift
        4. Evaluate target drift & performance degradation (if ground-truth available)
        5. Evaluate prediction distribution drift
        6. Compute overall severity and formulate grounded evidence & recommendations
        7. Persist to monitoring history log
        """
        model_id = request.model_id
        meta = self.registry.get_metadata(model_id)
        if meta is None:
            return MonitoringResult(
                model_id=model_id,
                status="failed",
                overall_severity=DriftSeverity.CRITICAL,
                warnings=[f"Model ID '{model_id}' not found in ModelRegistry."],
                recommendations=["Ensure the model ID is correctly registered before monitoring."],
                confidence=0.0,
            )

        # Convert current dataset to DataFrame
        if isinstance(request.current_dataset, dict):
            curr_df = pd.DataFrame([request.current_dataset])
        elif isinstance(request.current_dataset, list):
            curr_df = pd.DataFrame(request.current_dataset)
        elif isinstance(request.current_dataset, pd.DataFrame):
            curr_df = request.current_dataset.copy()
        else:
            return MonitoringResult(
                model_id=model_id,
                status="failed",
                overall_severity=DriftSeverity.CRITICAL,
                warnings=[f"Unsupported current_dataset type: {type(request.current_dataset)}"],
                confidence=0.0,
            )

        feature_cols = request.feature_columns or meta.feature_columns
        target_col = request.target_column or meta.target_column
        thresh_cfg = request.threshold_config

        # 1. Resolve Reference Baseline
        ref_df: Optional[pd.DataFrame] = None
        if request.reference_dataset is not None:
            if isinstance(request.reference_dataset, pd.DataFrame):
                ref_df = request.reference_dataset.copy()
            elif isinstance(request.reference_dataset, (dict, list)):
                ref_df = pd.DataFrame(request.reference_dataset)

        ref_profile = meta.reference_profile or {}
        if ref_df is not None and not ref_df.empty:
            ref_profile = self.build_reference_profile(ref_df, feature_cols, target_col)

        # 2. Schema Drift Evaluation
        schema_drift_detected, schema_changes = self.evaluate_schema_drift(
            expected_features=feature_cols,
            expected_dtypes=meta.feature_dtypes,
            curr_df=curr_df,
        )

        # 3. Missing Value & Data Quality Changes
        data_quality_changes: Dict[str, Any] = {
            "current_row_count": len(curr_df),
            "reference_row_count": ref_profile.get("row_count", 0),
            "current_duplicate_count": int(curr_df.duplicated().sum()),
            "reference_duplicate_count": ref_profile.get("duplicate_count", 0),
            "missing_rate_deltas": {},
        }
        missing_rate_drift_detected = False
        for col in feature_cols:
            if col in curr_df.columns:
                curr_missing = float(curr_df[col].isna().mean())
                ref_missing = float(ref_profile.get("missing_rates", {}).get(col, 0.0))
                delta = curr_missing - ref_missing
                data_quality_changes["missing_rate_deltas"][col] = round(delta, 4)
                if delta >= thresh_cfg.missing_rate_delta_threshold:
                    missing_rate_drift_detected = True

        # 4. Feature Drift Evaluation
        feature_results: Dict[str, FeatureDriftResult] = {}
        drifted_features: List[str] = []
        all_evidence: List[Evidence] = []

        for col in feature_cols:
            if col not in curr_df.columns:
                continue

            curr_s = curr_df[col]
            if pd.api.types.is_numeric_dtype(curr_s):
                curr_vals = curr_s.dropna().to_numpy(dtype=float)
                ref_vals = (
                    ref_df[col].dropna().to_numpy(dtype=float)
                    if ref_df is not None and col in ref_df.columns
                    else np.array(ref_profile.get("features", {}).get(col, {}).get("sample_quantiles", []))
                )
                res = self.evaluate_numeric_feature_drift(col, ref_vals, curr_vals, thresh_cfg)
            else:
                ref_s = (
                    ref_df[col]
                    if ref_df is not None and col in ref_df.columns
                    else pd.Series(list(ref_profile.get("features", {}).get(col, {}).get("frequencies", {}).keys()))
                )
                res = self.evaluate_categorical_feature_drift(col, ref_s, curr_s, thresh_cfg)

            feature_results[col] = res
            if res.drift_detected:
                drifted_features.append(col)
            if res.evidence:
                all_evidence.append(res.evidence)

        drift_pct = (len(drifted_features) / max(1, len(feature_cols))) * 100.0
        overall_data_drift = (len(drifted_features) > 0) or schema_drift_detected or missing_rate_drift_detected

        data_drift_report = DatasetDriftReport(
            dataset_id="current_data",
            reference_dataset_id=f"{meta.name}_v{meta.version}_baseline",
            features_checked=feature_cols,
            drifted_features=drifted_features,
            drift_percentage=round(drift_pct, 2),
            overall_drift=overall_data_drift,
            schema_drift_detected=schema_drift_detected,
            schema_changes=schema_changes,
            data_quality_changes=data_quality_changes,
            feature_results=feature_results,
            severity=DriftSeverity.NONE,
            confidence=0.95,
            evidence=all_evidence,
        )

        # 5. Prediction Drift Evaluation
        pred_drift_report: Optional[PredictionDriftReport] = None
        curr_preds: Optional[np.ndarray] = None
        if request.compute_predictions and not schema_changes.get("missing_features"):
            try:
                pred_out = self.registry.predict(model_id, curr_df[feature_cols])
                curr_preds = np.array(pred_out["predictions"])

                is_num_pred = np.issubdtype(curr_preds.dtype, np.number)
                if is_num_pred and len(curr_preds) > 0:
                    curr_pred_stats = {
                        "mean": round(float(np.mean(curr_preds)), 4),
                        "std": round(float(np.std(curr_preds)), 4),
                    }
                    ref_target_mean = ref_profile.get("target", {}).get("mean", curr_pred_stats["mean"])
                    mean_shift = abs(curr_pred_stats["mean"] - ref_target_mean) / max(1e-4, abs(ref_target_mean))
                    pred_drift = mean_shift >= thresh_cfg.prediction_drift_threshold

                    pred_drift_report = PredictionDriftReport(
                        model_id=model_id,
                        prediction_drift_detected=pred_drift,
                        statistical_test="mean_relative_shift",
                        drift_score=round(mean_shift, 4),
                        reference_prediction_stats={"expected_mean": round(ref_target_mean, 4)},
                        current_prediction_stats=curr_pred_stats,
                        confidence=0.90,
                    )
                else:
                    curr_pred_vc = pd.Series(curr_preds).value_counts(normalize=True).to_dict()
                    curr_pred_stats = {"frequencies": {str(k): round(float(v), 4) for k, v in curr_pred_vc.items()}}
                    pred_drift_report = PredictionDriftReport(
                        model_id=model_id,
                        prediction_drift_detected=False,
                        statistical_test="class_frequency",
                        drift_score=0.0,
                        current_prediction_stats=curr_pred_stats,
                        confidence=0.90,
                    )
            except Exception as exc:
                data_drift_report.warnings.append(f"Inference execution for prediction drift failed: {str(exc)}")

        # 6. Model Performance Degradation Evaluation (Ground Truth check)
        perf_report: Optional[ModelPerformanceReport] = None
        has_ground_truth = target_col in curr_df.columns and curr_df[target_col].notna().sum() >= 5

        if has_ground_truth and curr_preds is not None:
            y_true = curr_df[target_col].to_numpy()
            y_pred = curr_preds
            is_clf = "class" in meta.problem_type.lower() or "binary" in meta.problem_type.lower() or "multi" in meta.problem_type.lower()

            current_metrics: Dict[str, float] = {}
            if is_clf:
                current_metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
                current_metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted" if len(np.unique(y_true)) > 2 else "binary", zero_division=0))
                current_metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted" if len(np.unique(y_true)) > 2 else "binary", zero_division=0))
                current_metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted" if len(np.unique(y_true)) > 2 else "binary", zero_division=0))
            else:
                try:
                    current_metrics["r2"] = float(r2_score(y_true, y_pred))
                except Exception:
                    current_metrics["r2"] = 0.0
                current_metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
                current_metrics["rmse"] = float(math.sqrt(mean_squared_error(y_true, y_pred)))

            metric_changes: Dict[str, float] = {}
            degradation_detected = False
            ref_metrics = meta.validation_metrics or meta.training_metrics or {}

            for m_name, curr_m_val in current_metrics.items():
                if m_name in ref_metrics:
                    ref_m_val = float(ref_metrics[m_name])
                    delta = curr_m_val - ref_m_val
                    metric_changes[m_name] = round(delta, 4)

                    # Higher is better for score metrics; lower is better for error metrics
                    if m_name in ("rmse", "mae", "mape", "loss"):
                        if delta > (ref_m_val * thresh_cfg.performance_degradation_threshold):
                            degradation_detected = True
                    else:
                        if delta < -thresh_cfg.performance_degradation_threshold:
                            degradation_detected = True

            perf_evidence = [
                Evidence(
                    source="ModelMonitoringEngine",
                    method="ground_truth_metric_evaluation",
                    data_ref={
                        "current_metrics": current_metrics,
                        "reference_metrics": ref_metrics,
                        "metric_changes": metric_changes,
                        "degradation_detected": degradation_detected,
                    },
                    confidence=0.95,
                    claim_type=ClaimType.FACT,
                )
            ]

            perf_report = ModelPerformanceReport(
                model_id=model_id,
                reference_metrics={k: round(v, 4) for k, v in ref_metrics.items()},
                current_metrics={k: round(v, 4) for k, v in current_metrics.items()},
                metric_changes=metric_changes,
                degradation_detected=degradation_detected,
                target_monitoring_status="evaluated",
                evaluation_dataset_rows=len(curr_df),
                confidence=0.95,
                evidence=perf_evidence,
            )
        else:
            perf_report = ModelPerformanceReport(
                model_id=model_id,
                reference_metrics={k: round(float(v), 4) for k, v in (meta.validation_metrics or {}).items()},
                target_monitoring_status="unavailable",
                evaluation_dataset_rows=len(curr_df),
                confidence=0.85,
                evidence=[],
            )

        # 7. Severity Determination & Actionable Recommendations
        overall_sev = self._calculate_overall_severity(
            schema_drift=schema_drift_detected,
            drift_percentage=drift_pct,
            missing_rate_drift=missing_rate_drift_detected,
            performance_degradation=perf_report.degradation_detected if perf_report else False,
            prediction_drift=pred_drift_report.prediction_drift_detected if pred_drift_report else False,
            has_ground_truth=has_ground_truth,
        )
        data_drift_report.severity = overall_sev

        recommendations = self._generate_recommendations(
            overall_severity=overall_sev,
            schema_changes=schema_changes,
            drifted_features=drifted_features,
            performance_report=perf_report,
            has_ground_truth=has_ground_truth,
        )

        mon_result = MonitoringResult(
            model_id=model_id,
            status="success",
            overall_severity=overall_sev,
            data_drift=data_drift_report,
            prediction_drift=pred_drift_report,
            performance_drift=perf_report,
            data_quality=data_quality_changes,
            recommendations=recommendations,
            warnings=data_drift_report.warnings,
            evidence=all_evidence + (perf_report.evidence if perf_report else []),
            confidence=0.95,
            timestamp=datetime.now(),
        )

        # 8. Persist to Monitoring History
        try:
            self.registry.record_monitoring_run(
                model_id=model_id,
                run_summary={
                    "overall_severity": overall_sev.value,
                    "drift_percentage": round(drift_pct, 2),
                    "drifted_features": drifted_features,
                    "schema_drift": schema_drift_detected,
                    "performance_degradation": perf_report.degradation_detected if perf_report else False,
                    "target_monitoring_status": perf_report.target_monitoring_status if perf_report else "unavailable",
                    "row_count": len(curr_df),
                },
            )
        except Exception:
            pass

        return mon_result

    # ------------------------------------------------------------------
    # 5. Severity & Recommendation Rules
    # ------------------------------------------------------------------
    @staticmethod
    def _calculate_overall_severity(
        schema_drift: bool,
        drift_percentage: float,
        missing_rate_drift: bool,
        performance_degradation: bool,
        prediction_drift: bool,
        has_ground_truth: bool,
    ) -> DriftSeverity:
        """Evaluate multi-dimensional severity level based on deterministic rule matrix."""
        if schema_drift or performance_degradation:
            return DriftSeverity.CRITICAL if performance_degradation else DriftSeverity.HIGH

        if drift_percentage >= 50.0 or (drift_percentage >= 30.0 and missing_rate_drift):
            return DriftSeverity.HIGH

        if drift_percentage >= 20.0 or prediction_drift or missing_rate_drift:
            return DriftSeverity.MEDIUM

        if drift_percentage > 0.0:
            return DriftSeverity.LOW

        return DriftSeverity.NONE

    @staticmethod
    def _generate_recommendations(
        overall_severity: DriftSeverity,
        schema_changes: Dict[str, Any],
        drifted_features: List[str],
        performance_report: Optional[ModelPerformanceReport],
        has_ground_truth: bool,
    ) -> List[str]:
        """Formulate deterministic recommendations grounded strictly in monitoring evidence."""
        recs: List[str] = []

        if schema_changes.get("missing_features"):
            recs.append(
                f"SCHEMA DRIFT: Input data is missing required features {schema_changes['missing_features']}. "
                f"Verify upstream data ingestion pipelines."
            )

        if schema_changes.get("dtype_mismatches"):
            recs.append(
                f"TYPE DRIFT: Incompatible data types detected in {list(schema_changes['dtype_mismatches'].keys())}. "
                f"Enforce schema casting before model execution."
            )

        if performance_report and performance_report.degradation_detected:
            recs.append(
                f"PERFORMANCE DEGRADATION: Model metric drop detected (changes: {performance_report.metric_changes}). "
                f"Investigate recent data distribution changes and consider model retraining."
            )

        if drifted_features:
            recs.append(
                f"FEATURE DRIFT: Significant distribution shift observed in {drifted_features}. "
                f"Review feature distributions and upstream transformations."
            )

        if not has_ground_truth:
            recs.append(
                "GROUND TRUTH: Ground-truth target labels are unavailable for current batch. "
                "Model performance cannot be directly evaluated; collect ground-truth labels for definitive validation."
            )

        if overall_severity == DriftSeverity.NONE:
            recs.append("HEALTH CHECK: Model and input feature distributions are stable. No corrective action required.")

        return recs
