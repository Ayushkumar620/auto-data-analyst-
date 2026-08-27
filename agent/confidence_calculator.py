"""
Universal Data-Driven Confidence Engine.

Calculates grounded, mathematically bounded epistemic confidence scores [0.0, 1.0]
for all analytical agents based on:
1. Validation performance (R2, Accuracy, F1, sMAPE, Silhouette)
2. Sample size sufficiency (N / features ratio)
3. Data completeness and missingness rate
4. Cross-validation variance and model stability
5. Baseline backtest outperformance (forecasting)
6. Ambiguity in user intent or multiple candidate targets

Explicitly separates:
- confidence (epistemic certainty in analytical claim)
- confidence_level (probabilistic prediction interval width)
- validation_score (metric benchmark score)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np


@dataclass
class ConfidenceReport:
    """Detailed decomposition of how confidence was derived."""
    confidence: float
    confidence_level: float
    validation_score: float
    factors: Dict[str, float] = field(default_factory=dict)
    penalties: List[str] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": round(float(self.confidence), 4),
            "confidence_level": round(float(self.confidence_level), 4),
            "validation_score": round(float(self.validation_score), 4),
            "factors": {k: round(float(v), 4) for k, v in self.factors.items()},
            "penalties": self.penalties,
            "explanations": self.explanations,
        }


class ConfidenceCalculator:
    """Computes transparent, data-driven confidence for analytical agents."""

    @staticmethod
    def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        if math.isnan(value) or math.isinf(value):
            return min_val
        return max(min_val, min(max_val, float(value)))

    @classmethod
    def calculate_regression_confidence(
        cls,
        r2_score: Optional[float] = None,
        cv_scores: Optional[List[float]] = None,
        n_samples: int = 0,
        n_features: int = 1,
        missing_rate: float = 0.0,
        confidence_level: float = 0.80,
    ) -> ConfidenceReport:
        """Calculate confidence for supervised regression models."""
        penalties: List[str] = []
        explanations: List[str] = []
        factors: Dict[str, float] = {}

        # 1. Performance Base: R2 score (0 to 1)
        r2_val = r2_score if r2_score is not None and not math.isnan(r2_score) else 0.0
        val_score = cls._clamp(r2_val, min_val=0.0, max_val=1.0)
        perf_factor = 0.40 + 0.45 * val_score
        factors["performance"] = perf_factor
        explanations.append(f"Model holdout R2 score = {val_score:.3f}")

        # 2. Sample Size Sufficiency Ratio (N / p)
        sample_ratio = n_samples / max(1, n_features)
        if sample_ratio < 5:
            sample_factor = 0.60
            penalties.append(f"Low sample-to-feature ratio ({sample_ratio:.1f} < 5)")
        elif sample_ratio < 15:
            sample_factor = 0.85
        else:
            sample_factor = 1.0
        factors["sample_size"] = sample_factor

        # 3. Model Stability / Cross-Validation Variance
        cv_factor = 1.0
        if cv_scores and len(cv_scores) >= 2:
            cv_std = float(np.std(cv_scores))
            if cv_std > 0.20:
                cv_factor = max(0.60, 1.0 - cv_std)
                penalties.append(f"High cross-validation variance (std = {cv_std:.3f})")
            factors["cv_stability"] = cv_factor

        # 4. Missingness Penalty
        miss_factor = max(0.70, 1.0 - missing_rate)
        factors["completeness"] = miss_factor

        # Combine
        raw_conf = perf_factor * sample_factor * cv_factor * miss_factor
        final_conf = cls._clamp(raw_conf, min_val=0.10, max_val=0.98)

        return ConfidenceReport(
            confidence=final_conf,
            confidence_level=confidence_level,
            validation_score=val_score,
            factors=factors,
            penalties=penalties,
            explanations=explanations,
        )

    @classmethod
    def calculate_classification_confidence(
        cls,
        accuracy: Optional[float] = None,
        f1_score: Optional[float] = None,
        n_samples: int = 0,
        n_features: int = 1,
        is_imbalanced: bool = False,
        confidence_level: float = 0.80,
    ) -> ConfidenceReport:
        """Calculate confidence for supervised classification models."""
        penalties: List[str] = []
        explanations: List[str] = []
        factors: Dict[str, float] = {}

        primary_metric = f1_score if (f1_score is not None and is_imbalanced) else (accuracy or 0.0)
        val_score = cls._clamp(primary_metric, min_val=0.0, max_val=1.0)
        perf_factor = 0.40 + 0.50 * val_score
        factors["performance"] = perf_factor
        explanations.append(f"Classification performance score = {val_score:.3f}")

        sample_ratio = n_samples / max(1, n_features)
        sample_factor = 0.70 if sample_ratio < 10 else 1.0
        factors["sample_size"] = sample_factor

        if is_imbalanced:
            factors["class_balance"] = 0.88
            penalties.append("Class distribution exhibits significant imbalance")

        raw_conf = perf_factor * sample_factor * (0.88 if is_imbalanced else 1.0)
        final_conf = cls._clamp(raw_conf, min_val=0.10, max_val=0.98)

        return ConfidenceReport(
            confidence=final_conf,
            confidence_level=confidence_level,
            validation_score=val_score,
            factors=factors,
            penalties=penalties,
            explanations=explanations,
        )

    @classmethod
    def calculate_forecast_confidence(
        cls,
        validation_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        horizon: int = 5,
        n_obs: int = 20,
        confidence_level: float = 0.80,
    ) -> ConfidenceReport:
        """Calculate confidence for time-series forecasting models."""
        penalties: List[str] = []
        explanations: List[str] = []
        factors: Dict[str, float] = {}

        # 1. Baseline outperformance ratio (Model MAE vs Naive MAE)
        model_mae = validation_metrics.get("MAE", 1.0)
        base_mae = baseline_metrics.get("MAE", model_mae or 1.0)

        if base_mae > 1e-9:
            mae_ratio = model_mae / base_mae
            if mae_ratio < 0.85:
                perf_factor = 0.92
                explanations.append(f"Selected candidate outperforms naive baseline by {((1 - mae_ratio) * 100):.1f}%")
            elif mae_ratio <= 1.05:
                perf_factor = 0.82
            else:
                perf_factor = 0.65
                penalties.append("Candidate model exhibits higher error than naive baseline")
        else:
            perf_factor = 0.80
        factors["baseline_comparison"] = perf_factor

        # 2. Sample size sufficiency
        if n_obs < 10:
            sample_factor = 0.70
            penalties.append(f"Short historical series (N={n_obs} < 10)")
        elif n_obs < 30:
            sample_factor = 0.88
        else:
            sample_factor = 1.0
        factors["sample_size"] = sample_factor

        # 3. Horizon compounding uncertainty
        horizon_ratio = horizon / max(1, n_obs)
        if horizon_ratio > 0.40:
            horizon_factor = 0.75
            penalties.append(f"Forecast horizon ({horizon}) is large relative to history ({n_obs})")
        else:
            horizon_factor = max(0.85, 1.0 - (horizon * 0.015))
        factors["horizon_decay"] = horizon_factor

        raw_conf = perf_factor * sample_factor * horizon_factor
        final_conf = cls._clamp(raw_conf, min_val=0.20, max_val=0.95)

        val_score = cls._clamp(1.0 - (validation_metrics.get("sMAPE", 15.0) / 100.0), 0.0, 1.0)

        return ConfidenceReport(
            confidence=final_conf,
            confidence_level=confidence_level,
            validation_score=val_score,
            factors=factors,
            penalties=penalties,
            explanations=explanations,
        )

    @classmethod
    def calculate_descriptive_confidence(
        cls,
        completeness_pct: float = 100.0,
        n_samples: int = 10,
        ambiguity_level: float = 0.0,
    ) -> ConfidenceReport:
        """Calculate confidence for descriptive and exploratory analyses."""
        comp_factor = cls._clamp(completeness_pct / 100.0, 0.5, 1.0)
        size_factor = 0.75 if n_samples < 5 else 1.0
        amb_factor = max(0.50, 1.0 - ambiguity_level)

        final_conf = cls._clamp(comp_factor * size_factor * amb_factor, 0.30, 0.99)
        return ConfidenceReport(
            confidence=final_conf,
            confidence_level=1.0,
            validation_score=comp_factor,
            factors={"completeness": comp_factor, "sample_size": size_factor, "clarity": amb_factor},
        )

    @classmethod
    def calculate_anomaly_confidence(
        cls,
        n_samples: int = 10,
        n_features: int = 1,
        anomalies_found: int = 0,
        anomaly_rate: float = 0.05,
        method_suitability: float = 85.0,
        missing_rate: float = 0.0,
    ) -> ConfidenceReport:
        """Calculate confidence for anomaly detection outcomes."""
        penalties: List[str] = []
        explanations: List[str] = []
        factors: Dict[str, float] = {}

        # 1. Method suitability factor (0.5 to 1.0)
        suit_factor = cls._clamp(method_suitability / 100.0, 0.5, 1.0)
        factors["method_suitability"] = suit_factor
        explanations.append(f"Algorithm suitability score = {method_suitability:.1f}")

        # 2. Sample size factor
        if n_samples < 10:
            sample_factor = 0.65
            penalties.append(f"Small sample size for anomaly detection (N={n_samples} < 10)")
        elif n_samples < 30:
            sample_factor = 0.85
        else:
            sample_factor = 1.0
        factors["sample_size"] = sample_factor

        # 3. Anomaly rate reasonableness (between 0.1% and 25%)
        if anomaly_rate > 0.30:
            rate_factor = 0.70
            penalties.append(f"High anomaly rate ({anomaly_rate*100:.1f}%), potential noise contamination")
        elif anomaly_rate == 0.0:
            rate_factor = 0.85
            explanations.append("No anomalous observations detected")
        else:
            rate_factor = 0.95
        factors["rate_stability"] = rate_factor

        # 4. Missingness factor
        comp_factor = max(0.70, 1.0 - missing_rate)
        factors["completeness"] = comp_factor

        raw_conf = suit_factor * sample_factor * rate_factor * comp_factor
        final_conf = cls._clamp(raw_conf, min_val=0.25, max_val=0.95)

        return ConfidenceReport(
            confidence=final_conf,
            confidence_level=1.0,
            validation_score=suit_factor,
            factors=factors,
            penalties=penalties,
            explanations=explanations,
        )

    @classmethod
    def calculate_clustering_confidence(
        cls,
        silhouette_score: float = 0.5,
        davies_bouldin_score: float = 1.0,
        n_samples: int = 10,
        n_features: int = 2,
        k_clusters: int = 2,
        noise_ratio: float = 0.0,
        missing_rate: float = 0.0,
    ) -> ConfidenceReport:
        """Calculate confidence for clustering and segmentation outcomes."""
        penalties: List[str] = []
        explanations: List[str] = []
        factors: Dict[str, float] = {}

        # 1. Silhouette score quality (mapped from [-1, 1] to [0.2, 0.95])
        sil_clamped = cls._clamp((silhouette_score + 1.0) / 2.0, 0.0, 1.0)
        sil_factor = 0.30 + 0.65 * sil_clamped
        factors["silhouette_quality"] = sil_factor
        explanations.append(f"Holdout Silhouette Score = {silhouette_score:.3f}")

        # 2. Sample size adequacy
        sample_ratio = n_samples / max(1, n_features * k_clusters)
        if sample_ratio < 3:
            sample_factor = 0.65
            penalties.append(f"Low sample-to-parameter ratio ({sample_ratio:.1f} < 3)")
        elif sample_ratio < 10:
            sample_factor = 0.85
        else:
            sample_factor = 1.0
        factors["sample_size"] = sample_factor

        # 3. Noise penalty (if DBSCAN has >20% noise)
        noise_factor = max(0.60, 1.0 - (noise_ratio * 1.5))
        if noise_ratio > 0.20:
            penalties.append(f"Significant noise observations ({noise_ratio*100:.1f}%)")
        factors["density_coherence"] = noise_factor

        # 4. Missingness factor
        comp_factor = max(0.70, 1.0 - missing_rate)
        factors["completeness"] = comp_factor

        raw_conf = sil_factor * sample_factor * noise_factor * comp_factor
        final_conf = cls._clamp(raw_conf, min_val=0.20, max_val=0.95)

        return ConfidenceReport(
            confidence=final_conf,
            confidence_level=1.0,
            validation_score=sil_clamped,
            factors=factors,
            penalties=penalties,
            explanations=explanations,
        )

    @classmethod
    def calculate_statistical_relationship_confidence(
        cls,
        n_samples: int = 10,
        n_pairs: int = 1,
        top_effect_size: float = 0.5,
        min_adjusted_p: float = 0.01,
        missing_rate: float = 0.0,
        outlier_sensitivity: bool = False,
    ) -> ConfidenceReport:
        """Calculate confidence for statistical relationship & dependency outcomes."""
        penalties: List[str] = []
        explanations: List[str] = []
        factors: Dict[str, float] = {}

        # 1. Sample size adequacy
        if n_samples < 10:
            sample_factor = 0.65
            penalties.append(f"Small sample size for statistical testing (N={n_samples} < 10)")
        elif n_samples < 30:
            sample_factor = 0.85
        else:
            sample_factor = 1.0
        factors["sample_size"] = sample_factor

        # 2. Effect size & significance factor
        p_clamped = cls._clamp(min_adjusted_p, 0.0, 1.0)
        sig_factor = 0.50 + 0.45 * (1.0 - p_clamped) * cls._clamp(top_effect_size, 0.0, 1.0)
        factors["effect_significance"] = sig_factor
        explanations.append(f"Top effect size = {top_effect_size:.3f}, adjusted p = {min_adjusted_p:.4f}")

        # 3. Outlier sensitivity penalty
        outlier_factor = 0.85 if outlier_sensitivity else 1.0
        if outlier_sensitivity:
            penalties.append("Discrepancy detected between linear and rank associations (potential outlier influence)")
        factors["outlier_stability"] = outlier_factor

        # 4. Multiple testing load penalty (if testing > 50 pairs)
        mult_factor = 0.90 if n_pairs > 50 else 1.0
        factors["multiple_testing"] = mult_factor

        # 5. Data completeness
        comp_factor = max(0.70, 1.0 - missing_rate)
        factors["completeness"] = comp_factor

        raw_conf = sample_factor * sig_factor * outlier_factor * mult_factor * comp_factor
        final_conf = cls._clamp(raw_conf, min_val=0.25, max_val=0.95)

        return ConfidenceReport(
            confidence=final_conf,
            confidence_level=1.0,
            validation_score=sig_factor,
            factors=factors,
            penalties=penalties,
            explanations=explanations,
        )

    @classmethod
    def calculate_eda_confidence(
        cls,
        n_rows: int = 10,
        n_cols: int = 2,
        missing_rate: float = 0.0,
        unusable_ratio: float = 0.0,
        parse_success_rate: float = 1.0,
    ) -> ConfidenceReport:
        """
        Calculate confidence for exploratory data analysis and profiling.
        Separates epistemic profiling confidence from underlying data quality.
        """
        penalties: List[str] = []
        explanations: List[str] = []
        factors: Dict[str, float] = {}

        # 1. Sample Size Adequacy (N observations)
        if n_rows < 5:
            sample_factor = 0.65
            penalties.append(f"Very small row count (N={n_rows} < 5) limits statistical generalization")
        elif n_rows < 20:
            sample_factor = 0.85
        else:
            sample_factor = 1.0
        factors["sample_size"] = sample_factor

        # 2. Schema Clarity / Feature Count
        if n_cols == 1:
            col_factor = 0.80
            penalties.append("Single-column dataset limits multivariate profiling")
        else:
            col_factor = 1.0
        factors["schema_richness"] = col_factor

        # 3. Parseability & Type Certainty
        parse_factor = max(0.60, min(1.0, parse_success_rate))
        if parse_success_rate < 0.90:
            penalties.append(f"Some column values required type coercion/cleaning ({parse_success_rate*100:.1f}% success)")
        factors["type_certainty"] = parse_factor

        # 4. Usable Feature Availability
        usable_factor = max(0.60, 1.0 - unusable_ratio)
        factors["feature_usability"] = usable_factor

        # 5. Data Completeness Factor
        comp_factor = max(0.70, 1.0 - (missing_rate * 0.5))
        factors["completeness"] = comp_factor

        raw_conf = sample_factor * col_factor * parse_factor * usable_factor * comp_factor
        final_conf = cls._clamp(raw_conf, min_val=0.30, max_val=0.98)

        return ConfidenceReport(
            confidence=final_conf,
            confidence_level=1.0,
            validation_score=parse_factor,
            factors=factors,
            penalties=penalties,
            explanations=explanations,
        )




