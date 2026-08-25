"""
Schemas for Model Monitoring, Data Drift, and Model Performance Tracking.

Provides standardized, type-safe Pydantic contracts for:
- DriftThresholdConfig
- DriftRequest / MonitoringRequest
- FeatureDriftResult
- DatasetDriftReport
- ModelPerformanceReport
- PredictionDriftReport
- MonitoringResult
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from agent.schemas import ClaimType, Evidence


# ==============================================================================
# 1. Severity & Threshold Configuration
# ==============================================================================

class DriftSeverity(str, Enum):
    """Graduated severity levels for data and model drift."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DriftThresholdConfig(BaseModel):
    """Configurable statistical thresholds for drift and degradation detection."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    numeric_p_value_threshold: float = Field(
        default=0.05, description="Significance level (alpha) for 2-sample Kolmogorov-Smirnov test (p < alpha indicates drift)."
    )
    numeric_psi_threshold: float = Field(
        default=0.20, description="Population Stability Index (PSI) threshold for numeric features (PSI >= 0.2 indicates significant shift)."
    )
    categorical_p_value_threshold: float = Field(
        default=0.05, description="Significance level (alpha) for Chi-Square test of homogeneity."
    )
    categorical_psi_threshold: float = Field(
        default=0.20, description="Population Stability Index (PSI) threshold for categorical features."
    )
    missing_rate_delta_threshold: float = Field(
        default=0.10, description="Maximum allowable absolute increase in missing value percentage before flagging degradation (e.g. 0.10 = 10%)."
    )
    performance_degradation_threshold: float = Field(
        default=0.10, description="Maximum relative or absolute metric drop (e.g. 10% F1 or R2 drop) before flagging model degradation."
    )
    prediction_drift_threshold: float = Field(
        default=0.10, description="Threshold for prediction distribution divergence or mean shift."
    )

    def to_dict(self) -> Dict[str, float]:
        return {
            "numeric_p_value_threshold": self.numeric_p_value_threshold,
            "numeric_psi_threshold": self.numeric_psi_threshold,
            "categorical_p_value_threshold": self.categorical_p_value_threshold,
            "categorical_psi_threshold": self.categorical_psi_threshold,
            "missing_rate_delta_threshold": self.missing_rate_delta_threshold,
            "performance_degradation_threshold": self.performance_degradation_threshold,
            "prediction_drift_threshold": self.prediction_drift_threshold,
        }


# ==============================================================================
# 2. Monitoring Requests & Individual Feature Results
# ==============================================================================

class DriftRequest(BaseModel):
    """Request payload for model monitoring and data drift evaluation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str
    current_dataset: Any  # pd.DataFrame, dict, or records
    reference_dataset: Optional[Any] = None  # Optional explicit baseline; if omitted, registry profile is used
    feature_columns: Optional[List[str]] = None
    target_column: Optional[str] = None
    threshold_config: DriftThresholdConfig = Field(default_factory=DriftThresholdConfig)
    compute_predictions: bool = True


class FeatureDriftResult(BaseModel):
    """Statistical drift assessment for an individual feature column."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    feature_name: str
    drift_detected: bool
    drift_score: float
    statistical_test: str  # "kolmogorov_smirnov", "chi_square", "psi", "missing_rate_delta"
    p_value: Optional[float] = None
    reference_statistics: Dict[str, Any] = Field(default_factory=dict)
    current_statistics: Dict[str, Any] = Field(default_factory=dict)
    threshold: float
    severity: DriftSeverity = DriftSeverity.NONE
    evidence: Optional[Evidence] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "drift_detected": self.drift_detected,
            "drift_score": round(float(self.drift_score), 4),
            "statistical_test": self.statistical_test,
            "p_value": round(float(self.p_value), 4) if self.p_value is not None else None,
            "reference_statistics": self.reference_statistics,
            "current_statistics": self.current_statistics,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "confidence": self.confidence,
        }


# ==============================================================================
# 3. Aggregated Reports (Data, Performance, Prediction)
# ==============================================================================

class DatasetDriftReport(BaseModel):
    """Aggregated dataset drift summary across all checked features and schema."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str = "current_data"
    reference_dataset_id: str = "reference_data"
    features_checked: List[str] = Field(default_factory=list)
    drifted_features: List[str] = Field(default_factory=list)
    drift_percentage: float = 0.0
    overall_drift: bool = False
    schema_drift_detected: bool = False
    schema_changes: Dict[str, Any] = Field(default_factory=dict)
    data_quality_changes: Dict[str, Any] = Field(default_factory=dict)
    feature_results: Dict[str, FeatureDriftResult] = Field(default_factory=dict)
    severity: DriftSeverity = DriftSeverity.NONE
    warnings: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    evidence: List[Evidence] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "reference_dataset_id": self.reference_dataset_id,
            "features_checked": self.features_checked,
            "drifted_features": self.drifted_features,
            "drift_percentage": round(float(self.drift_percentage), 2),
            "overall_drift": self.overall_drift,
            "schema_drift_detected": self.schema_drift_detected,
            "schema_changes": self.schema_changes,
            "data_quality_changes": self.data_quality_changes,
            "feature_results": {k: v.to_dict() for k, v in self.feature_results.items()},
            "severity": self.severity.value,
            "warnings": self.warnings,
            "confidence": self.confidence,
        }


class ModelPerformanceReport(BaseModel):
    """Model performance comparison against reference metrics when ground-truth labels exist."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str
    reference_metrics: Dict[str, float] = Field(default_factory=dict)
    current_metrics: Dict[str, float] = Field(default_factory=dict)
    metric_changes: Dict[str, float] = Field(default_factory=dict)
    degradation_detected: bool = False
    target_monitoring_status: str = "evaluated"  # "evaluated", "unavailable"
    evaluation_dataset_rows: int = 0
    confidence: float = 1.0
    evidence: List[Evidence] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "reference_metrics": self.reference_metrics,
            "current_metrics": self.current_metrics,
            "metric_changes": self.metric_changes,
            "degradation_detected": self.degradation_detected,
            "target_monitoring_status": self.target_monitoring_status,
            "evaluation_dataset_rows": self.evaluation_dataset_rows,
            "confidence": self.confidence,
        }


class PredictionDriftReport(BaseModel):
    """Drift evaluation on model prediction outputs and probability distributions."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str
    prediction_drift_detected: bool = False
    statistical_test: str = "kolmogorov_smirnov"
    drift_score: float = 0.0
    p_value: Optional[float] = None
    reference_prediction_stats: Dict[str, Any] = Field(default_factory=dict)
    current_prediction_stats: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    evidence: List[Evidence] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prediction_drift_detected": self.prediction_drift_detected,
            "statistical_test": self.statistical_test,
            "drift_score": round(float(self.drift_score), 4),
            "p_value": round(float(self.p_value), 4) if self.p_value is not None else None,
            "reference_prediction_stats": self.reference_prediction_stats,
            "current_prediction_stats": self.current_prediction_stats,
            "confidence": self.confidence,
        }


# ==============================================================================
# 4. Master Monitoring Result Object
# ==============================================================================

class MonitoringResult(BaseModel):
    """Complete, end-to-end multi-dimensional model monitoring assessment."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str
    status: str = "success"  # "success", "partial", "failed"
    overall_severity: DriftSeverity = DriftSeverity.NONE
    data_drift: Optional[DatasetDriftReport] = None
    prediction_drift: Optional[PredictionDriftReport] = None
    performance_drift: Optional[ModelPerformanceReport] = None
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: float = 1.0
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "status": self.status,
            "overall_severity": self.overall_severity.value,
            "data_drift": self.data_drift.to_dict() if self.data_drift else None,
            "prediction_drift": self.prediction_drift.to_dict() if self.prediction_drift else None,
            "performance_drift": self.performance_drift.to_dict() if self.performance_drift else None,
            "data_quality": self.data_quality,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }

