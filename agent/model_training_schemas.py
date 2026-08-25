"""
Pydantic Schemas for Model Training, Evaluation, and Comparison Engine.

Defines:
- TrainingRequest: Input configuration, datasets, target, features, and validation strategies
- TrainingResult: Single model training metrics, validation scores, and artifact metadata
- ModelComparisonResult: Multi-algorithm benchmarking summary, ranking, and selected best model
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from agent.model_selection_schemas import ModelCandidate
from agent.schemas import Evidence


class TrainingRequest(BaseModel):
    """Configuration payload for training and validating candidate models."""
    target_column: str
    feature_columns: List[str] = Field(default_factory=list)
    task_type: str = "regression"  # regression, binary_classification, multiclass_classification, time_series_forecasting, clustering
    candidate_models: List[Union[str, ModelCandidate, Dict[str, Any]]] = Field(default_factory=list)
    validation_strategy: str = "5_fold_cv"  # 5_fold_cv, stratified_5_fold, time_series_split, train_val_test
    optimization_metric: str = "r2"  # r2, rmse, mae, f1, roc_auc, accuracy, silhouette
    preprocessing_config: Dict[str, Any] = Field(default_factory=dict)
    random_state: int = 42
    training_constraints: Dict[str, Any] = Field(default_factory=lambda: {"max_training_time_sec": 60, "max_models": 8})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "task_type": self.task_type,
            "candidate_models": [c.to_dict() if hasattr(c, "to_dict") else c for c in self.candidate_models],
            "validation_strategy": self.validation_strategy,
            "optimization_metric": self.optimization_metric,
            "preprocessing_config": self.preprocessing_config,
            "random_state": self.random_state,
            "training_constraints": self.training_constraints,
        }


class TrainingResult(BaseModel):
    """Comprehensive output metrics and artifact metadata for an individual trained model."""
    model_id: str
    model_name: str
    model_family: str = "ensemble"
    task_type: str = "regression"
    target: str
    features: List[str] = Field(default_factory=list)
    training_metrics: Dict[str, float] = Field(default_factory=dict)
    validation_metrics: Dict[str, float] = Field(default_factory=dict)
    primary_metric_name: str = "r2"
    primary_metric_value: float = 0.0
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    training_time_ms: float = 0.0
    model_artifact_path: Optional[str] = None
    feature_metadata: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    warnings: List[str] = Field(default_factory=list)
    status: str = "success"  # success, failed, partial
    error_message: Optional[str] = None
    overfitting_detected: bool = False
    overfitting_warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "task_type": self.task_type,
            "target": self.target,
            "features": self.features,
            "training_metrics": {k: round(float(v), 4) for k, v in self.training_metrics.items()},
            "validation_metrics": {k: round(float(v), 4) for k, v in self.validation_metrics.items()},
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_value": round(float(self.primary_metric_value), 4),
            "validation_results": self.validation_results,
            "training_time_ms": round(float(self.training_time_ms), 2),
            "model_artifact_path": self.model_artifact_path,
            "feature_metadata": self.feature_metadata,
            "evidence": [e.to_dict() if isinstance(e, Evidence) else e for e in self.evidence],
            "confidence": round(float(self.confidence), 4),
            "warnings": self.warnings,
            "status": self.status,
            "error_message": self.error_message,
            "overfitting_detected": self.overfitting_detected,
            "overfitting_warning": self.overfitting_warning,
        }


class ModelComparisonResult(BaseModel):
    """Aggregated comparison report across all trained candidates with winner selection."""
    candidates: List[TrainingResult] = Field(default_factory=list)
    ranking: List[Dict[str, Any]] = Field(default_factory=list)
    best_model: Optional[TrainingResult] = None
    optimization_metric: str = "r2"
    selection_reason: str = ""
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    status: str = "success"  # success, partial, failed
    warnings: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "ranking": self.ranking,
            "best_model": self.best_model.to_dict() if self.best_model else None,
            "optimization_metric": self.optimization_metric,
            "selection_reason": self.selection_reason,
            "evidence": [e.to_dict() if isinstance(e, Evidence) else e for e in self.evidence],
            "confidence": round(float(self.confidence), 4),
            "status": self.status,
            "warnings": self.warnings,
            "created_at": self.created_at,
        }
