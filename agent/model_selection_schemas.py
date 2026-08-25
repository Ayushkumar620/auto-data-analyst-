"""
Pydantic Schemas for Intelligent ML Model Selection.

Defines:
- MLTaskType & DataModality Enums
- ModelSelectionRequest: Input requirements, constraints, and dataset context
- ModelCandidate: Candidate algorithm metadata, suitability score, requirements, and hyperparameters
- ModelSelectionResult: Output model comparison plan, selected model, evaluation metrics, and evidence
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator

from agent.dataset_knowledge import DatasetKnowledge
from agent.intent import UserIntent
from agent.schemas import Evidence


class MLTaskType(str, Enum):
    """Machine learning task categories."""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    CLUSTERING = "clustering"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    IMAGE_CLASSIFICATION = "image_classification"
    ANOMALY_DETECTION = "anomaly_detection"
    UNKNOWN = "unknown"


class DataModality(str, Enum):
    """Input data structural modality."""
    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    IMAGE = "image"
    TEXT = "text"
    SPATIAL = "spatial"
    UNSUPPORTED = "unsupported"


class ModelCandidate(BaseModel):
    """Candidate algorithm specification and suitability evaluation."""
    model_name: str
    model_family: str  # linear, tree, ensemble, neural, convolutional, cluster, forecasting, neighbors
    supported_tasks: List[str] = Field(default_factory=list)
    suitability_score: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""
    requirements: List[str] = Field(default_factory=list)
    hyperparameter_space: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("suitability_score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("suitability_score must be between 0.0 and 1.0")
        return round(float(v), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "supported_tasks": self.supported_tasks,
            "suitability_score": self.suitability_score,
            "reason": self.reason,
            "requirements": self.requirements,
            "hyperparameter_space": self.hyperparameter_space,
        }


class ModelSelectionRequest(BaseModel):
    """Request payload specifying problem context, target, and constraints."""
    task_type: Optional[Union[MLTaskType, str]] = None
    dataset_knowledge: Optional[Union[DatasetKnowledge, Dict[str, Any]]] = None
    target_column: Optional[str] = None
    feature_columns: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    optimization_metric: Optional[str] = None
    max_training_time: Optional[float] = None
    preferred_interpretability: str = "medium"  # high, medium, low
    data_modality: Optional[Union[DataModality, str]] = None
    user_intent: Optional[Union[UserIntent, Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        dk_dict = self.dataset_knowledge.to_dict() if hasattr(self.dataset_knowledge, "to_dict") else self.dataset_knowledge
        ui_dict = self.user_intent.to_dict() if hasattr(self.user_intent, "to_dict") else self.user_intent
        task_str = self.task_type.value if isinstance(self.task_type, MLTaskType) else self.task_type
        mod_str = self.data_modality.value if isinstance(self.data_modality, DataModality) else self.data_modality
        return {
            "task_type": task_str,
            "dataset_knowledge": dk_dict,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "constraints": self.constraints,
            "optimization_metric": self.optimization_metric,
            "max_training_time": self.max_training_time,
            "preferred_interpretability": self.preferred_interpretability,
            "data_modality": mod_str,
            "user_intent": ui_dict,
        }


class ModelSelectionResult(BaseModel):
    """Output plan from Intelligent Model Selection Agent."""
    selected_model: Optional[str] = None
    candidates: List[ModelCandidate] = Field(default_factory=list)
    task_type: str = "regression"
    data_modality: str = "tabular"
    evaluation_metric: str = "r2"
    secondary_metrics: List[str] = Field(default_factory=list)
    selection_reason: str = ""
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    warnings: List[str] = Field(default_factory=list)
    leakage_warnings: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    comparison_plan: Dict[str, Any] = Field(default_factory=dict)
    target_column: Optional[str] = None
    feature_columns: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_model": self.selected_model,
            "candidates": [c.to_dict() for c in self.candidates],
            "task_type": self.task_type,
            "data_modality": self.data_modality,
            "evaluation_metric": self.evaluation_metric,
            "secondary_metrics": self.secondary_metrics,
            "selection_reason": self.selection_reason,
            "confidence": round(float(self.confidence), 4),
            "warnings": self.warnings,
            "leakage_warnings": self.leakage_warnings,
            "evidence": [e.to_dict() if isinstance(e, Evidence) else e for e in self.evidence],
            "comparison_plan": self.comparison_plan,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
        }
