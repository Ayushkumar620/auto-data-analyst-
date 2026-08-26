"""
Schemas for Autonomous Forecasting and What-If Scenario Analysis.

Provides Pydantic models for:
- ForecastRequest & ForecastSuitabilityResult
- ForecastPoint & ForecastResult
- WhatIfRequest, ScenarioResult & ScenarioComparison
"""
from __future__ import annotations

from enum import Enum
import time
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field

from agent.schemas import ClaimType, Evidence


class ForecastModelFamily(str, Enum):
    """Supported forecasting algorithm families."""
    NAIVE_LAST = "naive_last"
    SEASONAL_NAIVE = "seasonal_naive"
    MOVING_AVERAGE = "moving_average"
    LINEAR_TREND = "linear_trend"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    AUTOREGRESSIVE_ML = "autoregressive_ml"
    HOLT_WINTERS = "holt_winters"
    ARIMA_STATISTICAL = "arima_statistical"


class ForecastRequest(BaseModel):
    """Specification for a time-series forecasting request."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: Optional[Any] = None
    time_column: Optional[str] = None
    target_column: Optional[str] = None
    forecast_horizon: int = 6
    frequency: Optional[str] = None  # "D", "W", "M", "Q", "Y"
    confidence_level: float = 0.80
    grouping_columns: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    optimization_metric: str = "MAE"  # "MAE", "RMSE", "WAPE"
    model_candidates: List[str] = Field(default_factory=list)
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_column": self.time_column,
            "target_column": self.target_column,
            "forecast_horizon": self.forecast_horizon,
            "frequency": self.frequency,
            "confidence_level": self.confidence_level,
            "optimization_metric": self.optimization_metric,
        }


class ForecastSuitabilityResult(BaseModel):
    """Assessment of dataset readiness for time-series forecasting."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    suitable: bool
    score: float = Field(ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    detected_time_column: Optional[str] = None
    detected_target: Optional[str] = None
    detected_frequency: Optional[str] = None
    observation_count: int = 0
    time_range: Optional[Dict[str, str]] = None
    has_seasonality: bool = False
    has_trend: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ForecastPoint(BaseModel):
    """Single point in future forecast trajectory with uncertainty bounds."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: str
    prediction: float
    lower_bound: float
    upper_bound: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "prediction": round(float(self.prediction), 4),
            "lower_bound": round(float(self.lower_bound), 4),
            "upper_bound": round(float(self.upper_bound), 4),
        }


class ForecastResult(BaseModel):
    """Verified output of autonomous forecasting pipeline."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_id: str = Field(default_factory=lambda: f"fc_{uuid.uuid4().hex[:8]}")
    model_name: str
    model_family: str
    target: str
    time_column: str
    frequency: str
    forecast_horizon: int
    predictions: List[ForecastPoint] = Field(default_factory=list)
    confidence_level: float = 0.80
    validation_metrics: Dict[str, float] = Field(default_factory=dict)
    baseline_metrics: Dict[str, float] = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    slope: Optional[float] = None
    projected_change_pct: Optional[float] = None
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: float = 0.90
    status: str = "SUCCESS"  # "SUCCESS", "NOT_SUPPORTED", "FAILED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "target": self.target,
            "time_column": self.time_column,
            "frequency": self.frequency,
            "forecast_horizon": self.forecast_horizon,
            "predictions": [p.to_dict() for p in self.predictions],
            "confidence_level": self.confidence_level,
            "validation_metrics": {k: round(float(v), 4) for k, v in self.validation_metrics.items()},
            "baseline_metrics": {k: round(float(v), 4) for k, v in self.baseline_metrics.items()},
            "slope": round(float(self.slope), 4) if self.slope is not None else None,
            "projected_change_pct": self.projected_change_pct,
            "projected_change_percent": self.projected_change_pct,
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "limitations": self.limitations,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": round(float(self.confidence), 3),
            "status": self.status,
        }


# ==============================================================================
# What-If Scenario Models
# ==============================================================================

class WhatIfRequest(BaseModel):
    """Specification for a What-If counterfactual scenario simulation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: Optional[Any] = None
    target: Optional[str] = None
    scenario_name: str = "Custom Scenario"
    changed_variables: Dict[str, Any] = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)
    horizon: Optional[int] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "target": self.target,
            "changed_variables": self.changed_variables,
            "assumptions": self.assumptions,
        }


class ScenarioResult(BaseModel):
    """Deterministic output of a single What-If simulation scenario."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_name: str
    target_metric: str
    baseline_value: float
    scenario_value: float
    absolute_difference: float
    percentage_difference: float
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: float = 0.90

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "target_metric": self.target_metric,
            "baseline_value": round(float(self.baseline_value), 2),
            "scenario_value": round(float(self.scenario_value), 2),
            "absolute_difference": round(float(self.absolute_difference), 2),
            "percentage_difference": round(float(self.percentage_difference), 2),
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": round(float(self.confidence), 3),
        }


class ScenarioComparison(BaseModel):
    """Comparison matrix across multiple What-If scenarios."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_metric: str
    baseline_value: float
    scenarios: List[ScenarioResult] = Field(default_factory=list)
    ranked_scenarios: List[ScenarioResult] = Field(default_factory=list)
    summary: str = ""
    evidence: List[Evidence] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_metric": self.target_metric,
            "baseline_value": round(float(self.baseline_value), 2),
            "scenarios": [s.to_dict() for s in self.scenarios],
            "ranked_scenarios": [s.to_dict() for s in self.ranked_scenarios],
            "summary": self.summary,
            "evidence": [e.to_dict() for e in self.evidence],
        }

