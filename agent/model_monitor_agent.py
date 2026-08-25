"""
Autonomous Model Monitoring & Drift Agent.

Coordinates:
Task Input (model_id, current_data, reference_data, thresholds)
      ↓
Model Registry Inspection
      ↓
ModelMonitoringEngine Statistical Evaluation
      ↓
Evidence Grounding
      ↓
Standardized AgentResult
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from agent.base import BaseAgent
from agent.model_monitoring_engine import ModelMonitoringEngine
from agent.model_monitoring_schemas import (
    DriftRequest,
    DriftSeverity,
    DriftThresholdConfig,
    MonitoringResult,
)
from agent.schemas import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)
from backend.app.ml.registry import ModelRegistry


class ModelMonitorAgent(BaseAgent):
    """
    Autonomous Model Monitoring Agent inspecting live distributions, schema consistency,
    prediction drift, and performance degradation against registered model baselines.
    """
    name = "Model Monitor Agent"
    role = "lead_mlops_engineer"
    description = "Monitors model data drift, schema consistency, prediction drift, and performance degradation using statistical hypothesis tests."

    def __init__(self, data: Optional[Any] = None, registry: Optional[ModelRegistry] = None):
        super().__init__(data=data)
        self.registry = registry or ModelRegistry()
        self.engine = ModelMonitoringEngine(registry=self.registry)

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """
        Execute model monitoring run.
        Task parameters:
            - model_id: str (required registered model ID)
            - current_data: pd.DataFrame | dict (required new batch)
            - reference_data: Optional[pd.DataFrame | dict]
            - feature_columns: Optional[list[str]]
            - target_column: Optional[str]
            - thresholds: Optional[dict]
        """
        self._start()
        model_id = task.get("model_id")
        if not model_id:
            return self._error("Missing required 'model_id' parameter for model monitoring.", category=ErrorCategory.INPUT_VALIDATION)

        current_data = task.get("current_data") if task.get("current_data") is not None else task.get("data", self.data)
        if current_data is None:
            return self._error("Missing required 'current_data' parameter for drift evaluation.", category=ErrorCategory.INPUT_VALIDATION)

        reference_data = task.get("reference_data")
        feature_cols = task.get("feature_columns")
        target_col = task.get("target_column")
        threshold_dict = task.get("thresholds", {})
        thresh_cfg = DriftThresholdConfig(**threshold_dict) if threshold_dict else DriftThresholdConfig()

        drift_request = DriftRequest(
            model_id=model_id,
            current_dataset=current_data,
            reference_dataset=reference_data,
            feature_columns=feature_cols,
            target_column=target_col,
            threshold_config=thresh_cfg,
        )

        try:
            result = self.engine.monitor(drift_request)
        except Exception as exc:
            return self._error(f"Model monitoring execution failed: {str(exc)}", category=ErrorCategory.COMPUTATION)

        if result.status == "failed":
            return self._error(
                message=f"Monitoring failed for model '{model_id}'.",
                category=ErrorCategory.COMPUTATION,
                details={"warnings": result.warnings},
            )

        output_dict = result.to_dict()
        summary_msg = (
            f"Monitoring completed for model '{model_id}' with overall severity: {result.overall_severity.value}. "
            f"Drifted features: {len(result.data_drift.drifted_features) if result.data_drift else 0} "
            f"({result.data_drift.drift_percentage if result.data_drift else 0.0}%). "
            f"Performance degradation: {'DETECTED' if result.performance_drift and result.performance_drift.degradation_detected else 'NONE'}."
        )

        if result.overall_severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL):
            return self._partial(
                result=output_dict,
                message=summary_msg,
                warnings=result.warnings + result.recommendations,
                evidence=result.evidence,
                confidence=result.confidence,
            )

        return self._finish(
            result=output_dict,
            evidence=result.evidence,
            confidence=result.confidence,
            metadata={
                "model_id": model_id,
                "overall_severity": result.overall_severity.value,
                "drift_percentage": result.data_drift.drift_percentage if result.data_drift else 0.0,
            },
        )
