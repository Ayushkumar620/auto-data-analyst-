"""
Universal Anomaly Detection Agent.

Executes autonomous outlier and anomaly detection on arbitrary datasets using
the canonical AnomalyDetectionEngine and returns a standardized AgentResult.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from agent.agent_result import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)
from agent.base import BaseAgent


class AnomalyDetectionAgent(BaseAgent):
    """
    Autonomous Anomaly Detection Agent.
    Identifies multivariate and univariate outliers with explainable factor decomposition.
    """

    name = "Anomaly Detection Agent"
    description = "Detects multivariate and univariate outliers and anomalies using data-driven algorithms."
    role = "anomaly_detection"

    def run(self, task: Dict[str, Any]) -> AgentResult:
        self._start()
        try:
            from agent.pre_execution_validator import PreExecutionValidator
            from agent.confidence_calculator import ConfidenceCalculator
            from agent.result_validator import ResultValidator
            from agent.anomaly_detection_engine import AnomalyDetectionEngine

            data = task.get("data")
            features = task.get("features") or task.get("feature_columns")
            contamination = task.get("contamination", "auto")
            method = task.get("method")

            pre_audit = PreExecutionValidator.validate(
                data,
                task_type="anomaly_detection",
                feature_columns=features,
                agent_name=self.name,
            )
            if not pre_audit.is_valid:
                err = pre_audit.error
                return self._error(
                    message=err.user_message if err else "Anomaly detection pre-validation failed.",
                    code=err.code if err else "VALIDATION_FAILURE",
                    category=err.category if err else ErrorCategory.DATA_INVALID,
                    details=err.technical_details if err else {},
                )

            engine = AnomalyDetectionEngine()
            result = engine.detect(
                data=data,
                features=features,
                contamination=contamination,
                method=method,
            )

            if "error" in result:
                return self._error(
                    message=result["error"],
                    code="ANOMALY_DETECTION_FAILED",
                    category=result.get("category", ErrorCategory.MODEL_FAILURE),
                    details=result,
                    output=result,
                )

            method_used = result.get("method", "robust_zscore")
            method_family = result.get("method_family", "statistical")
            n_rows = result.get("rows_analyzed", 0)
            n_anomalies = result.get("anomalies_found", 0)
            anomaly_rate = result.get("anomaly_rate", 0.0)

            evidence = [
                self.make_evidence(
                    method=f"anomaly.{method_family}.{method_used}",
                    data_ref={
                        "method": method_used,
                        "rows_analyzed": n_rows,
                        "anomalies_found": n_anomalies,
                        "anomaly_rate": anomaly_rate,
                        "features_used": result.get("features_used", []),
                    },
                    confidence=0.85,
                    claim_type=ClaimType.OBSERVATION,
                    raw_value={
                        "anomalies_found": n_anomalies,
                        "anomaly_rate": anomaly_rate,
                        "contamination": result.get("contamination"),
                    },
                )
            ]

            conf_rep = ConfidenceCalculator.calculate_anomaly_confidence(
                n_samples=n_rows,
                n_features=len(result.get("features_used", [])) or 1,
                anomalies_found=n_anomalies,
                anomaly_rate=anomaly_rate,
                method_suitability=85.0,
            )

            raw_res = self._finish(
                result,
                evidence=evidence,
                confidence=conf_rep.confidence,
                model_used=method_used,
            )
            repaired_res, _ = ResultValidator().repair(raw_res, context={"data": data})
            return repaired_res
        except Exception as e:
            return self._error(str(e), category=ErrorCategory.COMPUTATION)
