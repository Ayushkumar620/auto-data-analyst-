"""Model Registry Agent - Manages deployed models, versions, and live inference."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

from agent.base import BaseAgent
from agent.schemas import AgentResult, ClaimType, ErrorCategory, Evidence
from backend.app.ml.registry import ModelRegistry


class ModelRegistryAgent(BaseAgent):
    """
    Autonomous Model Registry Agent.
    Orchestrates the querying, deployment lifecycle, version tracking,
    and inference execution of registered ML, ANN, and CNN models.
    """
    name = "Model Registry Agent"
    role = "mlops_engineer"
    description = "Tracks model versions, queries model artifacts, and runs live inference on deployed models."

    def __init__(self, data=None):
        super().__init__(data=data)
        self.registry = ModelRegistry()

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """
        Execute Model Registry task.
        Task parameters:
            - action: str ('list', 'get', 'predict', 'deploy', 'archive', 'delete')
            - model_id: str (required for get, predict, deploy, archive, delete)
            - data: pd.DataFrame or dict (required for predict)
            - family: str (optional filter for list)
            - problem_type: str (optional filter for list)
        """
        self._start()
        action = task.get("action", "list")
        model_id = task.get("model_id")

        try:
            if action in ("list", "list_models"):
                models = self.registry.list_models(
                    family=task.get("family"),
                    problem_type=task.get("problem_type"),
                    status=task.get("status"),
                )
                evidence = [
                    self.make_evidence(
                        method="model_registry_query",
                        data_ref={"total_models": len(models)},
                        confidence=1.0,
                        claim_type=ClaimType.FACT,
                    )
                ]
                return self._finish(
                    result={"models": models, "count": len(models)},
                    evidence=evidence,
                    confidence=1.0,
                )

            elif action in ("get", "get_metadata"):
                if not model_id:
                    return self._error("model_id is required for action 'get'.", category=ErrorCategory.INPUT_VALIDATION)
                meta = self.registry.get_metadata(model_id)
                if not meta:
                    return self._error(f"Model '{model_id}' not found.", category=ErrorCategory.INPUT_VALIDATION)
                return self._finish(
                    result={"metadata": meta.to_dict()},
                    evidence=[
                        self.make_evidence(
                            method="model_registry_lookup",
                            data_ref={"model_id": model_id, "name": meta.name, "version": meta.version},
                            confidence=1.0,
                            claim_type=ClaimType.FACT,
                        )
                    ],
                    confidence=1.0,
                )

            elif action == "predict":
                if not model_id:
                    return self._error("model_id is required for action 'predict'.", category=ErrorCategory.INPUT_VALIDATION)
                input_data = task.get("data") if task.get("data") is not None else self.data
                if input_data is None:
                    return self._error("Missing required input data for prediction.", category=ErrorCategory.INPUT_VALIDATION)

                prediction_res = self.registry.predict(model_id, input_data)
                evidence = [
                    self.make_evidence(
                        method="model_artifact_inference",
                        data_ref={
                            "model_id": model_id,
                            "sample_count": prediction_res.get("sample_count", 0),
                        },
                        confidence=0.95,
                        claim_type=ClaimType.FACT,
                    )
                ]
                return self._finish(
                    result=prediction_res,
                    evidence=evidence,
                    confidence=0.95,
                )

            elif action in ("deploy", "activate"):
                if not model_id:
                    return self._error("model_id is required.", category=ErrorCategory.INPUT_VALIDATION)
                success = self.registry.set_status(model_id, "active")
                if not success:
                    return self._error(f"Model '{model_id}' not found.", category=ErrorCategory.INPUT_VALIDATION)
                return self._finish(
                    result={"status": "active", "model_id": model_id},
                    evidence=[],
                    confidence=1.0,
                )

            elif action == "archive":
                if not model_id:
                    return self._error("model_id is required.", category=ErrorCategory.INPUT_VALIDATION)
                success = self.registry.set_status(model_id, "archived")
                if not success:
                    return self._error(f"Model '{model_id}' not found.", category=ErrorCategory.INPUT_VALIDATION)
                return self._finish(
                    result={"status": "archived", "model_id": model_id},
                    evidence=[],
                    confidence=1.0,
                )

            else:
                return self._error(f"Unknown registry action: '{action}'.", category=ErrorCategory.INPUT_VALIDATION)

        except Exception as exc:
            return self._error(f"Model Registry operation failed: {str(exc)}", category=ErrorCategory.COMPUTATION)
