"""
Universal Data Transformation & Feature Engineering Agent.

Orchestrates PreExecutionValidator, TransformationEngine, ConfidenceCalculator,
and ResultValidator into the canonical AgentResult lifecycle.
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


class TransformationAgent(BaseAgent):
    """
    Autonomous Data Transformation & Feature Engineering Agent.
    Transforms raw tabular data into clean, encoded, scaled feature matrices
    with explainable plans, leakage prevention, and fit/transform state management.
    """

    name = "Transformation Agent"
    description = "Transforms tabular data into model-ready numerical feature matrices with imputation, encoding, scaling, and feature engineering."
    role = "transformation"

    def run(self, task: Dict[str, Any]) -> AgentResult:
        self._start()
        try:
            from agent.pre_execution_validator import PreExecutionValidator
            from agent.confidence_calculator import ConfidenceCalculator
            from agent.result_validator import ResultValidator
            from agent.transformation_engine import TransformationEngine, TransformationPlan, TransformationState

            data = task.get("data")
            target = task.get("target")
            features = task.get("features") or task.get("selected_columns")
            task_type = task.get("task_type")
            config = task.get("config") or {}
            action = task.get("action", "fit_transform")  # fit, transform, fit_transform
            fitted_state = task.get("state") or task.get("fitted_state")

            # 1. Pre-execution validation
            pre_audit = PreExecutionValidator.validate(
                data,
                task_type="transformation",
                target=target,
                feature_columns=features,
                agent_name=self.name,
            )
            if not pre_audit.is_valid:
                err = pre_audit.error
                return self._error(
                    message=err.user_message if err else "Transformation pre-execution validation failed.",
                    code=err.code if err else "VALIDATION_FAILURE",
                    category=err.category if err else ErrorCategory.DATA_INVALID,
                    details=err.technical_details if err else {},
                )

            engine = TransformationEngine()

            # 2. Execution based on action
            if action == "fit":
                state: TransformationState = engine.fit(data, target=target, features=features, task_type=task_type, config=config)
                plan: TransformationPlan = engine.generate_plan(state)
                result_payload = {
                    "action": "fit",
                    "state": state.to_dict(),
                    "transformation_plan": plan.to_dict(),
                    "summary": {
                        "fitted_rows": state.fitted_row_count,
                        "fitted_columns": state.fitted_columns,
                        "selected_features_count": len(state.selected_features),
                        "generated_features_count": len(state.output_columns),
                        "target_name": state.target_name,
                    },
                }
            elif action == "transform":
                if not fitted_state:
                    return self._error("Fitted state is required for 'transform' action.", category=ErrorCategory.DATA_INVALID)
                transformed_df: pd.DataFrame = engine.transform(data, fitted_state)
                result_payload = {
                    "action": "transform",
                    "transformed_shape": list(transformed_df.shape),
                    "columns": list(transformed_df.columns),
                    "sample_records": transformed_df.iloc[:5].to_dict(orient="records"),
                }
            else:  # fit_transform
                transformed_df, state, plan = engine.fit_transform(data, target=target, features=features, task_type=task_type, config=config)
                result_payload = {
                    "action": "fit_transform",
                    "transformed_shape": list(transformed_df.shape),
                    "columns": list(transformed_df.columns),
                    "state": state.to_dict(),
                    "transformation_plan": plan.to_dict(),
                    "feature_metadata": state.feature_metadata,
                    "sample_records": transformed_df.iloc[:5].to_dict(orient="records"),
                    "summary": {
                        "input_rows": len(data) if isinstance(data, pd.DataFrame) else 0,
                        "input_columns": len(data.columns) if isinstance(data, pd.DataFrame) else 0,
                        "transformed_rows": len(transformed_df),
                        "transformed_features": len(transformed_df.columns),
                        "selected_features_count": len(state.selected_features),
                        "excluded_features_count": len(state.excluded_features),
                    },
                }

            # 3. Canonical Evidence Generation (ClaimType.OBSERVATION)
            evidence_list: List[Evidence] = []
            if action in ("fit", "fit_transform"):
                evidence_list.append(
                    self.make_evidence(
                        method="transformation.plan.summary",
                        data_ref={
                            "selected_features": result_payload.get("transformation_plan", {}).get("selected_features", []),
                            "generated_features": result_payload.get("transformation_plan", {}).get("generated_features", []),
                            "scaling": result_payload.get("transformation_plan", {}).get("scaling_strategy", {}),
                            "encoding": result_payload.get("transformation_plan", {}).get("encoding_strategy", {}),
                        },
                        confidence=0.95,
                        claim_type=ClaimType.OBSERVATION,
                        raw_value=len(result_payload.get("transformation_plan", {}).get("generated_features", [])),
                    )
                )

            # 4. Confidence Calculation
            n_rows = len(data) if isinstance(data, pd.DataFrame) else 0
            n_features = len(result_payload.get("columns", []))
            conf_rep = ConfidenceCalculator.calculate_transformation_confidence(
                n_rows=n_rows,
                n_features=n_features,
                missing_rate=0.0,
                unusable_ratio=0.0,
                coercion_success_rate=1.0,
            )

            raw_res = self._finish(
                result_payload,
                evidence=evidence_list,
                confidence=conf_rep.confidence,
                model_used="TransformationEngine",
            )

            # 5. Result Validation & Invariant Repair
            repaired_res, _ = ResultValidator().repair(raw_res, context={"data": data})
            return repaired_res

        except Exception as e:
            return self._error(str(e), category=ErrorCategory.COMPUTATION)