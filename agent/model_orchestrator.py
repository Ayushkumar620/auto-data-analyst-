"""
Unified Intelligent Model Orchestrator.

Coordinates:
User Command / Intent
      ↓
Dataset Knowledge
      ↓
Model Selection Agent
      ↓
Candidate Validation & Modality Checks
      ↓
┌─────────────────┬─────────────────┬─────────────────┐
│ Traditional ML  │       ANN       │       CNN       │
└─────────────────┴─────────────────┴─────────────────┘
      ↓
Parallel / Sequential Training & Cross-Validation
      ↓
Deterministic Metric Ranking & Winner Selection
      ↓
Model Registry Persistence (as production_candidate)
      ↓
Standardized AgentResult with Grounded Evidence
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid

import joblib
import numpy as np
import pandas as pd

from agent.ann_schemas import ANNConfig, auto_select_ann_architecture
from agent.base import BaseAgent
from agent.cnn_schemas import CNNConfig, auto_select_cnn_architecture
from agent.intent import UserIntent
from agent.model_selection_agent import ModelSelectionAgent
from agent.model_selection_schemas import (
    DataModality,
    MLTaskType,
    ModelCandidate,
    ModelSelectionRequest,
    ModelSelectionResult,
)
from agent.model_training_engine import (
    ANNTrainer,
    CNNTrainer,
    DataPreprocessor,
    ModelTrainingEngine,
    TraditionalMLTrainer,
)
from agent.model_training_schemas import (
    ModelComparisonResult,
    TrainingRequest,
    TrainingResult,
)
from agent.schemas import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)
from backend.app.ml.ann_engine import ANNEngine, ANNHyperparameters
from backend.app.ml.cnn_engine import CNNEngine, CNNHyperparameters
from backend.app.ml.registry import ModelArtifactMetadata, ModelRegistry


# ==============================================================================
# 1. Model Capability Schema & Unified Engine Interface
# ==============================================================================

@dataclass
class ModelCapability:
    """Capability specification declaring task, modality, and cost properties for a model engine."""
    model_family: str  # "traditional_ml", "ann", "cnn"
    model_name: str
    supported_tasks: List[str]
    supported_modalities: List[str]
    minimum_samples: int = 10
    interpretability_level: str = "medium"  # "high", "medium", "low", "black_box"
    training_cost: str = "low"  # "low", "medium", "high"
    prediction_cost: str = "low"
    requirements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_family": self.model_family,
            "model_name": self.model_name,
            "supported_tasks": self.supported_tasks,
            "supported_modalities": self.supported_modalities,
            "minimum_samples": self.minimum_samples,
            "interpretability_level": self.interpretability_level,
            "training_cost": self.training_cost,
            "prediction_cost": self.prediction_cost,
            "requirements": self.requirements,
        }


class BaseModelEngine(ABC):
    """Abstract common interface implemented by all model family adapters."""

    @property
    @abstractmethod
    def capability(self) -> ModelCapability:
        """Return declared engine capability metadata."""
        pass

    @abstractmethod
    def supports(self, candidate_name: str, task_type: str, modality: str = "tabular") -> bool:
        """Verify if this engine supports the specified algorithm, task, and data modality."""
        pass

    @abstractmethod
    def train_candidate(
        self,
        candidate: Union[str, ModelCandidate, Dict[str, Any]],
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        task_type: str,
        validation_strategy: str = "5_fold_cv",
        optimization_metric: str = "r2",
        random_state: int = 42,
    ) -> TrainingResult:
        """Execute cross-validation training and evaluate metrics for a candidate."""
        pass

    @abstractmethod
    def predict(
        self,
        model_obj: Any,
        preprocessor: Optional[Any],
        feature_columns: List[str],
        data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Perform schema-validated inference on new records."""
        pass


# ==============================================================================
# 2. Concrete Model Engine Adapters
# ==============================================================================

class TraditionalMLEngine(BaseModelEngine):
    """Unified engine adapter for Linear, Tree, Ensemble, SVM, and k-NN models."""

    def __init__(self, training_engine: Optional[ModelTrainingEngine] = None):
        self.training_engine = training_engine or ModelTrainingEngine()

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(
            model_family="traditional_ml",
            model_name="Traditional Machine Learning",
            supported_tasks=["regression", "binary_classification", "multiclass_classification", "clustering"],
            supported_modalities=["tabular"],
            minimum_samples=10,
            interpretability_level="high",
            training_cost="low",
            prediction_cost="low",
            requirements=["Numeric/categorical tabular features"],
        )

    def supports(self, candidate_name: str, task_type: str, modality: str = "tabular") -> bool:
        if modality.lower() not in ("tabular", "time_series"):
            return False
        name_lower = candidate_name.lower()
        if "ann" in name_lower or "cnn" in name_lower or "convolutional" in name_lower:
            return False
        return True

    def train_candidate(
        self,
        candidate: Union[str, ModelCandidate, Dict[str, Any]],
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        task_type: str,
        validation_strategy: str = "5_fold_cv",
        optimization_metric: str = "r2",
        random_state: int = 42,
    ) -> TrainingResult:
        return self.training_engine.train_and_validate_candidate(
            candidate=candidate,
            df=df,
            target_col=target_col,
            feature_cols=feature_cols,
            task_type=task_type,
            validation_strategy=validation_strategy,
            optimization_metric=optimization_metric,
            random_state=random_state,
        )

    def predict(
        self,
        model_obj: Any,
        preprocessor: Optional[Any],
        feature_columns: List[str],
        data: pd.DataFrame,
    ) -> Dict[str, Any]:
        df_feat = data[feature_columns].copy()
        X_t = preprocessor.transform(df_feat) if preprocessor else df_feat.to_numpy(dtype=float)
        preds = model_obj.predict(X_t)
        proba = model_obj.predict_proba(X_t).tolist() if hasattr(model_obj, "predict_proba") else None

        decoded_preds = preds.tolist()
        if preprocessor is not None and getattr(preprocessor, "target_encoder", None) is not None:
            try:
                decoded_preds = preprocessor.target_encoder.inverse_transform(preds).tolist()
            except Exception:
                pass

        return {"predictions": decoded_preds, "probabilities": proba}


class ANNEngineAdapter(BaseModelEngine):
    """Unified engine adapter for Deep Multi-Layer Perceptrons."""

    def __init__(self, training_engine: Optional[ModelTrainingEngine] = None):
        self.training_engine = training_engine or ModelTrainingEngine()

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(
            model_family="ann",
            model_name="Artificial Neural Network (ANN/MLP)",
            supported_tasks=["regression", "binary_classification", "multiclass_classification"],
            supported_modalities=["tabular"],
            minimum_samples=10,
            interpretability_level="black_box",
            training_cost="medium",
            prediction_cost="low",
            requirements=["StandardScaler normalization", "Encoded categorical variables"],
        )

    def supports(self, candidate_name: str, task_type: str, modality: str = "tabular") -> bool:
        if modality.lower() not in ("tabular", "time_series"):
            return False
        name_lower = candidate_name.lower()
        return "ann" in name_lower or "mlp" in name_lower or "neural" in name_lower

    def train_candidate(
        self,
        candidate: Union[str, ModelCandidate, Dict[str, Any]],
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        task_type: str,
        validation_strategy: str = "5_fold_cv",
        optimization_metric: str = "r2",
        random_state: int = 42,
    ) -> TrainingResult:
        return self.training_engine.train_and_validate_candidate(
            candidate=candidate,
            df=df,
            target_col=target_col,
            feature_cols=feature_cols,
            task_type=task_type,
            validation_strategy=validation_strategy,
            optimization_metric=optimization_metric,
            random_state=random_state,
        )

    def predict(
        self,
        model_obj: Any,
        preprocessor: Optional[Any],
        feature_columns: List[str],
        data: pd.DataFrame,
    ) -> Dict[str, Any]:
        df_feat = data[feature_columns].copy()
        X_t = preprocessor.transform(df_feat) if preprocessor else df_feat.to_numpy(dtype=float)
        preds = model_obj.predict(X_t)
        proba = model_obj.predict_proba(X_t).tolist() if hasattr(model_obj, "predict_proba") else None

        decoded_preds = preds.tolist()
        if preprocessor is not None and getattr(preprocessor, "target_encoder", None) is not None:
            try:
                decoded_preds = preprocessor.target_encoder.inverse_transform(preds).tolist()
            except Exception:
                pass

        return {"predictions": decoded_preds, "probabilities": proba}


class CNNEngineAdapter(BaseModelEngine):
    """Unified engine adapter for Convolutional Neural Networks on spatial, image, and signal data."""

    def __init__(self, training_engine: Optional[ModelTrainingEngine] = None):
        self.training_engine = training_engine or ModelTrainingEngine()
        self.cnn_engine = CNNEngine()

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(
            model_family="cnn",
            model_name="Convolutional Neural Network (CNN)",
            supported_tasks=["image_classification", "binary_classification", "multiclass_classification", "spatial_grid"],
            supported_modalities=["image", "spatial", "signal", "spatial_grid"],
            minimum_samples=10,
            interpretability_level="black_box",
            training_cost="high",
            prediction_cost="medium",
            requirements=["2D/3D spatial tensor or pixel feature matrix"],
        )

    def supports(self, candidate_name: str, task_type: str, modality: str = "tabular") -> bool:
        if modality.lower() not in ("image", "spatial", "signal", "spatial_grid"):
            return False
        name_lower = candidate_name.lower()
        return "cnn" in name_lower or "convolutional" in name_lower

    def train_candidate(
        self,
        candidate: Union[str, ModelCandidate, Dict[str, Any]],
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        task_type: str,
        validation_strategy: str = "5_fold_cv",
        optimization_metric: str = "accuracy",
        random_state: int = 42,
    ) -> TrainingResult:
        return self.training_engine.train_and_validate_candidate(
            candidate=candidate,
            df=df,
            target_col=target_col,
            feature_cols=feature_cols,
            task_type=task_type,
            validation_strategy=validation_strategy,
            optimization_metric=optimization_metric,
            random_state=random_state,
        )

    def predict(
        self,
        model_obj: Any,
        preprocessor: Optional[Any],
        feature_columns: List[str],
        data: pd.DataFrame,
    ) -> Dict[str, Any]:
        df_feat = data[feature_columns].copy()
        X_t = preprocessor.transform(df_feat) if preprocessor else df_feat.to_numpy(dtype=float)
        preds = model_obj.predict(X_t)
        proba = model_obj.predict_proba(X_t).tolist() if hasattr(model_obj, "predict_proba") else None

        decoded_preds = preds.tolist()
        if preprocessor is not None and getattr(preprocessor, "target_encoder", None) is not None:
            try:
                decoded_preds = preprocessor.target_encoder.inverse_transform(preds).tolist()
            except Exception:
                pass

        return {"predictions": decoded_preds, "probabilities": proba}


# ==============================================================================
# 3. Unified Intelligent Model Orchestrator
# ==============================================================================

class UnifiedModelOrchestrator:
    """
    Unified Orchestrator coordinating model candidate validation, multi-family engine routing,
    parallel/sequential execution, deterministic metric ranking, winner selection, and registry persistence.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        max_parallel_models: int = 4,
        max_training_time_sec: float = 300.0,
    ):
        self.registry = registry or ModelRegistry()
        self.max_parallel_models = max_parallel_models
        self.max_training_time_sec = max_training_time_sec
        self.training_engine = ModelTrainingEngine(registry=self.registry)

        # Register standard model engines
        self.engines: Dict[str, BaseModelEngine] = {
            "traditional_ml": TraditionalMLEngine(self.training_engine),
            "ann": ANNEngineAdapter(self.training_engine),
            "cnn": CNNEngineAdapter(self.training_engine),
        }
        self.selection_agent = ModelSelectionAgent()

    def resolve_engine_for_candidate(
        self,
        candidate_name: str,
        task_type: str,
        modality: str = "tabular",
    ) -> Optional[BaseModelEngine]:
        """Match candidate specification to appropriate engine adapter based on capability."""
        cand_lower = candidate_name.lower()
        if "cnn" in cand_lower or "convolutional" in cand_lower:
            return self.engines["cnn"]
        elif "ann" in cand_lower or "mlp" in cand_lower or "neural" in cand_lower:
            return self.engines["ann"]
        else:
            return self.engines["traditional_ml"]

    def validate_candidate_compatibility(
        self,
        candidate_name: str,
        task_type: str,
        modality: str,
        n_samples: int,
    ) -> Tuple[bool, Optional[str]]:
        """Perform upfront validation checking modality, task type, and sample count compatibility."""
        cand_lower = candidate_name.lower()

        # CNN Guardrail: Reject CNN on non-spatial tabular data
        if ("cnn" in cand_lower or "convolutional" in cand_lower) and modality.lower() == "tabular":
            return False, "CNN requires 2D/3D image or spatial grid input; incompatible with 1D tabular features."

        # Sample count check
        if n_samples < 10:
            return False, f"Insufficient samples ({n_samples}) for reliable model training (minimum 10 required)."

        return True, None

    def orchestrate(
        self,
        request: TrainingRequest,
        dataframe: pd.DataFrame,
        data_modality: str = "tabular",
        parallel: bool = True,
    ) -> ModelComparisonResult:
        """
        End-to-end multi-model orchestration lifecycle:
        1. Validate inputs & candidate eligibility
        2. Execute candidates across engines (parallel/sequential)
        3. Collect TrainingResult objects & isolate failures
        4. Rank models deterministically by optimization metric
        5. Refit winning model & persist to ModelRegistry
        6. Compose explainable rationale
        """
        start_time = time.time()
        if dataframe is None or dataframe.empty:
            return ModelComparisonResult(
                status="failed",
                selection_reason="Cannot train models on empty dataset.",
                warnings=["Empty dataframe supplied to orchestrator."],
                confidence=0.0,
            )

        target_col = request.target_column
        if target_col not in dataframe.columns:
            return ModelComparisonResult(
                status="failed",
                selection_reason=f"Target column '{target_col}' not found in dataset.",
                warnings=[f"Target column '{target_col}' missing."],
                confidence=0.0,
            )

        feature_cols = request.feature_columns or [c for c in dataframe.columns if c != target_col]
        is_clf = "class" in request.task_type.lower() or "binary" in request.task_type.lower() or "multi" in request.task_type.lower()
        n_samples = len(dataframe)

        # 1. Candidate Pool Resolution
        candidates_raw = request.candidate_models
        if not candidates_raw:
            # Dynamically plan candidates if none provided
            plan = self.selection_agent.plan_model_selection(
                ModelSelectionRequest(target_column=target_col, data_modality=data_modality),
                dataframe=dataframe,
            )
            candidates_raw = [c.model_name for c in plan.candidates if c.suitability_score > 0.0]

        # 2. Capability Validation
        valid_candidates: List[Union[str, ModelCandidate, Dict[str, Any]]] = []
        rejected_warnings: List[str] = []

        for cand in candidates_raw:
            cand_name = cand.model_name if isinstance(cand, ModelCandidate) else (cand.get("model_name") if isinstance(cand, dict) else str(cand))
            is_valid, reject_reason = self.validate_candidate_compatibility(
                cand_name, request.task_type, data_modality, n_samples
            )
            if is_valid:
                valid_candidates.append(cand)
            else:
                rejected_warnings.append(f"Candidate '{cand_name}' rejected: {reject_reason}")

        if not valid_candidates:
            return ModelComparisonResult(
                status="failed",
                selection_reason="All candidate models were rejected during upfront capability validation.",
                warnings=rejected_warnings,
                confidence=0.0,
            )

        # 3. Execution (Parallel with Sequential Fallback)
        training_results: List[TrainingResult] = []

        def _train_single(candidate_spec: Any) -> TrainingResult:
            c_name = candidate_spec.model_name if isinstance(candidate_spec, ModelCandidate) else (candidate_spec.get("model_name") if isinstance(candidate_spec, dict) else str(candidate_spec))
            engine = self.resolve_engine_for_candidate(c_name, request.task_type, data_modality)
            return engine.train_candidate(
                candidate=candidate_spec,
                df=dataframe,
                target_col=target_col,
                feature_cols=feature_cols,
                task_type=request.task_type,
                validation_strategy=request.validation_strategy,
                optimization_metric=request.optimization_metric,
                random_state=request.random_state,
            )

        if parallel and len(valid_candidates) > 1 and self.max_parallel_models > 1:
            try:
                workers = min(len(valid_candidates), self.max_parallel_models)
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(_train_single, cand) for cand in valid_candidates]
                    for fut in concurrent.futures.as_completed(futures, timeout=self.max_training_time_sec):
                        training_results.append(fut.result())
            except Exception as exc:
                # Safe fallback to sequential execution on error or timeout
                rejected_warnings.append(f"Parallel execution encountered issue ({str(exc)}); falling back to sequential.")
                training_results = [_train_single(cand) for cand in valid_candidates]
        else:
            training_results = [_train_single(cand) for cand in valid_candidates]

        # 4. Result Separation & Failure Isolation
        successful = [r for r in training_results if r.status == "success"]
        failed = [r for r in training_results if r.status == "failed"]

        if not successful:
            return ModelComparisonResult(
                candidates=training_results,
                status="failed",
                selection_reason="All evaluated model candidates failed during cross-validation.",
                warnings=rejected_warnings + [r.error_message or "Unknown error" for r in failed],
                confidence=0.0,
            )

        # 5. Deterministic Metric Ranking
        opt_metric = request.optimization_metric.lower()
        lower_is_better = opt_metric in ("rmse", "mae", "mape", "loss")

        successful.sort(
            key=lambda r: r.validation_metrics.get(r.primary_metric_name, 0.0),
            reverse=not lower_is_better,
        )

        ranking: List[Dict[str, Any]] = []
        for rank_idx, r in enumerate(successful, start=1):
            ranking.append({
                "rank": rank_idx,
                "model_name": r.model_name,
                "model_family": r.model_family,
                "model_id": r.model_id,
                "primary_metric": r.primary_metric_name,
                "score": r.primary_metric_value,
                "training_time_ms": r.training_time_ms,
                "overfitting_detected": r.overfitting_detected,
            })

        best = successful[0]
        direction_str = "lowest" if lower_is_better else "highest"
        selection_reason = (
            f"'{best.model_name}' was selected because it achieved the {direction_str} validated "
            f"{best.primary_metric_name} ({best.primary_metric_value:.4f}) among {len(successful)} successfully "
            f"benchmarked algorithms. "
        )
        if failed:
            selection_reason += f"Note: {len(failed)} candidate(s) failed during evaluation. "
        if rejected_warnings:
            selection_reason += f"{len(rejected_warnings)} candidate(s) excluded due to data modality constraints."

        # 6. Fit Best Model on Full Dataset & Register as production_candidate
        try:
            full_preprocessor = DataPreprocessor()
            full_preprocessor.fit(dataframe[feature_cols], dataframe[target_col], is_classification=is_clf)
            X_full = full_preprocessor.transform(dataframe[feature_cols])
            y_full = full_preprocessor.transform_target(dataframe[target_col])

            full_estimator, _, _ = self.training_engine.instantiate_model(
                best.model_name, request.task_type, random_state=request.random_state
            )
            full_estimator.fit(X_full, y_full)

            meta = self.registry.register_model(
                name=best.model_name,
                model_object=full_estimator,
                model_family=best.model_family,
                algorithm=best.model_name,
                problem_type=request.task_type,
                target_column=target_col,
                feature_columns=feature_cols,
                training_metrics=best.training_metrics,
                validation_metrics=best.validation_metrics,
                primary_metric_name=best.primary_metric_name,
                primary_metric_value=best.primary_metric_value,
                preprocessor=full_preprocessor,
                tags=[request.task_type, request.validation_strategy, "production_candidate", f"modality_{data_modality}"],
            )
            best.model_artifact_path = str(self.registry.artifacts_dir / f"{meta.model_id}.joblib")
            best.model_id = meta.model_id
        except Exception as exc:
            best.warnings.append(f"Model artifact registration failed: {str(exc)}")

        # 7. Traceable Evidence
        evidence_list = [
            Evidence(
                source="UnifiedModelOrchestrator",
                method="multi_family_model_orchestration",
                data_ref={
                    "winner": best.model_name,
                    "model_family": best.model_family,
                    "target": target_col,
                    "optimization_metric": best.primary_metric_name,
                    "winning_score": best.primary_metric_value,
                    "total_candidates": len(valid_candidates),
                    "successful_count": len(successful),
                    "failed_count": len(failed),
                    "ranking": ranking,
                },
                confidence=best.confidence,
                claim_type=ClaimType.FACT,
            )
        ]

        overall_status = "success" if len(failed) == 0 and len(rejected_warnings) == 0 else "partial"
        all_warnings = rejected_warnings + [f"Candidate '{f.model_name}' failed: {f.error_message}" for f in failed]

        return ModelComparisonResult(
            candidates=training_results,
            ranking=ranking,
            best_model=best,
            optimization_metric=best.primary_metric_name,
            selection_reason=selection_reason.strip(),
            evidence=evidence_list,
            confidence=best.confidence,
            status=overall_status,
            warnings=all_warnings,
        )

    # ------------------------------------------------------------------
    # Unified Prediction Routing
    # ------------------------------------------------------------------
    def predict(
        self,
        model_id: str,
        new_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Unified prediction routing across any registered model family (Traditional ML, ANN, CNN).
        Enforces schema validation, feature order, preprocessor transformations, and probability extraction.
        """
        if isinstance(new_data, dict):
            df_new = pd.DataFrame([new_data])
        elif isinstance(new_data, list):
            df_new = pd.DataFrame(new_data)
        elif isinstance(new_data, pd.DataFrame):
            df_new = new_data.copy()
        else:
            raise ValueError(f"Unsupported data format for prediction: {type(new_data)}")

        model_obj, preprocessor, meta = self.registry.get_model(model_id)
        engine = self.engines.get(meta.model_family, self.engines["traditional_ml"])

        # Validate feature completeness
        missing_features = [f for f in meta.feature_columns if f not in df_new.columns]
        if missing_features:
            raise ValueError(f"Prediction input missing required features: {missing_features}")

        # Enforce exact training feature order
        df_ordered = df_new[meta.feature_columns].copy()
        pred_output = engine.predict(
            model_obj=model_obj,
            preprocessor=preprocessor,
            feature_columns=meta.feature_columns,
            data=df_ordered,
        )

        return {
            "model_id": model_id,
            "model_name": meta.name,
            "model_family": meta.model_family,
            "version": meta.version,
            "target_column": meta.target_column,
            "predictions": pred_output["predictions"],
            "probabilities": pred_output.get("probabilities"),
            "row_count": len(df_new),
            "status": "success",
        }


# ==============================================================================
# 4. Model Orchestrator Agent (BaseAgent Subclass)
# ==============================================================================

class ModelOrchestratorAgent(BaseAgent):
    """
    Autonomous Model Orchestrator Agent coordinating end-to-end model selection,
    multi-family engine dispatch, parallel validation, and registry tracking.
    """
    name = "Model Orchestrator Agent"
    role = "lead_ml_architect"
    description = "Intelligently selects, routes, validates, benchmarks, and registers winning ML/DL models across algorithms."

    def __init__(self, data: Optional[Any] = None, registry: Optional[ModelRegistry] = None):
        super().__init__(data=data)
        self.orchestrator = UnifiedModelOrchestrator(registry=registry)

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """
        Execute model orchestration lifecycle.
        Task parameters:
            - data: pd.DataFrame or dict
            - target: str (target column name)
            - features: list[str] (optional feature column names)
            - task_type: str (regression, binary_classification, etc.)
            - modality: str (tabular, image, spatial, signal)
            - candidates: list[str | ModelCandidate] (optional candidate pool)
            - metric: str (optimization metric)
            - parallel: bool (default: True)
        """
        self._start()
        data = task.get("data") if task.get("data") is not None else self.data

        if isinstance(data, dict):
            for df in data.values():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df_target = df
                    break
            else:
                return self._error("No valid pandas DataFrame found in task data.", category=ErrorCategory.INPUT_VALIDATION)
        elif isinstance(data, pd.DataFrame):
            df_target = data
        else:
            return self._error("Missing required DataFrame input for model orchestration.", category=ErrorCategory.INPUT_VALIDATION)

        target = task.get("target")
        if not target or target not in df_target.columns:
            num_cols = df_target.select_dtypes(include=["number"]).columns
            target = num_cols[-1] if len(num_cols) > 0 else df_target.columns[-1]

        features = task.get("features")
        task_type = task.get("task_type", "regression")
        modality = task.get("modality", "tabular")
        candidates = task.get("candidates")
        metric = task.get("metric", "r2" if "reg" in task_type else "accuracy")
        parallel = bool(task.get("parallel", True))

        request = TrainingRequest(
            target_column=target,
            feature_columns=features,
            task_type=task_type,
            candidate_models=candidates or [],
            optimization_metric=metric,
        )

        try:
            comparison_result = self.orchestrator.orchestrate(
                request=request,
                dataframe=df_target,
                data_modality=modality,
                parallel=parallel,
            )
        except Exception as exc:
            return self._error(f"Model orchestration failed: {str(exc)}", category=ErrorCategory.COMPUTATION)

        if comparison_result.status == "failed":
            return self._error(
                comparison_result.selection_reason or "All candidate models failed during orchestration.",
                category=ErrorCategory.COMPUTATION,
                details={"warnings": comparison_result.warnings},
            )

        output_data = comparison_result.to_dict()
        if comparison_result.status == "partial":
            return self._partial(
                result=output_data,
                message=comparison_result.selection_reason,
                warnings=comparison_result.warnings,
                evidence=comparison_result.evidence,
                confidence=comparison_result.confidence,
            )

        return self._finish(
            result=output_data,
            evidence=comparison_result.evidence,
            confidence=comparison_result.confidence,
            metadata={"winner": comparison_result.best_model.model_name if comparison_result.best_model else "none"},
        )
