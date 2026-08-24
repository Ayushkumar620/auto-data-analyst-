"""Artificial Neural Network (ANN) Agent - Orchestrates Tabular Deep Learning."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

from agent.base import BaseAgent
from agent.schemas import AgentResult, ClaimType, ErrorCategory, Evidence
from backend.app.ml.ann_engine import ANNEngine, ANNHyperparameters, ANNTrainingResult


class ANNAgent(BaseAgent):
    """
    Autonomous Artificial Neural Network (ANN / MLP) Agent.
    Constructs, trains, tunes, and evaluates deep neural network architectures on tabular datasets.
    Tracks epoch loss curves, checks early stopping, and benchmarks directly against traditional ML models.
    """
    name = "ANN Agent"
    role = "deep_learning_engineer"
    description = "Trains modular Artificial Neural Networks (ANN/MLP) with hyperparameter tuning and ML comparison."

    def __init__(self, data=None):
        super().__init__(data=data)
        self.engine = ANNEngine()

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """
        Execute ANN model training / tuning.
        Task parameters:
            - data: pd.DataFrame or dict (optional, defaults to self.data)
            - target: str (target column name, optional)
            - features: list[str] (feature column names, optional)
            - layers: list[int] or tuple[int] (hidden layer sizes, e.g. [128, 64])
            - activation: str ('relu', 'tanh', 'logistic')
            - epochs: int (max epochs / iterations)
            - learning_rate: float
            - tune: bool (if True, runs multi-architecture search)
            - compare_with_ml: bool (default: True)
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
            return self._error("Missing required DataFrame input.", category=ErrorCategory.INPUT_VALIDATION)

        if len(df_target) < 10:
            return self._error(f"Dataset requires at least 10 samples for ANN training. Found {len(df_target)}.", category=ErrorCategory.DATA_QUALITY)

        # Detect target
        target = task.get("target")
        if not target or target not in df_target.columns:
            num_cols = df_target.select_dtypes(include=["number"]).columns
            if len(num_cols) > 0:
                target = num_cols[-1]
            else:
                target = df_target.columns[-1]

        features = task.get("features")
        should_tune = bool(task.get("tune", False))
        compare_ml = bool(task.get("compare_with_ml", True))

        # Hyperparameter setup
        layers = tuple(task.get("layers", (128, 64)))
        activation = task.get("activation", "relu")
        epochs = int(task.get("epochs", 200))
        lr = float(task.get("learning_rate", 0.001))

        hyperparams = ANNHyperparameters(
            hidden_layer_sizes=layers,
            activation=activation,
            max_iter=epochs,
            learning_rate_init=lr,
        )

        try:
            if should_tune:
                result_obj, tuning_trials = self.engine.tune_architecture(
                    dataframe=df_target,
                    target_column=target,
                    feature_columns=features,
                )
            else:
                result_obj = self.engine.train_and_evaluate(
                    dataframe=df_target,
                    target_column=target,
                    feature_columns=features,
                    hyperparams=hyperparams,
                    compare_with_ml=compare_ml,
                )
                tuning_trials = []
        except Exception as exc:
            return self._error(f"ANN training failed: {str(exc)}", category=ErrorCategory.COMPUTATION)

        # Build traceable Evidence
        evidence_list: List[Evidence] = []

        # 1. Primary ANN metric evidence
        evidence_list.append(
            self.make_evidence(
                method="ann_training_and_evaluation",
                data_ref={
                    "target": target,
                    "model": result_obj.model_name,
                    "architecture": result_obj.architecture_summary,
                    "epochs_trained": result_obj.epochs_trained,
                    "primary_metric": result_obj.primary_metric_name,
                    "metric_value": result_obj.primary_metric_value,
                },
                confidence=0.95,
                claim_type=ClaimType.FACT,
            )
        )

        # 2. Comparison with traditional ML evidence
        if result_obj.comparison_with_ml and "best_traditional_ml_model" in result_obj.comparison_with_ml:
            comp = result_obj.comparison_with_ml
            evidence_list.append(
                self.make_evidence(
                    method="ann_vs_traditional_ml_comparison",
                    data_ref={
                        "target": target,
                        "best_ml_model": comp.get("best_traditional_ml_model"),
                        "ml_score": comp.get("traditional_ml_score"),
                        "ann_score": comp.get("ann_score"),
                        "ann_outperformed_ml": comp.get("ann_outperformed_ml"),
                    },
                    confidence=0.90,
                    claim_type=ClaimType.OBSERVATION,
                )
            )

        output = {
            "problem_type": result_obj.problem_type.value,
            "target": target,
            "architecture_summary": result_obj.architecture_summary,
            "hyperparameters": result_obj.hyperparameters,
            "metrics": result_obj.metrics,
            "primary_metric_name": result_obj.primary_metric_name,
            "primary_metric_value": result_obj.primary_metric_value,
            "epochs_trained": result_obj.epochs_trained,
            "stopped_early": result_obj.stopped_early,
            "loss_curve": result_obj.loss_curve,
            "comparison_with_ml": result_obj.comparison_with_ml,
            "tuning_trials": tuning_trials,
        }

        return self._finish(
            result=output,
            evidence=evidence_list,
            confidence=0.95,
            metadata={"architecture": result_obj.architecture_summary, "target": target},
        )
