"""Modular Artificial Neural Network (ANN) Engine for Tabular Deep Learning.

Supports:
- Multi-Layer Perceptron (MLP) architectures for Classification and Regression
- Flexible Hyperparameter configurations:
  - Hidden layer depths and neuron topologies (e.g. (128, 64), (256, 128, 64))
  - Activations: ReLU, Tanh, Logistic, Identity
  - Optimizers: Adam, SGD, L-BFGS
  - Learning rates, batch sizes, epochs (max_iter)
  - Regularization (alpha / L2 penalty)
- Loss curve tracking and convergence diagnostics
- Early stopping with validation tracking
- Automatic architecture search / tuning
- Automated side-by-side comparison with traditional ML baselines
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from backend.app.ml.model_selection import (
    MLModelComparisonEngine,
    ProblemType,
)


@dataclass
class ANNHyperparameters:
    """Hyperparameter specification for the Artificial Neural Network."""
    hidden_layer_sizes: Tuple[int, ...] = (128, 64)
    activation: str = "relu"  # 'relu', 'tanh', 'logistic', 'identity'
    solver: str = "adam"  # 'adam', 'sgd', 'lbfgs'
    alpha: float = 0.0001  # L2 regularization parameter
    learning_rate_init: float = 0.001
    learning_rate: str = "adaptive"  # 'constant', 'invscaling', 'adaptive'
    max_iter: int = 200  # Number of epochs
    batch_size: Union[int, str] = "auto"
    early_stopping: bool = True
    n_iter_no_change: int = 10
    validation_fraction: float = 0.15
    random_state: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "activation": self.activation,
            "solver": self.solver,
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
            "early_stopping": self.early_stopping,
            "n_iter_no_change": self.n_iter_no_change,
            "validation_fraction": self.validation_fraction,
        }


@dataclass
class ANNTrainingResult:
    """Evaluation, loss history, and ML comparison result for an ANN run."""
    problem_type: ProblemType
    model_name: str
    architecture_summary: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    primary_metric_name: str
    primary_metric_value: float
    loss_curve: List[float] = field(default_factory=list)
    validation_scores: List[float] = field(default_factory=list)
    epochs_trained: int = 0
    stopped_early: bool = False
    training_duration_ms: float = 0.0
    comparison_with_ml: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    target_name: str = ""
    status: str = "success"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_type": self.problem_type.value,
            "model_name": self.model_name,
            "architecture_summary": self.architecture_summary,
            "target_name": self.target_name,
            "feature_names": self.feature_names,
            "hyperparameters": self.hyperparameters,
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_value": round(float(self.primary_metric_value), 4),
            "metrics": {k: round(float(v), 4) for k, v in self.metrics.items()},
            "loss_curve": [round(float(l), 5) for l in self.loss_curve],
            "validation_scores": [round(float(s), 5) for s in self.validation_scores],
            "epochs_trained": self.epochs_trained,
            "stopped_early": self.stopped_early,
            "training_duration_ms": round(float(self.training_duration_ms), 2),
            "comparison_with_ml": self.comparison_with_ml,
            "status": self.status,
            "error_message": self.error_message,
        }


class ANNEngine:
    """Modular Artificial Neural Network Engine for Tabular Datasets."""

    def __init__(self):
        self.ml_engine = MLModelComparisonEngine()

    # ------------------------------------------------------------------
    # Data Preprocessing for Neural Networks
    # ------------------------------------------------------------------
    def prepare_data(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], ProblemType]:
        """Normalize and scale features specifically for gradient-based neural network training."""
        df = dataframe.copy()
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe.")

        if feature_columns is None:
            features = [c for c in df.columns if c != target_column]
        else:
            features = [c for c in feature_columns if c in df.columns and c != target_column]

        if not features:
            raise ValueError("No feature columns available for ANN training.")

        X_df = df[features].copy()
        y_series = df[target_column].copy()

        # Handle datetime columns in features
        for col in X_df.columns:
            if pd.api.types.is_datetime64_any_dtype(X_df[col]):
                X_df[col] = pd.to_datetime(X_df[col]).astype("int64") // 10**9

        # Impute and encode categorical features
        for col in X_df.select_dtypes(include=["object", "string", "category"]).columns:
            X_df[col] = X_df[col].fillna("Missing").astype(str)
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col])

        # Impute numeric features with median
        for col in X_df.select_dtypes(include=[np.number]).columns:
            median_val = X_df[col].median()
            X_df[col] = X_df[col].fillna(median_val if not np.isnan(median_val) else 0.0)

        # Detect problem type
        problem_type = self.ml_engine.detect_problem_type(y_series)

        if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            y_clean = y_series.fillna(y_series.mode()[0] if not y_series.mode().empty else 0)
            le = LabelEncoder()
            y_arr = le.fit_transform(y_clean.astype(str))
        else:
            y_num = pd.to_numeric(y_series, errors="coerce")
            median_y = y_num.median()
            y_arr = y_num.fillna(median_y if not np.isnan(median_y) else 0.0).to_numpy(dtype=float)

        # Standard scaling (zero mean, unit variance) is essential for ANN convergence
        scaler = StandardScaler()
        X_arr = scaler.fit_transform(X_df.to_numpy(dtype=float))

        return X_arr, y_arr, features, problem_type

    # ------------------------------------------------------------------
    # ANN Training & Evaluation
    # ------------------------------------------------------------------
    def train_and_evaluate(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        hyperparams: Optional[ANNHyperparameters] = None,
        compare_with_ml: bool = True,
        test_size: float = 0.2,
    ) -> ANNTrainingResult:
        """Train an Artificial Neural Network, extract training curves, and benchmark against ML."""
        params = hyperparams or ANNHyperparameters()
        X, y, features, problem_type = self.prepare_data(dataframe, target_column, feature_columns)

        if len(X) < 10:
            raise ValueError(f"Need at least 10 samples to train an ANN. Found {len(X)}.")

        # Train/Test Split
        is_classification = problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION)
        strat = y if is_classification and min(np.bincount(y)) >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=params.random_state, stratify=strat
        )

        # Instantiate MLP Model
        start_time = time.time()
        if is_classification:
            model = MLPClassifier(
                hidden_layer_sizes=params.hidden_layer_sizes,
                activation=params.activation,
                solver=params.solver,
                alpha=params.alpha,
                learning_rate_init=params.learning_rate_init,
                learning_rate=params.learning_rate,
                max_iter=params.max_iter,
                batch_size=params.batch_size,
                early_stopping=params.early_stopping,
                n_iter_no_change=params.n_iter_no_change,
                validation_fraction=params.validation_fraction,
                random_state=params.random_state,
            )
        else:
            model = MLPRegressor(
                hidden_layer_sizes=params.hidden_layer_sizes,
                activation=params.activation,
                solver=params.solver,
                alpha=params.alpha,
                learning_rate_init=params.learning_rate_init,
                learning_rate=params.learning_rate,
                max_iter=params.max_iter,
                batch_size=params.batch_size,
                early_stopping=params.early_stopping,
                n_iter_no_change=params.n_iter_no_change,
                validation_fraction=params.validation_fraction,
                random_state=params.random_state,
            )

        model.fit(X_train, y_train)
        duration_ms = (time.time() - start_time) * 1000
        test_preds = model.predict(X_test)

        # Extract loss curve and validation curve
        loss_curve = list(getattr(model, "loss_curve_", []))
        val_scores = list(getattr(model, "validation_scores_", []))
        epochs_trained = int(getattr(model, "n_iter_", len(loss_curve)))
        stopped_early = bool(epochs_trained < params.max_iter)

        # Compute Metrics
        metrics: Dict[str, float] = {}
        if is_classification:
            test_acc = float(accuracy_score(y_test, test_preds))
            test_f1 = float(f1_score(y_test, test_preds, average="weighted", zero_division=0))
            test_prec = float(precision_score(y_test, test_preds, average="weighted", zero_division=0))
            test_rec = float(recall_score(y_test, test_preds, average="weighted", zero_division=0))

            metrics["accuracy"] = test_acc
            metrics["f1_score"] = test_f1
            metrics["precision"] = test_prec
            metrics["recall"] = test_rec

            primary_name = "accuracy"
            primary_val = test_acc
        else:
            test_r2 = float(r2_score(y_test, test_preds))
            test_mse = float(mean_squared_error(y_test, test_preds))
            test_rmse = float(np.sqrt(test_mse))
            test_mae = float(mean_absolute_error(y_test, test_preds))

            metrics["r2_score"] = test_r2
            metrics["rmse"] = test_rmse
            metrics["mae"] = test_mae
            metrics["mse"] = test_mse

            primary_name = "r2_score"
            primary_val = test_r2

        # Architecture summary string
        arch_layers = [f"Input({len(features)})"]
        for size in params.hidden_layer_sizes:
            arch_layers.append(f"Dense({size}, {params.activation})")
        out_dim = len(np.unique(y)) if is_classification and len(np.unique(y)) > 2 else 1
        arch_layers.append(f"Output({out_dim})")
        arch_summary = " -> ".join(arch_layers)

        # Benchmark comparison with traditional ML models
        comparison: Dict[str, Any] = {}
        if compare_with_ml:
            try:
                ml_report = self.ml_engine.benchmark_models(
                    dataframe=dataframe,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    cv_folds=3,
                )
                best_ml = ml_report.best_model
                comparison = {
                    "best_traditional_ml_model": best_ml.model_name,
                    "traditional_ml_score": round(float(best_ml.primary_metric_value), 4),
                    "ann_score": round(float(primary_val), 4),
                    "ann_delta": round(float(primary_val - best_ml.primary_metric_value), 4),
                    "ann_outperformed_ml": bool(primary_val >= best_ml.primary_metric_value),
                    "traditional_ml_leaderboard": ml_report.leaderboard[:3],
                }
            except Exception as exc:
                comparison = {"comparison_error": str(exc)}

        return ANNTrainingResult(
            problem_type=problem_type,
            model_name="Artificial Neural Network (ANN / MLP)",
            architecture_summary=arch_summary,
            hyperparameters=params.to_dict(),
            metrics=metrics,
            primary_metric_name=primary_name,
            primary_metric_value=primary_val,
            loss_curve=loss_curve,
            validation_scores=val_scores,
            epochs_trained=epochs_trained,
            stopped_early=stopped_early,
            training_duration_ms=duration_ms,
            comparison_with_ml=comparison,
            feature_names=features,
            target_name=target_column,
            status="success",
        )

    # ------------------------------------------------------------------
    # Automatic Architecture Tuning
    # ------------------------------------------------------------------
    def tune_architecture(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[ANNTrainingResult, List[Dict[str, Any]]]:
        """Evaluate multiple candidate ANN architectures (Shallow, Medium, Deep, Wide) and pick the best."""
        candidate_configs = [
            ANNHyperparameters(hidden_layer_sizes=(64,), activation="relu", max_iter=150),
            ANNHyperparameters(hidden_layer_sizes=(128, 64), activation="relu", max_iter=200),
            ANNHyperparameters(hidden_layer_sizes=(128, 64, 32), activation="relu", max_iter=250),
            ANNHyperparameters(hidden_layer_sizes=(128, 64), activation="tanh", max_iter=200),
        ]

        trials: List[Dict[str, Any]] = []
        best_result: Optional[ANNTrainingResult] = None
        best_score = -999.0

        for idx, config in enumerate(candidate_configs, start=1):
            try:
                res = self.train_and_evaluate(
                    dataframe=dataframe,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    hyperparams=config,
                    compare_with_ml=False,
                )
                score = res.primary_metric_value
                trials.append({
                    "trial": idx,
                    "architecture": res.architecture_summary,
                    "activation": config.activation,
                    "score": round(float(score), 4),
                    "epochs_trained": res.epochs_trained,
                    "status": "success",
                })
                if score > best_score or best_result is None:
                    best_score = score
                    best_result = res
            except Exception as exc:
                trials.append({
                    "trial": idx,
                    "architecture": str(config.hidden_layer_sizes),
                    "status": "failed",
                    "error": str(exc),
                })

        if best_result is None:
            raise RuntimeError("All ANN architecture configurations failed.")

        # Re-run best configuration with full ML comparison
        best_full = self.train_and_evaluate(
            dataframe=dataframe,
            target_column=target_column,
            feature_columns=feature_columns,
            hyperparams=ANNHyperparameters(
                hidden_layer_sizes=tuple(best_result.hyperparameters["hidden_layer_sizes"]),
                activation=best_result.hyperparameters["activation"],
            ),
            compare_with_ml=True,
        )

        return best_full, trials
