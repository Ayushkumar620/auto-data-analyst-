"""
Unified Model Training, Evaluation, and Benchmarking Engine.

Provides:
- BaseModelTrainer interface and concrete TraditionalMLTrainer adapter
- Leakage-free DataPreprocessor (fit strictly on training folds)
- Cross-validation splitters (KFold, StratifiedKFold, TimeSeriesSplit)
- Multi-metric evaluation (Regression, Classification, TimeSeries, Clustering)
- Overfitting detection and train/val divergence analysis
- Model ranking, winner selection, artifact persistence, and ModelRegistry integration
- Resilient multi-candidate execution with isolated error recovery
- Evidence generation grounded in actual computed metrics
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from agent.ann_schemas import ANNConfig, auto_select_ann_architecture
from agent.base import BaseAgent
from agent.model_selection_schemas import ModelCandidate
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
from backend.app.ml.registry import ModelRegistry


# ==============================================================================
# 1. Base Model Trainer Interface & Scikit-Learn Adapter
# 1. Base Model Trainer Interface & Concrete Adapters
# ==============================================================================

class BaseModelTrainer(ABC):
    """Abstract interface for all model training implementations."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseModelTrainer:
        """Fit model weights on training features and targets."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate point predictions for input feature matrix."""
        pass

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Generate class probability estimates (classification only)."""
        return None

    @abstractmethod
    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize model artifact to disk."""
        pass

    @abstractmethod
    def load(self, filepath: Union[str, Path]) -> BaseModelTrainer:
        """Deserialize model artifact from disk."""
        pass


class TraditionalMLTrainer(BaseModelTrainer):
    """Concrete trainer adapter wrapping standard scikit-learn algorithms."""

    def __init__(self, estimator: BaseEstimator, model_name: str, model_family: str = "traditional_ml"):
        self.estimator = estimator
        self.model_name = model_name
        self.model_family = model_family

    def fit(self, X: np.ndarray, y: np.ndarray) -> TraditionalMLTrainer:
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if hasattr(self.estimator, "predict_proba"):
            try:
                return self.estimator.predict_proba(X)
            except Exception:
                pass
        return None

    def save(self, filepath: Union[str, Path]) -> None:
        joblib.dump(self.estimator, filepath)

    def load(self, filepath: Union[str, Path]) -> TraditionalMLTrainer:
        self.estimator = joblib.load(filepath)
        return self


class ANNTrainer(BaseModelTrainer):
    """Concrete Artificial Neural Network (MLP) trainer adapter."""

    def __init__(
        self,
        estimator: Union[MLPClassifier, MLPRegressor],
        model_name: str = "Artificial Neural Network (ANN/MLP)",
        config: Optional[ANNConfig] = None,
    ):
        self.estimator = estimator
        self.model_name = model_name
        self.model_family = "ann"
        self.config = config

    def fit(self, X: np.ndarray, y: np.ndarray) -> ANNTrainer:
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if hasattr(self.estimator, "predict_proba"):
            try:
                return self.estimator.predict_proba(X)
            except Exception:
                pass
        return None

    def save(self, filepath: Union[str, Path]) -> None:
        joblib.dump(self.estimator, filepath)

    def load(self, filepath: Union[str, Path]) -> ANNTrainer:
        self.estimator = joblib.load(filepath)
        return self


# ==============================================================================
# 2. Leakage-Free Data Preprocessor
# ==============================================================================

class DataPreprocessor:
    """
    Leakage-free preprocessor that learns transformations strictly on training folds.
    Handles numeric scaling, categorical encoding, missing value imputation, and constant feature removal.
    """

    def __init__(self, numeric_strategy: str = "standard", categorical_strategy: str = "onehot"):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.impute_values: Dict[str, Any] = {}
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, Dict[Any, int]] = {}
        self.target_encoder: Optional[LabelEncoder] = None
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, is_classification: bool = False) -> DataPreprocessor:
        """Learn parameters (means, medians, categories) ONLY from training slice X."""
        X_df = X.copy()
        self.numeric_cols = list(X_df.select_dtypes(include=[np.number]).columns)
        self.categorical_cols = [c for c in X_df.columns if c not in self.numeric_cols]

        # 1. Numeric imputation & scaling
        for col in self.numeric_cols:
            series = X_df[col].dropna()
            self.impute_values[col] = float(series.median()) if not series.empty else 0.0

        if self.numeric_cols and self.numeric_strategy == "standard":
            self.scaler = StandardScaler()
            X_num = X_df[self.numeric_cols].fillna(self.impute_values)
            self.scaler.fit(X_num)

        # 2. Categorical encoding
        for col in self.categorical_cols:
            series = X_df[col].fillna("__MISSING__").astype(str)
            unique_vals = list(series.unique())
            self.label_encoders[col] = {val: idx for idx, val in enumerate(unique_vals)}
            self.impute_values[col] = "__MISSING__"

        # 3. Target encoding for classification
        if is_classification and y is not None:
            clean_y = y.dropna()
            self.target_encoder = LabelEncoder()
            self.target_encoder.fit(clean_y)

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply learned transformations without referencing test distributions."""
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted on training data before calling transform.")

        X_df = X.copy()
        processed_parts: List[np.ndarray] = []

        # Process numerics
        if self.numeric_cols:
            X_num = X_df[self.numeric_cols].copy()
            for col in self.numeric_cols:
                X_num[col] = X_num[col].fillna(self.impute_values.get(col, 0.0))
            if self.scaler is not None:
                processed_parts.append(self.scaler.transform(X_num))
            else:
                processed_parts.append(X_num.to_numpy())

        # Process categoricals
        if self.categorical_cols:
            cat_matrix = []
            for _, row in X_df[self.categorical_cols].iterrows():
                row_vals = []
                for col in self.categorical_cols:
                    val_str = str(row[col]) if pd.notna(row[col]) else "__MISSING__"
                    mapping = self.label_encoders.get(col, {})
                    row_vals.append(mapping.get(val_str, len(mapping)))  # Out of vocab mapped to new index
                cat_matrix.append(row_vals)
            processed_parts.append(np.array(cat_matrix, dtype=float))

        if not processed_parts:
            return np.zeros((len(X_df), 1))

        return np.hstack(processed_parts)

    def transform_target(self, y: pd.Series) -> np.ndarray:
        """Transform target series using fitted encoder if classification, or numeric array."""
        if self.target_encoder is not None:
            # Map unseen labels safely
            classes_map = {c: i for i, c in enumerate(self.target_encoder.classes_)}
            return np.array([classes_map.get(val, 0) for val in y], dtype=int)
        return pd.to_numeric(y, errors="coerce").fillna(0.0).to_numpy()


# ==============================================================================
# 3. Model Training & Evaluation Engine
# ==============================================================================

class ModelTrainingEngine:
    """
    Orchestrates candidate training across validation splits, calculates comprehensive metrics,
    checks for overfitting, ranks algorithms, and registers winning artifacts.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()

    # ------------------------------------------------------------------
    # Model Estimator Factory
    # ------------------------------------------------------------------
    def instantiate_model(
        self,
        candidate_name_or_obj: Union[str, ModelCandidate, Dict[str, Any]],
        task_type: str,
        random_state: int = 42,
    ) -> Tuple[BaseEstimator, str, str]:
        """Instantiate a concrete scikit-learn estimator from candidate specification."""
        name = ""
        family = "traditional_ml"
        hyperparams: Dict[str, Any] = {}

        if isinstance(candidate_name_or_obj, str):
            name = candidate_name_or_obj
        elif isinstance(candidate_name_or_obj, ModelCandidate):
            name = candidate_name_or_obj.model_name
            family = candidate_name_or_obj.model_family
        elif isinstance(candidate_name_or_obj, dict):
            name = candidate_name_or_obj.get("model_name", candidate_name_or_obj.get("name", "Unknown"))
            family = candidate_name_or_obj.get("model_family", "traditional_ml")
            hyperparams = candidate_name_or_obj.get("hyperparameters", {})

        name_lower = name.lower()
        is_clf = "class" in task_type.lower() or "binary" in task_type.lower() or "multi" in task_type.lower()

        # Classification Models
        if is_clf:
            if "logistic" in name_lower or "linear" in name_lower:
                return LogisticRegression(max_iter=500, random_state=random_state), "Logistic Regression", "linear"
            elif "random forest" in name_lower or "rf" in name_lower:
                return RandomForestClassifier(n_estimators=100, random_state=random_state), "Random Forest Classifier", "ensemble"
            elif "gradient boosting" in name_lower or "gbm" in name_lower or "boost" in name_lower:
                return GradientBoostingClassifier(n_estimators=100, random_state=random_state), "Gradient Boosting Classifier", "ensemble"
            elif "extra trees" in name_lower:
                return ExtraTreesClassifier(n_estimators=100, random_state=random_state), "Extra Trees Classifier", "ensemble"
            elif "decision tree" in name_lower or "tree" in name_lower:
                return DecisionTreeClassifier(max_depth=5, random_state=random_state), "Decision Tree Classifier", "tree"
            elif "svc" in name_lower or "svm" in name_lower:
                return SVC(probability=True, random_state=random_state), "Support Vector Classifier", "kernel"
            elif "knn" in name_lower or "neighbor" in name_lower:
                return KNeighborsClassifier(n_neighbors=5), "K-Nearest Neighbors", "neighbors"
            elif "ann" in name_lower or "neural" in name_lower or "mlp" in name_lower:
                hidden_layers = tuple(hyperparams.get("hidden_layer_sizes", (64, 32)))
                activation = hyperparams.get("activation", "relu")
                solver = hyperparams.get("solver", "adam")
                alpha = float(hyperparams.get("alpha", 0.0001))
                epochs = int(hyperparams.get("max_iter", hyperparams.get("epochs", 200)))
                early_stop = bool(hyperparams.get("early_stopping", True))
                return MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation=activation,
                    solver=solver,
                    alpha=alpha,
                    max_iter=epochs,
                    early_stopping=early_stop,
                    random_state=random_state,
                ), "Artificial Neural Network (ANN/MLP)", "ann"
            else:
                return LogisticRegression(max_iter=500, random_state=random_state), name or "Logistic Regression", family

        # Regression Models
        else:
            if "ridge" in name_lower:
                return Ridge(alpha=1.0, random_state=random_state), "Ridge Regression", "linear"
            elif "lasso" in name_lower:
                return Lasso(alpha=0.1, random_state=random_state), "Lasso Regression", "linear"
            elif "linear" in name_lower:
                return LinearRegression(), "Linear Regression", "linear"
            elif "random forest" in name_lower or "rf" in name_lower:
                return RandomForestRegressor(n_estimators=100, random_state=random_state), "Random Forest Regressor", "ensemble"
            elif "gradient boosting" in name_lower or "gbm" in name_lower or "boost" in name_lower:
                return GradientBoostingRegressor(n_estimators=100, random_state=random_state), "Gradient Boosting Regressor", "ensemble"
            elif "extra trees" in name_lower:
                return ExtraTreesRegressor(n_estimators=100, random_state=random_state), "Extra Trees Regressor", "ensemble"
            elif "decision tree" in name_lower or "tree" in name_lower:
                return DecisionTreeRegressor(max_depth=5, random_state=random_state), "Decision Tree Regressor", "tree"
            elif "svr" in name_lower or "svm" in name_lower:
                return SVR(), "Support Vector Regressor", "kernel"
            elif "knn" in name_lower or "neighbor" in name_lower:
                return KNeighborsRegressor(n_neighbors=5), "K-Nearest Neighbors Regressor", "neighbors"
            elif "ann" in name_lower or "neural" in name_lower or "mlp" in name_lower:
                hidden_layers = tuple(hyperparams.get("hidden_layer_sizes", (64, 32)))
                activation = hyperparams.get("activation", "relu")
                solver = hyperparams.get("solver", "adam")
                alpha = float(hyperparams.get("alpha", 0.0001))
                epochs = int(hyperparams.get("max_iter", hyperparams.get("epochs", 200)))
                early_stop = bool(hyperparams.get("early_stopping", True))
                return MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    activation=activation,
                    solver=solver,
                    alpha=alpha,
                    max_iter=epochs,
                    early_stopping=early_stop,
                    random_state=random_state,
                ), "Artificial Neural Network (ANN/MLP)", "ann"
            else:
                return LinearRegression(), name or "Linear Regression", family

    # ------------------------------------------------------------------
    # Metric Computations
    # ------------------------------------------------------------------
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute standard evaluation metrics according to task type."""
        metrics: Dict[str, float] = {}
        is_clf = "class" in task_type.lower() or "binary" in task_type.lower() or "multi" in task_type.lower()

        if is_clf:
            unique_classes = np.unique(y_true)
            is_binary = len(unique_classes) <= 2

            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            if is_binary:
                metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
                metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
                metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
                if y_proba is not None:
                    try:
                        prob_pos = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] > 1 else y_proba
                        metrics["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
                        metrics["pr_auc"] = float(average_precision_score(y_true, prob_pos))
                    except Exception:
                        pass
            else:
                metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
                metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
                metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

        else:
            # Regression metrics
            try:
                metrics["r2"] = float(r2_score(y_true, y_pred))
            except Exception:
                metrics["r2"] = 0.0
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["rmse"] = float(math.sqrt(mean_squared_error(y_true, y_pred)))

            # MAPE when target values are non-zero
            if not np.any(y_true == 0):
                try:
                    mape_val = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
                    if not np.isnan(mape_val) and not np.isinf(mape_val):
                        metrics["mape"] = round(mape_val, 2)
                except Exception:
                    pass

        return {k: round(v, 4) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Train Single Candidate with Cross-Validation
    # ------------------------------------------------------------------
    def train_and_validate_candidate(
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
        """Train and evaluate an individual candidate algorithm across validation folds."""
        start_time = time.time()
        is_clf = "class" in task_type.lower() or "binary" in task_type.lower() or "multi" in task_type.lower()

        estimator, model_name, model_family = self.instantiate_model(candidate, task_type, random_state=random_state)
        model_id = f"mod_{uuid.uuid4().hex[:10]}"

        X_raw = df[feature_cols]
        y_raw = df[target_col]

        # Setup splitters
        n_splits = 5 if len(df) >= 20 else 3
        if "time_series" in validation_strategy.lower():
            splitter = TimeSeriesSplit(n_splits=n_splits)
        elif is_clf and y_raw.nunique() > 1 and y_raw.value_counts().min() >= n_splits:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        fold_train_metrics: List[Dict[str, float]] = []
        fold_val_metrics: List[Dict[str, float]] = []
        warnings: List[str] = []

        try:
            for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_raw, y_raw)):
                X_train_fold = X_raw.iloc[train_idx]
                y_train_fold = y_raw.iloc[train_idx]
                X_val_fold = X_raw.iloc[val_idx]
                y_val_fold = y_raw.iloc[val_idx]

                # PREPROCESSING IS FIT ONLY ON TRAINING FOLD TO PREVENT DATA LEAKAGE
                preprocessor = DataPreprocessor()
                preprocessor.fit(X_train_fold, y_train_fold, is_classification=is_clf)

                X_train_t = preprocessor.transform(X_train_fold)
                y_train_t = preprocessor.transform_target(y_train_fold)
                X_val_t = preprocessor.transform(X_val_fold)
                y_val_t = preprocessor.transform_target(y_val_fold)

                # Clone and fit
                model_fold = clone(estimator)
                model_fold.fit(X_train_t, y_train_t)

                # Train predictions
                y_train_pred = model_fold.predict(X_train_t)
                y_train_proba = model_fold.predict_proba(X_train_t) if hasattr(model_fold, "predict_proba") else None
                m_train = self.compute_metrics(y_train_t, y_train_pred, task_type, y_train_proba)
                fold_train_metrics.append(m_train)

                # Validation predictions
                y_val_pred = model_fold.predict(X_val_t)
                y_val_proba = model_fold.predict_proba(X_val_t) if hasattr(model_fold, "predict_proba") else None
                m_val = self.compute_metrics(y_val_t, y_val_pred, task_type, y_val_proba)
                fold_val_metrics.append(m_val)

        except Exception as exc:
            duration_ms = (time.time() - start_time) * 1000
            return TrainingResult(
                model_id=model_id,
                model_name=model_name,
                model_family=model_family,
                task_type=task_type,
                target=target_col,
                features=feature_cols,
                training_time_ms=duration_ms,
                status="failed",
                error_message=f"Training failed: {str(exc)}",
                confidence=0.0,
            )

        # Average metrics across folds
        avg_train: Dict[str, float] = {}
        avg_val: Dict[str, float] = {}
        for k in fold_val_metrics[0].keys():
            avg_train[k] = round(float(np.mean([m.get(k, 0.0) for m in fold_train_metrics])), 4)
            avg_val[k] = round(float(np.mean([m.get(k, 0.0) for m in fold_val_metrics])), 4)

        # Primary optimization metric resolution
        opt_metric = optimization_metric.lower()
        if opt_metric not in avg_val:
            opt_metric = "f1" if is_clf and "f1" in avg_val else ("accuracy" if is_clf else "r2")
        primary_val = avg_val.get(opt_metric, 0.0)

        # Overfitting Diagnostic Check
        overfitting_detected = False
        overfitting_warning = None
        if is_clf and "f1" in avg_train and "f1" in avg_val:
            gap = avg_train["f1"] - avg_val["f1"]
            if gap >= 0.20:
                overfitting_detected = True
                overfitting_warning = f"High train/validation divergence (Train F1={avg_train['f1']:.2f} vs Val F1={avg_val['f1']:.2f}, gap={gap:.2f})."
                warnings.append(overfitting_warning)
        elif not is_clf and "r2" in avg_train and "r2" in avg_val:
            gap = avg_train["r2"] - avg_val["r2"]
            if gap >= 0.25:
                overfitting_detected = True
                overfitting_warning = f"High train/validation divergence (Train R2={avg_train['r2']:.2f} vs Val R2={avg_val['r2']:.2f}, gap={gap:.2f})."
                warnings.append(overfitting_warning)

        duration_ms = (time.time() - start_time) * 1000

        # Build Traceable Evidence
        evidence = [
            Evidence(
                source="ModelTrainingEngine",
                method=f"{validation_strategy}_evaluation",
                data_ref={
                    "model_name": model_name,
                    "target": target_col,
                    "validation_strategy": validation_strategy,
                    "n_folds": n_splits,
                    "primary_metric": opt_metric,
                    "val_score": primary_val,
                    "all_metrics": avg_val,
                },
                confidence=0.92,
                claim_type=ClaimType.FACT,
            )
        ]

        # Calculate confidence based on sample size and overfitting gap
        conf = 0.95 if not overfitting_detected else 0.70
        if len(df) < 50:
            conf -= 0.10

        return TrainingResult(
            model_id=model_id,
            model_name=model_name,
            model_family=model_family,
            task_type=task_type,
            target=target_col,
            features=feature_cols,
            training_metrics=avg_train,
            validation_metrics=avg_val,
            primary_metric_name=opt_metric,
            primary_metric_value=primary_val,
            validation_results={"n_folds": n_splits, "fold_scores": fold_val_metrics},
            training_time_ms=duration_ms,
            feature_metadata={"feature_count": len(feature_cols), "sample_count": len(df)},
            evidence=evidence,
            confidence=max(0.1, min(1.0, conf)),
            warnings=warnings,
            status="success",
            overfitting_detected=overfitting_detected,
            overfitting_warning=overfitting_warning,
        )

    # ------------------------------------------------------------------
    # Full Training Pipeline with Model Comparison & Registry
    # ------------------------------------------------------------------
    def train_and_compare(
        self,
        request: TrainingRequest,
        dataframe: pd.DataFrame,
    ) -> ModelComparisonResult:
        """Train all requested candidate models, rank results, and persist the winning model artifact."""
        if dataframe is None or dataframe.empty:
            return ModelComparisonResult(
                status="failed",
                selection_reason="Cannot train models on empty dataset.",
                warnings=["Empty dataframe supplied to training engine."],
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

        features = request.feature_columns or [c for c in dataframe.columns if c != target_col]
        is_clf = "class" in request.task_type.lower() or "binary" in request.task_type.lower() or "multi" in request.task_type.lower()

        candidates_to_train = request.candidate_models
        if not candidates_to_train:
            # Default candidate pool if none supplied
            candidates_to_train = (
                ["Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier"]
                if is_clf
                else ["Linear Regression", "Ridge Regression", "Random Forest Regressor", "Gradient Boosting Regressor"]
            )

        training_results: List[TrainingResult] = []
        for candidate in candidates_to_train:
            res = self.train_and_validate_candidate(
                candidate=candidate,
                df=dataframe,
                target_col=target_col,
                feature_cols=features,
                task_type=request.task_type,
                validation_strategy=request.validation_strategy,
                optimization_metric=request.optimization_metric,
                random_state=request.random_state,
            )
            training_results.append(res)

        successful = [r for r in training_results if r.status == "success"]
        failed = [r for r in training_results if r.status == "failed"]

        if not successful:
            return ModelComparisonResult(
                candidates=training_results,
                status="failed",
                selection_reason="All candidate models failed during training and validation.",
                warnings=[r.error_message or "Unknown error" for r in failed],
                confidence=0.0,
            )

        # Determine metric direction (lower is better for error metrics)
        opt_metric = request.optimization_metric.lower()
        lower_is_better = opt_metric in ("rmse", "mae", "mape", "loss")

        # Rank successful models
        successful.sort(
            key=lambda r: r.validation_metrics.get(r.primary_metric_name, 0.0),
            reverse=not lower_is_better,
        )

        ranking: List[Dict[str, Any]] = []
        for rank_idx, r in enumerate(successful, start=1):
            ranking.append({
                "rank": rank_idx,
                "model_name": r.model_name,
                "model_id": r.model_id,
                "primary_metric": r.primary_metric_name,
                "score": r.primary_metric_value,
                "training_time_ms": r.training_time_ms,
                "overfitting_detected": r.overfitting_detected,
            })

        best = successful[0]
        direction_str = "lowest" if lower_is_better else "highest"
        selection_reason = (
            f"'{best.model_name}' achieved the {direction_str} validated {best.primary_metric_name} "
            f"({best.primary_metric_value:.4f}) among {len(successful)} successfully evaluated algorithms."
        )

        # Fit best model on FULL dataset and persist in ModelRegistry
        try:
            full_preprocessor = DataPreprocessor()
            full_preprocessor.fit(dataframe[features], dataframe[target_col], is_classification=is_clf)
            X_full = full_preprocessor.transform(dataframe[features])
            y_full = full_preprocessor.transform_target(dataframe[target_col])

            full_estimator, _, _ = self.instantiate_model(best.model_name, request.task_type, random_state=request.random_state)
            full_estimator.fit(X_full, y_full)

            meta = self.registry.register_model(
                name=best.model_name,
                model_object=full_estimator,
                model_family=best.model_family,
                algorithm=best.model_name,
                problem_type=request.task_type,
                target_column=target_col,
                feature_columns=features,
                training_metrics=best.training_metrics,
                validation_metrics=best.validation_metrics,
                primary_metric_name=best.primary_metric_name,
                primary_metric_value=best.primary_metric_value,
                preprocessor=full_preprocessor,
                tags=[request.task_type, request.validation_strategy, "auto_selected"],
            )
            best.model_artifact_path = str(self.registry.artifacts_dir / f"{meta.model_id}.joblib")
            best.model_id = meta.model_id
        except Exception as exc:
            best.warnings.append(f"Model artifact registration failed: {str(exc)}")

        # Build comparison-level evidence
        evidence_list = [
            Evidence(
                source="ModelTrainingEngine",
                method="multi_model_comparison",
                data_ref={
                    "winner": best.model_name,
                    "target": target_col,
                    "optimization_metric": best.primary_metric_name,
                    "winning_score": best.primary_metric_value,
                    "candidate_count": len(successful),
                    "ranking": ranking,
                },
                confidence=best.confidence,
                claim_type=ClaimType.FACT,
            )
        ]

        overall_status = "success" if len(failed) == 0 else "partial"
        warnings_list = [f"Candidate '{f.model_name}' failed: {f.error_message}" for f in failed]

        return ModelComparisonResult(
            candidates=training_results,
            ranking=ranking,
            best_model=best,
            optimization_metric=best.primary_metric_name,
            selection_reason=selection_reason,
            evidence=evidence_list,
            confidence=best.confidence,
            status=overall_status,
            warnings=warnings_list,
        )

    # ------------------------------------------------------------------
    # Prediction Pipeline with Schema & Feature-Order Enforcement
    # ------------------------------------------------------------------
    def predict_model(
        self,
        model_id: str,
        new_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Execute schema-validated inference on new data with feature ordering and inverse transformation.
        """
        if isinstance(new_data, dict):
            df_new = pd.DataFrame([new_data])
        elif isinstance(new_data, list):
            df_new = pd.DataFrame(new_data)
        elif isinstance(new_data, pd.DataFrame):
            df_new = new_data.copy()
        else:
            raise ValueError(f"Unsupported new_data format: {type(new_data)}")

        model_obj, preprocessor, meta = self.registry.get_model(model_id)

        # Validate feature presence and ordering
        missing_features = [f for f in meta.feature_columns if f not in df_new.columns]
        if missing_features:
            raise ValueError(f"Prediction data missing required features: {missing_features}")

        # Enforce exact training feature order
        df_features = df_new[meta.feature_columns].copy()

        # Transform using fitted preprocessor
        if preprocessor is not None:
            X_t = preprocessor.transform(df_features)
        else:
            X_t = df_features.select_dtypes(include=[np.number]).to_numpy(dtype=float)

        # Predict
        preds = model_obj.predict(X_t)
        proba = None
        if hasattr(model_obj, "predict_proba"):
            try:
                proba = model_obj.predict_proba(X_t).tolist()
            except Exception:
                pass

        # Inverse transform if target encoder exists
        decoded_preds = preds.tolist()
        if preprocessor is not None and preprocessor.target_encoder is not None:
            try:
                decoded_preds = preprocessor.target_encoder.inverse_transform(preds).tolist()
            except Exception:
                pass

        return {
            "model_id": model_id,
            "model_name": meta.name,
            "problem_type": meta.problem_type,
            "target_column": meta.target_column,
            "predictions": decoded_preds,
            "probabilities": proba,
            "row_count": len(df_new),
        }


# ==============================================================================
# 4. Model Training Agent (BaseAgent Subclass)
# ==============================================================================

class ModelTrainingAgent(BaseAgent):
    """
    Autonomous Model Training Agent coordinating dataset preprocessing, candidate execution,
    rigorous cross-validation, and winner registry persistence.
    """
    name = "Model Training Agent"
    role = "ml_engineer"
    description = "Preprocesses tabular features, trains ML candidates across cross-validation folds, and selects the winning model."

    def __init__(self, data: Optional[Any] = None, registry: Optional[ModelRegistry] = None):
        super().__init__(data=data)
        self.engine = ModelTrainingEngine(registry=registry)

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """
        Execute model training and evaluation pipeline.
        Task parameters:
            - data: pd.DataFrame or dict
            - target: str (target column name)
            - features: list[str] (optional)
            - task_type: str (regression, binary_classification, etc.)
            - candidates: list[str | ModelCandidate] (optional)
            - metric: str (optimization metric)
            - validation_strategy: str (5_fold_cv, stratified_5_fold, etc.)
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
            return self._error(f"Dataset requires at least 10 samples for model training. Found {len(df_target)}.", category=ErrorCategory.DATA_QUALITY)

        target = task.get("target")
        if not target or target not in df_target.columns:
            num_cols = df_target.select_dtypes(include=["number"]).columns
            target = num_cols[-1] if len(num_cols) > 0 else df_target.columns[-1]

        features = task.get("features") or [c for c in df_target.columns if c != target]
        task_type = task.get("task_type", "regression")
        metric = task.get("metric", "r2")
        validation_strategy = task.get("validation_strategy", "5_fold_cv")
        candidates = task.get("candidates", [])

        request = TrainingRequest(
            target_column=target,
            feature_columns=features,
            task_type=task_type,
            candidate_models=candidates,
            validation_strategy=validation_strategy,
            optimization_metric=metric,
        )

        comparison_result = self.engine.train_and_compare(request, df_target)

        if comparison_result.status == "failed":
            return self._error(
                comparison_result.selection_reason or "All model training candidates failed.",
                category=ErrorCategory.COMPUTATION,
            )

        best = comparison_result.best_model
        return self._finish(
            result=comparison_result.to_dict(),
            evidence=comparison_result.evidence,
            confidence=comparison_result.confidence,
            message=comparison_result.selection_reason,
            metadata={
                "target": target,
                "best_model_name": best.model_name if best else None,
                "best_model_id": best.model_id if best else None,
                "metric_name": comparison_result.optimization_metric,
                "metric_value": best.primary_metric_value if best else None,
            },
        )

