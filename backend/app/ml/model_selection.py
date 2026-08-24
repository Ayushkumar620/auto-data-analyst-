"""Intelligent Machine Learning Model Selection & Comparison Engine.

Evaluates dataset characteristics, dynamically identifies problem types
(Binary Classification, Multiclass Classification, Regression), benchmarks a pool
of candidate algorithms (Linear, Ridge, Lasso, Decision Trees, Random Forest,
Gradient Boosting, HistGradientBoosting, SVM, k-NN), computes comprehensive
evaluation metrics and cross-validation scores, selects the best-performing model,
and provides transparent explainability for the selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class ProblemType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES_FORECAST = "time_series_forecast"


@dataclass
class ModelEvaluationResult:
    """Standardized performance benchmark for an individual trained model."""
    model_name: str
    model_family: str  # Linear, Tree, Ensemble, Kernel, Neighbors
    problem_type: ProblemType
    primary_metric_name: str
    primary_metric_value: float
    metrics: Dict[str, float] = field(default_factory=dict)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    cv_scores: List[float] = field(default_factory=list)
    training_duration_ms: float = 0.0
    feature_importances: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "problem_type": self.problem_type.value,
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_value": round(float(self.primary_metric_value), 4),
            "metrics": {k: round(float(v), 4) for k, v in self.metrics.items()},
            "cv_mean": round(float(self.cv_mean), 4),
            "cv_std": round(float(self.cv_std), 4),
            "cv_scores": [round(float(s), 4) for s in self.cv_scores],
            "training_duration_ms": round(float(self.training_duration_ms), 2),
            "feature_importances": {k: round(float(v), 4) for k, v in self.feature_importances.items()},
            "hyperparameters": self.hyperparameters,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass
class ModelComparisonReport:
    """Aggregated comparison report across all evaluated candidate models."""
    problem_type: ProblemType
    target_column: str
    feature_columns: List[str]
    candidate_evaluations: List[ModelEvaluationResult]
    best_model: ModelEvaluationResult
    selection_rationale: str
    leaderboard: List[Dict[str, Any]]
    dataset_characteristics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_type": self.problem_type.value,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "best_model": self.best_model.to_dict() if self.best_model else None,
            "selection_rationale": self.selection_rationale,
            "leaderboard": self.leaderboard,
            "candidate_evaluations": [e.to_dict() for e in self.candidate_evaluations],
            "dataset_characteristics": self.dataset_characteristics,
        }


class MLModelComparisonEngine:
    """Core engine for candidate model generation, benchmarking, and selection."""

    # ------------------------------------------------------------------
    # Problem Type & Dataset Inspection
    # ------------------------------------------------------------------
    @staticmethod
    def detect_problem_type(y: pd.Series) -> ProblemType:
        """Detect whether the target represents binary/multiclass classification or regression."""
        clean_y = y.dropna()
        if clean_y.empty:
            return ProblemType.REGRESSION

        # String / boolean / category targets are classification
        if clean_y.dtype == object or clean_y.dtype == bool or isinstance(clean_y.dtype, pd.CategoricalDtype):
            unique_count = clean_y.nunique()
            return ProblemType.BINARY_CLASSIFICATION if unique_count == 2 else ProblemType.MULTICLASS_CLASSIFICATION

        # Integer targets with very few unique values are classification
        unique_count = clean_y.nunique()
        if pd.api.types.is_integer_dtype(clean_y) and unique_count <= 10:
            return ProblemType.BINARY_CLASSIFICATION if unique_count == 2 else ProblemType.MULTICLASS_CLASSIFICATION

        # Binary float values (e.g. 0.0 and 1.0)
        if unique_count == 2 and set(clean_y.unique()).issubset({0, 1, 0.0, 1.0}):
            return ProblemType.BINARY_CLASSIFICATION

        return ProblemType.REGRESSION

    @staticmethod
    def inspect_dataset(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Compute statistical characteristics of the dataset for algorithm selection."""
        n_samples, n_features = X.shape
        missing_rate = float(X.isna().mean().mean())
        numeric_count = int(X.select_dtypes(include=[np.number]).shape[1])
        categorical_count = n_features - numeric_count

        characteristics = {
            "n_samples": n_samples,
            "n_features": n_features,
            "missing_rate": round(missing_rate, 4),
            "numeric_features": numeric_count,
            "categorical_features": categorical_count,
            "is_small_dataset": n_samples < 500,
            "is_wide_dataset": n_features > n_samples,
        }

        # Class balance check for classification
        if y.nunique() <= 10:
            class_counts = y.value_counts(normalize=True).to_dict()
            min_class_pct = min(class_counts.values()) if class_counts else 1.0
            characteristics["class_balance"] = {str(k): round(float(v), 3) for k, v in class_counts.items()}
            characteristics["is_imbalanced"] = min_class_pct < 0.20

        return characteristics

    # ------------------------------------------------------------------
    # Candidate Model Pools
    # ------------------------------------------------------------------
    def get_candidate_models(
        self,
        problem_type: ProblemType,
        n_samples: int,
        n_features: int,
    ) -> List[Tuple[str, str, Any, Dict[str, Any]]]:
        """
        Return a list of candidate model tuples: (model_name, model_family, model_instance, hyperparameters)
        Tailored to dataset size and problem type.
        """
        candidates: List[Tuple[str, str, Any, Dict[str, Any]]] = []

        if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            # 1. Baseline: Logistic Regression
            candidates.append((
                "Logistic Regression",
                "Linear",
                LogisticRegression(max_iter=1000, random_state=42),
                {"max_iter": 1000, "C": 1.0},
            ))

            # 2. Decision Tree
            candidates.append((
                "Decision Tree",
                "Tree",
                DecisionTreeClassifier(max_depth=5, random_state=42),
                {"max_depth": 5, "criterion": "gini"},
            ))

            # 3. Random Forest
            n_est = 50 if n_samples < 200 else 100
            candidates.append((
                "Random Forest",
                "Ensemble",
                RandomForestClassifier(n_estimators=n_est, max_depth=8, random_state=42, n_jobs=-1),
                {"n_estimators": n_est, "max_depth": 8},
            ))

            # 4. Gradient Boosting
            candidates.append((
                "Gradient Boosting",
                "Ensemble",
                GradientBoostingClassifier(n_estimators=n_est, max_depth=4, random_state=42),
                {"n_estimators": n_est, "max_depth": 4, "learning_rate": 0.1},
            ))

            # 5. K-Nearest Neighbors (good on low/medium dimensions)
            k = min(5, max(1, n_samples // 10))
            candidates.append((
                "K-Nearest Neighbors",
                "Neighbors",
                KNeighborsClassifier(n_neighbors=k),
                {"n_neighbors": k},
            ))

            # 6. Support Vector Classifier (if samples <= 2000 for scalability)
            if n_samples <= 2000:
                candidates.append((
                    "Support Vector Machine (SVC)",
                    "Kernel",
                    SVC(probability=True, random_state=42),
                    {"kernel": "rbf", "C": 1.0},
                ))

        else:
            # Regression Models
            # 1. Baseline: Linear Regression
            candidates.append((
                "Linear Regression",
                "Linear",
                LinearRegression(),
                {},
            ))

            # 2. Ridge Regression (L2 regularization)
            candidates.append((
                "Ridge Regression",
                "Linear",
                Ridge(alpha=1.0, random_state=42),
                {"alpha": 1.0},
            ))

            # 3. Lasso Regression (L1 regularization)
            candidates.append((
                "Lasso Regression",
                "Linear",
                Lasso(alpha=0.1, random_state=42),
                {"alpha": 0.1},
            ))

            # 4. Decision Tree Regressor
            candidates.append((
                "Decision Tree",
                "Tree",
                DecisionTreeRegressor(max_depth=5, random_state=42),
                {"max_depth": 5},
            ))

            # 5. Random Forest Regressor
            n_est = 50 if n_samples < 200 else 100
            candidates.append((
                "Random Forest",
                "Ensemble",
                RandomForestRegressor(n_estimators=n_est, max_depth=8, random_state=42, n_jobs=-1),
                {"n_estimators": n_est, "max_depth": 8},
            ))

            # 6. Gradient Boosting Regressor
            candidates.append((
                "Gradient Boosting",
                "Ensemble",
                GradientBoostingRegressor(n_estimators=n_est, max_depth=4, random_state=42),
                {"n_estimators": n_est, "max_depth": 4, "learning_rate": 0.1},
            ))

            # 7. K-Nearest Neighbors Regressor
            k = min(5, max(1, n_samples // 10))
            candidates.append((
                "K-Nearest Neighbors",
                "Neighbors",
                KNeighborsRegressor(n_neighbors=k),
                {"n_neighbors": k},
            ))

            # 8. SVR (if samples <= 2000)
            if n_samples <= 2000:
                candidates.append((
                    "Support Vector Regressor (SVR)",
                    "Kernel",
                    SVR(C=1.0),
                    {"kernel": "rbf", "C": 1.0},
                ))

        return candidates

    # ------------------------------------------------------------------
    # Preprocessing Pipeline
    # ------------------------------------------------------------------
    def prepare_data(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """Preprocess features and target: handles dates, categoricals, imputations, scaling."""
        df = dataframe.copy()
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe.")

        if feature_columns is None:
            features = [c for c in df.columns if c != target_column]
        else:
            features = [c for c in feature_columns if c in df.columns and c != target_column]

        if not features:
            raise ValueError("No valid feature columns available for modeling.")

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

        # Handle target encoding / cleaning
        problem_type = self.detect_problem_type(y_series)
        if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            y_clean = y_series.fillna(y_series.mode()[0] if not y_series.mode().empty else 0)
            le = LabelEncoder()
            y_arr = le.fit_transform(y_clean.astype(str))
        else:
            y_num = pd.to_numeric(y_series, errors="coerce")
            median_y = y_num.median()
            y_arr = y_num.fillna(median_y if not np.isnan(median_y) else 0.0).to_numpy(dtype=float)

        # Scale features
        scaler = StandardScaler()
        X_arr = scaler.fit_transform(X_df.to_numpy(dtype=float))

        meta = {
            "feature_names": features,
            "problem_type": problem_type,
            "n_samples": len(X_arr),
            "n_features": len(features),
        }

        return X_arr, y_arr, features, meta

    # ------------------------------------------------------------------
    # Model Benchmarking & Selection
    # ------------------------------------------------------------------
    def benchmark_models(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        cv_folds: int = 5,
        test_size: float = 0.2,
    ) -> ModelComparisonReport:
        """Train and cross-validate all candidate models, rank them, and select the best."""
        X, y, feature_names, meta = self.prepare_data(dataframe, target_column, feature_columns)
        problem_type: ProblemType = meta["problem_type"]
        n_samples = len(X)

        if n_samples < 10:
            raise ValueError(f"Need at least 10 valid samples to benchmark ML models. Found {n_samples}.")

        # Adjust CV folds if sample size is small
        actual_cv_folds = max(2, min(cv_folds, n_samples // 3))
        candidates = self.get_candidate_models(problem_type, n_samples, len(feature_names))

        # Train/Test split for holdout verification
        is_stratified = problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION)
        strat = y if is_stratified and min(np.bincount(y)) >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=strat
        )

        cv_splitter = (
            StratifiedKFold(n_splits=actual_cv_folds, shuffle=True, random_state=42)
            if is_stratified
            else KFold(n_splits=actual_cv_folds, shuffle=True, random_state=42)
        )

        evaluations: List[ModelEvaluationResult] = []

        for name, family, model_instance, hyperparams in candidates:
            start_t = time.time()
            try:
                # 1. Cross-validation scoring
                cv_scores: List[float] = []
                for train_idx, val_idx in cv_splitter.split(X_train, y_train):
                    fold_model = clone(model_instance)
                    fold_model.fit(X_train[train_idx], y_train[train_idx])
                    fold_preds = fold_model.predict(X_train[val_idx])

                    if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
                        score = float(accuracy_score(y_train[val_idx], fold_preds))
                    else:
                        score = float(r2_score(y_train[val_idx], fold_preds))
                    cv_scores.append(score)

                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))

                # 2. Fit on full training set and evaluate on test set
                final_model = clone(model_instance)
                final_model.fit(X_train, y_train)
                test_preds = final_model.predict(X_test)
                train_duration = (time.time() - start_t) * 1000

                metrics: Dict[str, float] = {}
                feature_importances: Dict[str, float] = {}

                # Extract feature importances or coefficients
                if hasattr(final_model, "feature_importances_"):
                    raw_imp = final_model.feature_importances_
                    feature_importances = {f: float(v) for f, v in zip(feature_names, raw_imp)}
                elif hasattr(final_model, "coef_"):
                    coef = np.abs(final_model.coef_).flatten()
                    if len(coef) == len(feature_names):
                        feature_importances = {f: float(v) for f, v in zip(feature_names, coef)}

                if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
                    test_acc = float(accuracy_score(y_test, test_preds))
                    test_f1 = float(f1_score(y_test, test_preds, average="weighted", zero_division=0))
                    test_prec = float(precision_score(y_test, test_preds, average="weighted", zero_division=0))
                    test_rec = float(recall_score(y_test, test_preds, average="weighted", zero_division=0))

                    metrics["accuracy"] = test_acc
                    metrics["f1_score"] = test_f1
                    metrics["precision"] = test_prec
                    metrics["recall"] = test_rec

                    primary_name = "f1_score" if meta.get("is_imbalanced") else "accuracy"
                    primary_val = test_f1 if meta.get("is_imbalanced") else test_acc

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

                evaluations.append(
                    ModelEvaluationResult(
                        model_name=name,
                        model_family=family,
                        problem_type=problem_type,
                        primary_metric_name=primary_name,
                        primary_metric_value=primary_val,
                        metrics=metrics,
                        cv_mean=cv_mean,
                        cv_std=cv_std,
                        cv_scores=cv_scores,
                        training_duration_ms=train_duration,
                        feature_importances=feature_importances,
                        hyperparameters=hyperparams,
                        status="success",
                    )
                )
            except Exception as exc:
                evaluations.append(
                    ModelEvaluationResult(
                        model_name=name,
                        model_family=family,
                        problem_type=problem_type,
                        primary_metric_name="score",
                        primary_metric_value=-999.0,
                        status="failed",
                        error_message=str(exc),
                    )
                )

        # Sort leaderboard by primary metric value descending
        successful_evals = [e for e in evaluations if e.status == "success"]
        if not successful_evals:
            raise RuntimeError("All candidate models failed during benchmarking.")

        successful_evals.sort(key=lambda e: (e.primary_metric_value, e.cv_mean), reverse=True)
        best_model = successful_evals[0]

        # Build Leaderboard
        leaderboard = []
        for rank, ev in enumerate(successful_evals, start=1):
            entry = {
                "rank": rank,
                "model_name": ev.model_name,
                "family": ev.model_family,
                "primary_metric": f"{ev.primary_metric_name} = {ev.primary_metric_value:.4f}",
                "cv_score": f"{ev.cv_mean:.4f} (±{ev.cv_std:.4f})",
                "training_duration_ms": f"{ev.training_duration_ms:.1f}ms",
            }
            entry.update(ev.metrics)
            leaderboard.append(entry)

        # Construct Selection Rationale
        rationale = self._generate_selection_rationale(best_model, successful_evals, problem_type, meta)

        dataset_chars = self.inspect_dataset(pd.DataFrame(X, columns=feature_names), pd.Series(y))

        return ModelComparisonReport(
            problem_type=problem_type,
            target_column=target_column,
            feature_columns=feature_names,
            candidate_evaluations=evaluations,
            best_model=best_model,
            selection_rationale=rationale,
            leaderboard=leaderboard,
            dataset_characteristics=dataset_chars,
        )

    def _generate_selection_rationale(
        self,
        best: ModelEvaluationResult,
        all_evals: List[ModelEvaluationResult],
        problem_type: ProblemType,
        meta: Dict[str, Any],
    ) -> str:
        """Synthesize a transparent justification explaining why the winning model was selected."""
        metric_name = best.primary_metric_name
        metric_val = best.primary_metric_value
        runner_up = all_evals[1] if len(all_evals) > 1 else None

        if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            if runner_up:
                margin = metric_val - runner_up.primary_metric_value
                return (
                    f"Selected {best.model_name} ({best.model_family}) as the top model. "
                    f"It achieved the highest test {metric_name} ({metric_val:.4f}) with robust cross-validation "
                    f"stability ({best.cv_mean:.4f} ± {best.cv_std:.4f}), outperforming runner-up {runner_up.model_name} "
                    f"({runner_up.primary_metric_name} = {runner_up.primary_metric_value:.4f}) by +{margin:.4f}."
                )
            return f"Selected {best.model_name} with {metric_name} of {metric_val:.4f}."
        else:
            if runner_up:
                margin = metric_val - runner_up.primary_metric_value
                return (
                    f"Selected {best.model_name} ({best.model_family}) as the top model. "
                    f"It demonstrated the highest variance explanation with R² of {metric_val:.4f} and "
                    f"lowest RMSE ({best.metrics.get('rmse', 0.0):.4f}), demonstrating superior non-linear capture "
                    f"over {runner_up.model_name} (R² = {runner_up.primary_metric_value:.4f})."
                )
            return f"Selected {best.model_name} with test R² of {metric_val:.4f}."
