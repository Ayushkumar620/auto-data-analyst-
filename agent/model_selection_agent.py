"""
Intelligent Machine Learning Model Selection Agent.

Automatically analyzes dataset characteristics, modality, and user intent to:
- Detect ML task type (regression, binary/multiclass classification, clustering, time series, image)
- Detect data modality (tabular, time series, image, spatial)
- Generate candidate algorithm pools across model families (linear, tree, ensemble, neural, clustering)
- Compute explainable suitability scores (0.0 to 1.0)
- Enforce CNN / ANN guardrails (e.g. CNN suitability = 0.0 for tabular data)
- Detect potential data leakage (target-derived features, identifier columns, future leakage)
- Select optimal evaluation metrics (F1 for imbalanced classification, R2/RMSE for regression, etc.)
- Hand off structured comparison plans to model training engines
- Benchmark and rank candidate models when execution is requested
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from agent.base import BaseAgent
from agent.dataset_knowledge import DatasetKnowledge
from agent.intent import CommandIntelligenceAgent, IntentType, UserIntent
from agent.model_selection_schemas import (
    DataModality,
    MLTaskType,
    ModelCandidate,
    ModelSelectionRequest,
    ModelSelectionResult,
)
from agent.schemas import AgentError, AgentResult, AgentStatus, ClaimType, ErrorCategory, Evidence
from backend.app.core.llm_provider import BaseLLMProvider, LLMClientFactory
from backend.app.ml.model_selection import (
    MLModelComparisonEngine,
    ModelComparisonReport,
    ProblemType,
)


class ModelSelectionAgent(BaseAgent):
    """
    Autonomous Machine Learning Model Selection Agent.
    Evaluates dataset structure, benchmarks candidate algorithms across model families,
    ranks them on standardized cross-validation & holdout metrics, and explains the winner.
    """
    name = "Model Selection Agent"
    role = "ml_engineer"
    description = "Intelligently selects and benchmarks candidate ML algorithms tailored to data modality and task."

    def __init__(
        self,
        data: Optional[Any] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
    ):
        super().__init__(data=data)
        self.engine = MLModelComparisonEngine()
        self.llm_provider = llm_provider
        self.command_agent = CommandIntelligenceAgent(llm_provider=llm_provider)

    # ------------------------------------------------------------------
    # 1. Modality & Task Type Detection
    # ------------------------------------------------------------------
    def detect_data_modality(
        self,
        df: pd.DataFrame,
        intent: Optional[UserIntent] = None,
        knowledge: Optional[DatasetKnowledge] = None,
    ) -> DataModality:
        """Detect whether input data represents tabular, time series, image, or spatial structures."""
        if df is None or df.empty:
            return DataModality.TABULAR

        # 1. Image Check: Look for pixel grid columns or 3D/4D shapes
        pixel_cols = [c for c in df.columns if re.match(r"^pixel[_\d]+$", str(c).lower()) or str(c).lower().startswith("px_")]
        if len(pixel_cols) >= 16 or len(df.columns) in (784, 1024, 4096, 3072):
            return DataModality.IMAGE

        # 2. Time Series Check: Date column present + sequential time series intent
        has_date = (
            any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)
            or (knowledge is not None and len(knowledge.date_columns) > 0)
        )
        if has_date and intent is not None:
            if intent.intent_type == IntentType.FORECASTING or "forecasting" in intent.required_capabilities:
                return DataModality.TIME_SERIES

        # 3. Spatial Check: Coordinate heatmaps or explicit lat/lon matrices
        spatial_cols = [c for c in df.columns if str(c).lower() in ("latitude", "longitude", "lat", "lon", "coord_x", "coord_y")]
        if len(spatial_cols) >= 2 and len(df.columns) <= 5:
            return DataModality.SPATIAL

        return DataModality.TABULAR

    def detect_task_type(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        intent: Optional[UserIntent] = None,
        knowledge: Optional[DatasetKnowledge] = None,
    ) -> MLTaskType:
        """Automatically determine the machine learning task type."""
        # 1. Check intent override
        if intent is not None:
            if intent.intent_type == IntentType.FORECASTING or "forecasting" in intent.required_capabilities:
                return MLTaskType.TIME_SERIES_FORECASTING
            if intent.intent_type == IntentType.ANOMALY_DETECTION or "anomaly_detection" in intent.required_capabilities:
                return MLTaskType.ANOMALY_DETECTION
            if "clustering" in intent.required_capabilities or (target_column is None and "grouping" in intent.objective.lower()):
                return MLTaskType.CLUSTERING

        # 2. If no target specified
        if not target_column or target_column not in df.columns:
            return MLTaskType.CLUSTERING

        # 3. Target Inspection
        target_series = df[target_column].dropna()
        if target_series.empty:
            return MLTaskType.UNKNOWN

        unique_count = target_series.nunique()

        # Binary classification
        if unique_count == 2:
            return MLTaskType.BINARY_CLASSIFICATION

        # Categorical / string / object target
        if (
            pd.api.types.is_object_dtype(target_series)
            or pd.api.types.is_string_dtype(target_series)
            or pd.api.types.is_bool_dtype(target_series)
            or isinstance(target_series.dtype, pd.CategoricalDtype)
        ):
            if unique_count <= 20:
                return MLTaskType.MULTICLASS_CLASSIFICATION
            return MLTaskType.REGRESSION

        # Integer target with few discrete categories
        if pd.api.types.is_integer_dtype(target_series) and unique_count <= 10:
            return MLTaskType.MULTICLASS_CLASSIFICATION

        # Continuous numeric target
        if pd.api.types.is_numeric_dtype(target_series):
            return MLTaskType.REGRESSION

        return MLTaskType.UNKNOWN

    # ------------------------------------------------------------------
    # 2. Data Leakage Detection
    # ------------------------------------------------------------------
    def detect_data_leakage(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        feature_columns: List[str],
    ) -> List[str]:
        """Identify potential target leakage, identifier features, and post-outcome columns."""
        leakage_warnings: List[str] = []
        if not target_column or target_column not in df.columns:
            return leakage_warnings

        target_base = re.sub(r"[_\W]+", "", target_column.lower())
        target_series = df[target_column]

        for col in feature_columns:
            if col == target_column or col not in df.columns:
                continue

            col_base = re.sub(r"[_\W]+", "", str(col).lower())

            # 1. Name derivative check (e.g. churn_reason when target is churn)
            if (target_base in col_base and col_base != target_base) or (col_base in target_base and len(col_base) > 4):
                leakage_warnings.append(
                    f"Feature '{col}' appears to be derived from target column '{target_column}' (possible target leakage)."
                )

            # 2. Identifier columns (e.g. customer_id, order_id)
            if any(id_kw in str(col).lower() for id_kw in ("customer_id", "user_id", "order_id", "account_id", "uuid", "row_id")):
                leakage_warnings.append(
                    f"Feature '{col}' is an entity identifier and should not be used as a predictive feature."
                )
            elif df[col].nunique() == len(df) and len(df) > 20:
                leakage_warnings.append(
                    f"Feature '{col}' has 100% unique values (high cardinality identifier), risking memorization."
                )

            # 3. Post-outcome / future tokens
            if any(post_kw in str(col).lower() for post_kw in ("next_", "future_", "post_", "subsequent_", "outcome_")):
                leakage_warnings.append(
                    f"Feature '{col}' indicates a future or post-outcome event, causing temporal leakage."
                )

            # 4. Near-perfect correlation check for numeric features
            if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(target_series):
                valid_mask = df[col].notna() & target_series.notna()
                if valid_mask.sum() > 5:
                    corr = float(np.abs(np.corrcoef(df.loc[valid_mask, col], target_series.loc[valid_mask])[0, 1]))
                    if not np.isnan(corr) and corr >= 0.99:
                        leakage_warnings.append(
                            f"Feature '{col}' has near-perfect correlation (|r| = {corr:.3f}) with target '{target_column}', indicating direct leakage."
                        )

        return leakage_warnings

    # ------------------------------------------------------------------
    # 3. Evaluation Metric Selection
    # ------------------------------------------------------------------
    def select_evaluation_metrics(
        self,
        task_type: MLTaskType,
        target_series: Optional[pd.Series] = None,
    ) -> Tuple[str, List[str]]:
        """Select primary and secondary evaluation metrics based on task type and class balance."""
        if task_type == MLTaskType.REGRESSION:
            return "r2", ["r2", "rmse", "mae"]

        if task_type == MLTaskType.BINARY_CLASSIFICATION:
            # Check class balance
            if target_series is not None and not target_series.empty:
                val_counts = target_series.value_counts(normalize=True)
                min_class_ratio = float(val_counts.min()) if not val_counts.empty else 0.5
                if min_class_ratio <= 0.25:  # Imbalanced
                    return "f1", ["f1", "roc_auc", "precision", "recall", "accuracy"]
            return "roc_auc", ["roc_auc", "accuracy", "f1", "precision", "recall"]

        if task_type == MLTaskType.MULTICLASS_CLASSIFICATION:
            if target_series is not None and not target_series.empty:
                val_counts = target_series.value_counts(normalize=True)
                if val_counts.min() <= 0.15:
                    return "f1_weighted", ["f1_weighted", "accuracy", "precision_weighted", "recall_weighted"]
            return "accuracy", ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

        if task_type == MLTaskType.CLUSTERING:
            return "silhouette", ["silhouette", "davies_bouldin"]

        if task_type == MLTaskType.TIME_SERIES_FORECASTING:
            return "rmse", ["rmse", "mae", "mape"]

        if task_type == MLTaskType.IMAGE_CLASSIFICATION:
            return "accuracy", ["accuracy", "top_5_accuracy"]

        return "accuracy", ["accuracy"]

    # ------------------------------------------------------------------
    # 4. Candidate Model Generation & Suitability Scoring
    # ------------------------------------------------------------------
    def generate_and_score_candidates(
        self,
        task_type: MLTaskType,
        modality: DataModality,
        n_samples: int,
        n_features: int,
        preferred_interpretability: str = "medium",
        is_imbalanced: bool = False,
    ) -> List[ModelCandidate]:
        """Generate algorithm pool with transparent suitability scores (0.0 to 1.0)."""
        candidates: List[ModelCandidate] = []

        # ==================== TABULAR REGRESSION ====================
        if task_type == MLTaskType.REGRESSION and modality == DataModality.TABULAR:
            # Linear Regression
            lin_score = 0.85 if n_samples < 200 and n_features <= 10 else 0.70
            if preferred_interpretability == "high":
                lin_score += 0.10
            candidates.append(
                ModelCandidate(
                    model_name="Linear Regression",
                    model_family="linear",
                    supported_tasks=["regression"],
                    suitability_score=min(lin_score, 1.0),
                    reason="Fast, highly interpretable baseline for continuous targets with linear trends.",
                    requirements=["Numeric feature scaling", "No severe collinearity"],
                    hyperparameter_space={"fit_intercept": [True, False]},
                )
            )

            # Ridge / Lasso
            reg_score = 0.85 if n_features > 10 else 0.75
            candidates.append(
                ModelCandidate(
                    model_name="Ridge Regression",
                    model_family="linear",
                    supported_tasks=["regression"],
                    suitability_score=reg_score,
                    reason="L2-regularized linear model effective for handling multicollinearity across numeric features.",
                    requirements=["Standardized feature scaling"],
                    hyperparameter_space={"alpha": [0.01, 0.1, 1.0, 10.0]},
                )
            )

            # Random Forest Regressor
            rf_score = 0.92 if n_samples >= 30 else 0.80
            candidates.append(
                ModelCandidate(
                    model_name="Random Forest Regressor",
                    model_family="ensemble",
                    supported_tasks=["regression"],
                    suitability_score=rf_score,
                    reason="Nonlinear tree ensemble robust to outliers, mixed feature types, and non-scaled inputs.",
                    requirements=["Moderate memory for ensemble trees"],
                    hyperparameter_space={"n_estimators": [100, 200], "max_depth": [5, 10, None]},
                )
            )

            # Gradient Boosting Regressor
            gb_score = 0.94 if n_samples >= 50 else 0.78
            candidates.append(
                ModelCandidate(
                    model_name="Gradient Boosting Regressor",
                    model_family="ensemble",
                    supported_tasks=["regression"],
                    suitability_score=gb_score,
                    reason="Sequential boosting trees optimizing residual errors with high predictive accuracy on tabular data.",
                    requirements=["Numeric preprocessing"],
                    hyperparameter_space={"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
                )
            )

            # ANN (MLP Regressor)
            ann_score = 0.75 if n_samples >= 200 else (0.40 if n_samples < 50 else 0.60)
            candidates.append(
                ModelCandidate(
                    model_name="Artificial Neural Network (ANN/MLP)",
                    model_family="neural",
                    supported_tasks=["regression"],
                    suitability_score=ann_score,
                    reason="Deep Multi-Layer Perceptron capturing complex high-order non-linear feature interactions when sample size allows.",
                    requirements=["StandardScaler normalization", "Large sample size (N >= 200 recommended)"],
                    hyperparameter_space={"hidden_layer_sizes": [(128, 64), (64, 32)], "alpha": [0.0001, 0.001], "max_iter": [200]},
                )
            )

            # CNN (Rejected for Tabular Data)
            candidates.append(
                ModelCandidate(
                    model_name="Convolutional Neural Network (CNN)",
                    model_family="convolutional",
                    supported_tasks=["image_classification", "spatial"],
                    suitability_score=0.0,
                    reason="CNN requires 2D/3D spatial grid structure (images/tensors); not applicable for 1D tabular feature vectors.",
                    requirements=["2D image or spatial grid input"],
                    hyperparameter_space={},
                )
            )

        # ==================== TABULAR CLASSIFICATION ====================
        elif task_type in (MLTaskType.BINARY_CLASSIFICATION, MLTaskType.MULTICLASS_CLASSIFICATION) and modality == DataModality.TABULAR:
            # Logistic Regression
            lr_score = 0.85 if n_samples < 200 else 0.75
            if preferred_interpretability == "high":
                lr_score += 0.10
            candidates.append(
                ModelCandidate(
                    model_name="Logistic Regression",
                    model_family="linear",
                    supported_tasks=["binary_classification", "multiclass_classification"],
                    suitability_score=min(lr_score, 1.0),
                    reason="Probabilistic linear classifier providing calibrated probability estimates and feature coefficients.",
                    requirements=["Feature scaling", "Encoded categories"],
                    hyperparameter_space={"C": [0.1, 1.0, 10.0], "class_weight": ["balanced", None]},
                )
            )

            # Random Forest Classifier
            rf_score = 0.94 if is_imbalanced else 0.92
            candidates.append(
                ModelCandidate(
                    model_name="Random Forest Classifier",
                    model_family="ensemble",
                    supported_tasks=["binary_classification", "multiclass_classification"],
                    suitability_score=rf_score,
                    reason="Ensemble of decision trees with bagging; excellent on tabular data with class imbalance and mixed types.",
                    requirements=["Categorical label encoding"],
                    hyperparameter_space={"n_estimators": [100, 200], "class_weight": ["balanced", None]},
                )
            )

            # Gradient Boosting Classifier
            gb_score = 0.93 if n_samples >= 50 else 0.78
            candidates.append(
                ModelCandidate(
                    model_name="Gradient Boosting Classifier",
                    model_family="ensemble",
                    supported_tasks=["binary_classification", "multiclass_classification"],
                    suitability_score=gb_score,
                    reason="Iterative gradient boosting minimizing cross-entropy loss with state-of-the-art tabular accuracy.",
                    requirements=["Imputed missing values", "Numeric encoding"],
                    hyperparameter_space={"learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
                )
            )

            # ANN (MLP Classifier)
            ann_score = 0.75 if n_samples >= 200 else (0.35 if n_samples < 50 else 0.55)
            candidates.append(
                ModelCandidate(
                    model_name="Artificial Neural Network (ANN/MLP)",
                    model_family="neural",
                    supported_tasks=["binary_classification", "multiclass_classification"],
                    suitability_score=ann_score,
                    reason="Multi-layer neural network with non-linear activations; candidate for complex feature interactions.",
                    requirements=["StandardScaler normalization", "Large sample size (N >= 200)"],
                    hyperparameter_space={"hidden_layer_sizes": [(128, 64), (64, 32)], "alpha": [0.0001, 0.001]},
                )
            )

            # CNN (Rejected for Tabular Data)
            candidates.append(
                ModelCandidate(
                    model_name="Convolutional Neural Network (CNN)",
                    model_family="convolutional",
                    supported_tasks=["image_classification", "spatial"],
                    suitability_score=0.0,
                    reason="CNN requires 2D/3D spatial grid structure; not applicable for standard tabular 1D feature vectors.",
                    requirements=["2D image or spatial grid input"],
                    hyperparameter_space={},
                )
            )

        # ==================== CLUSTERING ====================
        elif task_type == MLTaskType.CLUSTERING:
            candidates.append(
                ModelCandidate(
                    model_name="K-Means Clustering",
                    model_family="cluster",
                    supported_tasks=["clustering"],
                    suitability_score=0.90,
                    reason="Centroid-based partition clustering optimal for spherical and evenly-sized group discovery.",
                    requirements=["Normalized continuous features", "Preset cluster count k"],
                    hyperparameter_space={"n_clusters": [2, 3, 5, 8]},
                )
            )
            candidates.append(
                ModelCandidate(
                    model_name="DBSCAN Clustering",
                    model_family="cluster",
                    supported_tasks=["clustering"],
                    suitability_score=0.82,
                    reason="Density-based clustering capable of discovering arbitrary cluster shapes and isolating outliers.",
                    requirements=["Feature scaling"],
                    hyperparameter_space={"eps": [0.3, 0.5, 0.8], "min_samples": [5, 10]},
                )
            )

        # ==================== TIME SERIES ====================
        elif task_type == MLTaskType.TIME_SERIES_FORECASTING or modality == DataModality.TIME_SERIES:
            candidates.append(
                ModelCandidate(
                    model_name="ARIMA / Exponential Smoothing",
                    model_family="forecasting",
                    supported_tasks=["time_series_forecasting"],
                    suitability_score=0.92,
                    reason="Statistical time series forecasting with auto-regressive moving average and trend decomposition.",
                    requirements=["Sequential timestamp index"],
                    hyperparameter_space={"periods": [5, 10]},
                )
            )

        # ==================== IMAGE / SPATIAL ====================
        elif modality == DataModality.IMAGE or task_type == MLTaskType.IMAGE_CLASSIFICATION:
            candidates.append(
                ModelCandidate(
                    model_name="Convolutional Neural Network (CNN)",
                    model_family="convolutional",
                    supported_tasks=["image_classification", "spatial"],
                    suitability_score=0.96,
                    reason="Hierarchical spatial convolution filters with translation invariance; optimal for 2D/3D image tensors.",
                    requirements=["2D/3D image grid tensor"],
                    hyperparameter_space={"filters": [16, 32], "kernel_size": [3]},
                )
            )
            candidates.append(
                ModelCandidate(
                    model_name="Random Forest (Flattened Pixels)",
                    model_family="ensemble",
                    supported_tasks=["image_classification"],
                    suitability_score=0.40,
                    reason="Non-spatial baseline evaluating flattened pixel vectors without convolutional inductive bias.",
                    requirements=["Flattened 1D vectors"],
                    hyperparameter_space={"n_estimators": [100]},
                )
            )

        # Sort candidates descending by suitability score
        candidates.sort(key=lambda c: c.suitability_score, reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # 5. Model Selection Planning Interface
    # ------------------------------------------------------------------
    def plan_model_selection(
        self,
        request: ModelSelectionRequest,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> ModelSelectionResult:
        """Analyze dataset and formulate a structured ModelSelectionResult without forcing execution."""
        df = dataframe if dataframe is not None else self.data
        if df is None or not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()

        # 1. Feature & Target Extraction
        target = request.target_column
        if not target and df is not None and not df.empty:
            num_cols = df.select_dtypes(include=["number"]).columns
            target = num_cols[-1] if len(num_cols) > 0 else df.columns[-1]

        features = request.feature_columns or [c for c in df.columns if c != target]

        # 2. Modality & Task Type
        intent_obj: Optional[UserIntent] = None
        if isinstance(request.user_intent, UserIntent):
            intent_obj = request.user_intent
        elif isinstance(request.user_intent, dict):
            intent_obj = UserIntent.from_dict(request.user_intent)

        modality = request.data_modality
        if not modality or modality == DataModality.TABULAR.value:
            modality = self.detect_data_modality(df, intent=intent_obj, knowledge=request.dataset_knowledge)
        if isinstance(modality, str):
            modality = DataModality(modality)

        task_type = request.task_type
        if not task_type or task_type == MLTaskType.UNKNOWN.value:
            task_type = self.detect_task_type(df, target_column=target, intent=intent_obj, knowledge=request.dataset_knowledge)
        if isinstance(task_type, str):
            task_type = MLTaskType(task_type)

        # 3. Target and Feature Validation Warnings
        warnings: List[str] = []
        if df.empty:
            warnings.append("Dataset is empty. Candidates are generated from schema prior specifications.")
        elif len(df) < 10:
            warnings.append(f"Sample size ({len(df)} rows) is small. Overfitting risk on complex models.")

        if target and target not in df.columns:
            warnings.append(f"Specified target '{target}' does not exist in dataset columns.")

        # 4. Data Leakage Detection
        leakage_warnings = self.detect_data_leakage(df, target, features)

        # 5. Metrics Selection
        target_series = df[target] if (target and target in df.columns) else None
        primary_metric, secondary_metrics = self.select_evaluation_metrics(task_type, target_series)

        # 6. Candidate Generation & Suitability Scoring
        n_samples = len(df) if not df.empty else 100
        n_features = len(features) if features else 5
        is_imbalanced = False
        if target_series is not None and not target_series.empty and target_series.nunique() <= 10:
            is_imbalanced = bool(target_series.value_counts(normalize=True).min() <= 0.25)

        candidates = self.generate_and_score_candidates(
            task_type=task_type,
            modality=modality,
            n_samples=n_samples,
            n_features=n_features,
            preferred_interpretability=request.preferred_interpretability,
            is_imbalanced=is_imbalanced,
        )

        selected_candidate = candidates[0].model_name if candidates else None
        selection_reason = (
            f"'{selected_candidate}' was selected as the top candidate for {task_type.value} on {modality.value} data "
            f"because it achieved the highest suitability score ({candidates[0].suitability_score:.2f}) based on "
            f"dataset size ({n_samples} rows, {n_features} features) and algorithmic inductive bias."
        ) if candidates else "No eligible candidates found."

        # 7. Comparison & Training Handoff Plan
        comparison_plan = {
            "task_type": task_type.value,
            "data_modality": modality.value,
            "target": target,
            "features": features,
            "evaluation_metric": primary_metric,
            "secondary_metrics": secondary_metrics,
            "selection_strategy": "5_fold_stratified_cv" if task_type in (MLTaskType.BINARY_CLASSIFICATION, MLTaskType.MULTICLASS_CLASSIFICATION) else "5_fold_kfold_cv",
            "candidate_order": [c.model_name for c in candidates if c.suitability_score > 0.0],
        }

        # 8. Evidence
        evidence_list: List[Evidence] = [
            Evidence(
                source="ModelSelectionAgent",
                method="suitability_scoring",
                data_ref={
                    "task_type": task_type.value,
                    "data_modality": modality.value,
                    "target_column": target,
                    "candidate_count": len(candidates),
                    "top_model": selected_candidate,
                    "primary_metric": primary_metric,
                },
                confidence=0.92,
                claim_type=ClaimType.INFERENCE,
            )
        ]

        return ModelSelectionResult(
            selected_model=selected_candidate,
            candidates=candidates,
            task_type=task_type.value,
            data_modality=modality.value,
            evaluation_metric=primary_metric,
            secondary_metrics=secondary_metrics,
            selection_reason=selection_reason,
            confidence=0.92,
            warnings=warnings,
            leakage_warnings=leakage_warnings,
            evidence=evidence_list,
            comparison_plan=comparison_plan,
            target_column=target,
            feature_columns=features,
        )

    def select_candidates(
        self,
        command_or_intent: Union[str, UserIntent],
        dataframe: pd.DataFrame,
        dataset_knowledge: Optional[DatasetKnowledge] = None,
    ) -> ModelSelectionResult:
        """High-level natural language entry point for autonomous model selection."""
        if dataset_knowledge is None and dataframe is not None and not dataframe.empty:
            from backend.app.core.semantic import SemanticSchemaAgent
            dataset_knowledge = SemanticSchemaAgent().build_knowledge(dataframe)

        intent: UserIntent
        if isinstance(command_or_intent, str):
            intent = self.command_agent.analyze_intent(command_or_intent, dataset_knowledge=dataset_knowledge)
        else:
            intent = command_or_intent

        target = intent.metrics[0] if intent.metrics else None
        if not target and dataframe is not None:
            cmd_lower = str(command_or_intent).lower()
            for col in dataframe.columns:
                if str(col).lower() in cmd_lower:
                    target = str(col)
                    break

        features = intent.dimensions if intent.dimensions else [c for c in dataframe.columns if c != target]

        request = ModelSelectionRequest(
            dataset_knowledge=dataset_knowledge,
            target_column=target,
            feature_columns=features,
            user_intent=intent,
        )
        return self.plan_model_selection(request, dataframe=dataframe)

    # ------------------------------------------------------------------
    # 6. Benchmark Execution (BaseAgent run implementation)
    # ------------------------------------------------------------------
    def run(self, task: Dict[str, Any]) -> AgentResult:
        """
        Execute ML model selection and benchmarking on the target dataset.
        Task parameters:
            - data: pd.DataFrame or dict
            - target: str (target column name, auto-detected if omitted)
            - features: list[str]
            - cv_folds: int (default: 5)
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
            return self._error(f"Dataset requires at least 10 samples for ML benchmarking. Found {len(df_target)}.", category=ErrorCategory.DATA_QUALITY)

        # Detect target column if not provided
        target = task.get("target")
        if not target or target not in df_target.columns:
            num_cols = df_target.select_dtypes(include=["number"]).columns
            target = num_cols[-1] if len(num_cols) > 0 else df_target.columns[-1]

        features = task.get("features")
        cv_folds = int(task.get("cv_folds", 5))

        # Check leakage
        leakage = self.detect_data_leakage(df_target, target, features or [c for c in df_target.columns if c != target])

        try:
            report: ModelComparisonReport = self.engine.benchmark_models(
                dataframe=df_target,
                target_column=target,
                feature_columns=features,
                cv_folds=cv_folds,
            )
        except Exception as exc:
            return self._error(f"Model benchmarking failed: {str(exc)}", category=ErrorCategory.COMPUTATION)

        # Build traceable Evidence
        evidence_list: List[Evidence] = []
        best = report.best_model

        if best:
            evidence_list.append(
                self.make_evidence(
                    method="cross_validation_benchmark",
                    data_ref={
                        "target": target,
                        "model_name": best.model_name,
                        "problem_type": report.problem_type.value,
                        "primary_metric": best.primary_metric_name,
                        "metric_value": best.primary_metric_value,
                        "cv_mean": best.cv_mean,
                        "cv_std": best.cv_std,
                    },
                    confidence=0.95,
                    claim_type=ClaimType.FACT,
                )
            )

            if best.feature_importances:
                sorted_feats = sorted(best.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
                evidence_list.append(
                    self.make_evidence(
                        method="feature_importance_analysis",
                        data_ref={
                            "model_name": best.model_name,
                            "top_features": {k: v for k, v in sorted_feats},
                        },
                        confidence=0.90,
                        claim_type=ClaimType.INFERENCE,
                    )
                )

        output_payload = report.to_dict()
        if leakage:
            output_payload["leakage_warnings"] = leakage

        return self._finish(
            result=output_payload,
            evidence=evidence_list,
            confidence=0.95 if best else 0.5,
            message=f"Evaluated {len(report.candidate_evaluations)} candidate models. Best: {best.model_name if best else 'None'} ({best.primary_metric_name}={best.primary_metric_value:.4f})",
            metadata={
                "problem_type": report.problem_type.value,
                "target": target,
                "best_model_name": best.model_name if best else None,
                "candidate_count": len(report.candidate_evaluations),
            },
        )
