"""Unified Machine Learning & Deep Learning Model Registry.

Provides versioning, artifact serialization, metadata tracking, schema validation,
and inference execution across all model families (Traditional ML, ANN, CNN, Forecasting).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class ModelArtifactMetadata:
    """Metadata schema for a registered model artifact."""
    model_id: str
    name: str
    version: int
    model_family: str  # "traditional_ml", "ann", "cnn", "forecasting"
    algorithm: str
    problem_type: str  # "binary_classification", "multiclass_classification", "regression", "time_series_forecast"
    target_column: str
    feature_columns: List[str]
    feature_dtypes: Dict[str, str]
    hyperparameters: Dict[str, Any]
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    primary_metric_name: str
    primary_metric_value: float
    loss_curve: List[float] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"  # "active", "staging", "archived"
    tags: List[str] = field(default_factory=list)
    preprocessor_meta: Dict[str, Any] = field(default_factory=dict)
    reference_profile: Dict[str, Any] = field(default_factory=dict)
    feature_importances: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelArtifactMetadata:
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


class ModelRegistry:
    """
    Persistent registry for storing, versioning, querying, and deploying
    machine learning and deep learning models.
    """

    def __init__(self, registry_dir: Optional[Union[str, Path]] = None):
        if registry_dir is None:
            # Default storage inside backend storage directory
            base_path = Path(__file__).resolve().parent.parent.parent.parent
            self.registry_dir = base_path / "storage" / "model_registry"
        else:
            self.registry_dir = Path(registry_dir)

        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.registry_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.registry_dir / "registry_index.json"
        self._init_index()

    def _init_index(self) -> None:
        if not self.index_file.exists():
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _read_index(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_index(self, index_data: Dict[str, Dict[str, Any]]) -> None:
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

    # ------------------------------------------------------------------
    # Model Registration
    # ------------------------------------------------------------------
    def register_model(
        self,
        name: str,
        model_object: Any,
        model_family: str,
        algorithm: str,
        problem_type: str,
        target_column: str,
        feature_columns: List[str],
        feature_dtypes: Optional[Dict[str, str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_metrics: Optional[Dict[str, float]] = None,
        validation_metrics: Optional[Dict[str, float]] = None,
        primary_metric_name: str = "score",
        primary_metric_value: float = 0.0,
        loss_curve: Optional[List[float]] = None,
        preprocessor: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        reference_profile: Optional[Dict[str, Any]] = None,
        feature_importances: Optional[Dict[str, float]] = None,
    ) -> ModelArtifactMetadata:
        """Serialize a trained model and save complete metadata to the registry."""
        index = self._read_index()

        # Determine version number for this model name
        existing_versions = [
            meta["version"]
            for meta in index.values()
            if meta.get("name") == name
        ]
        version = max(existing_versions) + 1 if existing_versions else 1

        model_id = f"mod_{uuid.uuid4().hex[:10]}"
        artifact_path = self.artifacts_dir / f"{model_id}.joblib"
        prep_path = self.artifacts_dir / f"{model_id}_prep.joblib" if preprocessor else None

        # Save binary artifact
        joblib.dump(model_object, artifact_path)
        if preprocessor and prep_path:
            joblib.dump(preprocessor, prep_path)

        metadata = ModelArtifactMetadata(
            model_id=model_id,
            name=name,
            version=version,
            model_family=model_family,
            algorithm=algorithm,
            problem_type=problem_type,
            target_column=target_column,
            feature_columns=feature_columns,
            feature_dtypes=feature_dtypes or {},
            hyperparameters=hyperparameters or {},
            training_metrics=training_metrics or {},
            validation_metrics=validation_metrics or {},
            primary_metric_name=primary_metric_name,
            primary_metric_value=round(float(primary_metric_value), 4),
            loss_curve=loss_curve or [],
            status="active",
            tags=tags or [],
            preprocessor_meta={"has_preprocessor": preprocessor is not None},
            reference_profile=reference_profile or {},
            feature_importances=feature_importances or {},
        )

        # Update index
        index[model_id] = metadata.to_dict()
        self._write_index(index)

        return metadata

    # ------------------------------------------------------------------
    # Query & Retrieval
    # ------------------------------------------------------------------
    def get_metadata(self, model_id: str) -> Optional[ModelArtifactMetadata]:
        """Fetch metadata for a registered model ID."""
        index = self._read_index()
        if model_id not in index:
            return None
        return ModelArtifactMetadata.from_dict(index[model_id])

    def get_model(self, model_id: str) -> Tuple[Any, Optional[Any], ModelArtifactMetadata]:
        """Load the model object, preprocessor (if any), and metadata from disk."""
        meta = self.get_metadata(model_id)
        if meta is None:
            raise KeyError(f"Model ID '{model_id}' not found in registry.")

        artifact_path = self.artifacts_dir / f"{model_id}.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model binary artifact missing at {artifact_path}")

        model_obj = joblib.load(artifact_path)

        prep_obj = None
        prep_path = self.artifacts_dir / f"{model_id}_prep.joblib"
        if prep_path.exists():
            prep_obj = joblib.load(prep_path)

        return model_obj, prep_obj, meta

    def list_models(
        self,
        family: Optional[str] = None,
        problem_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List registered models sorted by creation date descending."""
        index = self._read_index()
        results = []

        for meta_dict in index.values():
            if family and meta_dict.get("model_family") != family:
                continue
            if problem_type and meta_dict.get("problem_type") != problem_type:
                continue
            if status and meta_dict.get("status") != status:
                continue
            results.append(meta_dict)

        results.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return results

    def set_status(self, model_id: str, status: str) -> bool:
        """Update lifecycle status of a model (e.g. 'active', 'staging', 'archived')."""
        index = self._read_index()
        if model_id not in index:
            return False
        index[model_id]["status"] = status
        self._write_index(index)
        return True

    def delete_model(self, model_id: str) -> bool:
        """Remove a registered model and its artifacts from storage."""
        index = self._read_index()
        if model_id not in index:
            return False

        del index[model_id]
        self._write_index(index)

        # Delete artifact files
        artifact_path = self.artifacts_dir / f"{model_id}.joblib"
        if artifact_path.exists():
            try:
                os.remove(artifact_path)
            except OSError:
                pass

        prep_path = self.artifacts_dir / f"{model_id}_prep.joblib"
        if prep_path.exists():
            try:
                os.remove(prep_path)
            except OSError:
                pass

        return True

    # ------------------------------------------------------------------
    # Live Model Inference Execution
    # ------------------------------------------------------------------
    def predict(
        self,
        model_id: str,
        input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Execute model inference with schema validation and preprocessor scaling."""
        model_obj, preprocessor, meta = self.get_model(model_id)

        # Convert input to DataFrame
        if isinstance(input_data, dict):
            df_in = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df_in = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df_in = input_data.copy()
        else:
            raise TypeError("input_data must be a dict, list of dicts, or pandas DataFrame.")

        # Validate and align expected features
        missing_features = [col for col in meta.feature_columns if col not in df_in.columns]
        if missing_features:
            raise ValueError(f"Input data is missing required feature columns: {missing_features}")

        X_df = df_in[meta.feature_columns].copy()

        # Handle datetimes and categoricals
        for col in X_df.columns:
            if pd.api.types.is_datetime64_any_dtype(X_df[col]):
                X_df[col] = pd.to_datetime(X_df[col]).astype("int64") // 10**9

        for col in X_df.select_dtypes(include=["object", "string", "category"]).columns:
            X_df[col] = X_df[col].fillna("Missing").astype(str)
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col])

        # Impute numeric
        for col in X_df.select_dtypes(include=[np.number]).columns:
            med = X_df[col].median()
            X_df[col] = X_df[col].fillna(med if not np.isnan(med) else 0.0)

        X_mat = X_df.to_numpy(dtype=float)

        # Apply preprocessor if stored
        if preprocessor is not None:
            try:
                X_mat = preprocessor.transform(X_mat)
            except Exception:
                pass

        # Run model prediction
        raw_preds = model_obj.predict(X_mat)

        # Calculate prediction probabilities if available
        probabilities = None
        if hasattr(model_obj, "predict_proba"):
            try:
                raw_prob = model_obj.predict_proba(X_mat)
                probabilities = raw_prob.tolist()
            except Exception:
                probabilities = None

        preds_list = raw_preds.tolist() if isinstance(raw_preds, np.ndarray) else list(raw_preds)

        return {
            "model_id": model_id,
            "model_name": meta.name,
            "version": meta.version,
            "problem_type": meta.problem_type,
            "target_column": meta.target_column,
            "sample_count": len(df_in),
            "predictions": preds_list,
            "probabilities": probabilities,
        }

    # ------------------------------------------------------------------
    # Monitoring History Tracking
    # ------------------------------------------------------------------
    def record_monitoring_run(self, model_id: str, run_summary: Dict[str, Any]) -> None:
        """Persist a model monitoring assessment to the monitoring history log."""
        history_file = self.registry_dir / "monitoring_history.json"
        history = []
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

        entry = {
            "run_id": f"mon_{uuid.uuid4().hex[:8]}",
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            **run_summary,
        }
        history.append(entry)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def get_monitoring_history(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve monitoring history records filtered by model ID."""
        history_file = self.registry_dir / "monitoring_history.json"
        if not history_file.exists():
            return []
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            return []

        if model_id:
            return [h for h in history if h.get("model_id") == model_id]
        return history

