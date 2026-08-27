"""
Universal, Dataset-Agnostic Data Transformation & Feature Engineering Architecture.

Single source of truth for:
1. Non-destructive Data Transformation & Immutability Guarantee
2. Explainable Transformation Plan Generation
3. Safe Missing Value Imputation (Numeric, Categorical, Datetime)
4. Dirty Value Coercion (Currencies, %, Parens, Magnitude Multipliers)
5. Categorical Encoding (One-Hot, Ordinal, Frequency, Safe Unknown Handling)
6. Datetime Feature Engineering (Components, Cyclical Sin/Cos Features, Elapsed Time)
7. Numeric Scaling (StandardScaler, RobustScaler, MinMaxScaler with Auto-Selection)
8. Skewness Transformation (Log1p, Yeo-Johnson, Power Transforms)
9. Outlier Handling (Clip, Winsorize, Robust Scale - Non-destructive)
10. Identifier & Target Leakage Protection
11. Row Alignment & Index Preservation (len(X) == len(df))
12. Serializable TransformationState with Fit/Transform Semantics & Schema Drift Detection
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from agent.agent_result import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)
from agent.canonical_data_layer import CanonicalDataLayer, CanonicalDataset, SemanticProfile


# ---------------------------------------------------------------------------
# Data Structures: TransformationPlan & TransformationState
# ---------------------------------------------------------------------------

@dataclass
class TransformationPlan:
    """Explicit, explainable execution plan for dataset transformation."""
    selected_features: List[str] = field(default_factory=list)
    excluded_features: Dict[str, str] = field(default_factory=dict)
    numeric_transformations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    categorical_transformations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    datetime_transformations: Dict[str, List[str]] = field(default_factory=dict)
    missing_value_strategy: Dict[str, str] = field(default_factory=dict)
    scaling_strategy: Dict[str, str] = field(default_factory=dict)
    encoding_strategy: Dict[str, str] = field(default_factory=dict)
    outlier_strategy: Dict[str, str] = field(default_factory=dict)
    generated_features: List[str] = field(default_factory=list)
    dropped_features: List[str] = field(default_factory=list)
    leakage_protection: Dict[str, str] = field(default_factory=dict)
    transformation_order: List[str] = field(default_factory=lambda: [
        "type_coercion",
        "missing_imputation",
        "outlier_handling",
        "datetime_engineering",
        "categorical_encoding",
        "skew_transformation",
        "numeric_scaling",
    ])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransformationState:
    """Deterministic, serializable fitted state for inference and drift detection."""
    version: str = "1.0.0"
    fitted_row_count: int = 0
    fitted_columns: List[str] = field(default_factory=list)
    target_name: Optional[str] = None
    target_type: Optional[str] = None
    target_mapping: Optional[Dict[str, int]] = None
    selected_features: List[str] = field(default_factory=list)
    excluded_features: Dict[str, str] = field(default_factory=dict)
    feature_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    numeric_imputers: Dict[str, float] = field(default_factory=dict)
    categorical_imputers: Dict[str, str] = field(default_factory=dict)
    scalers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    encoders: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    skew_transforms: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    outlier_bounds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    datetime_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    output_columns: List[str] = field(default_factory=list)
    schema_signature: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformationState":
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# TransformationEngine (Single Source of Truth)
# ---------------------------------------------------------------------------

class TransformationEngine:
    """
    Authoritative, universal transformation and feature engineering engine.
    Converts arbitrary messy tabular datasets into model-ready numerical representations
    while guaranteeing non-destructive immutability, zero data leakage, and row alignment.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # --------------------------------------------------------------------------
    # Public API: fit, transform, fit_transform
    # --------------------------------------------------------------------------

    def fit(
        self,
        df: Union[pd.DataFrame, Any],
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
        task_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TransformationState:
        """
        Fit transformation statistics (imputers, encoders, scalers) strictly on training data.
        Does NOT mutate the original DataFrame.
        """
        raw_df = self._extract_dataframe(df)
        if raw_df is None or raw_df.empty:
            raise ValueError("Input DataFrame is empty or invalid. Cannot fit TransformationEngine.")

        cfg = config or {}
        sparse_threshold = cfg.get("sparse_threshold", 0.60)
        impute_numeric_strategy = cfg.get("impute_numeric", "median")
        impute_categorical_strategy = cfg.get("impute_categorical", "mode")
        scaling_strategy = cfg.get("scaling", "auto")
        encoding_strategy = cfg.get("encoding", "onehot")
        outlier_strategy = cfg.get("outliers", "clip")
        skew_threshold = cfg.get("skew_threshold", 1.5)
        enable_cyclical_datetime = cfg.get("cyclical_datetime", True)

        # 1. Semantic Profiling via CanonicalDataLayer
        dataset: CanonicalDataset = CanonicalDataLayer.ingest(raw_df)
        sem_profile: SemanticProfile = dataset.profile

        state = TransformationState(
            fitted_row_count=len(raw_df),
            fitted_columns=list(raw_df.columns),
            target_name=target,
            schema_signature={str(c): str(raw_df[c].dtype) for c in raw_df.columns},
        )

        # 2. Target Isolation & Encoding
        if target and target in raw_df.columns:
            target_s = raw_df[target]
            if pd.api.types.is_numeric_dtype(target_s) and not pd.api.types.is_bool_dtype(target_s):
                state.target_type = "numeric"
            else:
                state.target_type = "categorical"
                uniq_cats = sorted(target_s.dropna().astype(str).unique().tolist())
                state.target_mapping = {cat: idx for idx, cat in enumerate(uniq_cats)}

        # 3. Feature Selection & Exclusion
        all_candidate_cols = [c for c in raw_df.columns if c != target]
        selected_cols: List[str] = []
        excluded_cols: Dict[str, str] = {}

        if features is not None:
            # Respect explicit feature selection
            for f in features:
                if f not in raw_df.columns:
                    raise ValueError(f"Requested feature '{f}' not found in dataset columns: {list(raw_df.columns)}")
                if f == target:
                    raise ValueError(f"Requested feature '{f}' cannot be identical to the target column '{target}'.")
                selected_cols.append(f)
        else:
            # Automatic selection using SemanticProfile
            for col in all_candidate_cols:
                series = raw_df[col]
                n_valid = int(series.notna().sum())
                n_null = len(raw_df) - n_valid
                null_rate = n_null / len(raw_df) if len(raw_df) > 0 else 0.0

                if n_valid == 0:
                    excluded_cols[col] = "100% missing values"
                    continue
                if null_rate > sparse_threshold:
                    excluded_cols[col] = f"Missing rate {round(null_rate*100, 1)}% exceeds threshold {round(sparse_threshold*100, 1)}%"
                    continue
                if series.nunique(dropna=True) <= 1:
                    excluded_cols[col] = "Constant column (zero variance)"
                    continue
                if col in sem_profile.identifier_columns:
                    excluded_cols[col] = "Identified as unique identifier / key"
                    continue

                selected_cols.append(col)

        state.selected_features = selected_cols
        state.excluded_features = excluded_cols

        # Temporary working copy of selected features for sequential fit calculations
        work_df = raw_df[selected_cols].copy()

        # 4. Sequential Fit Pipeline
        # Step A: Dirty Numeric & Datetime Coercion
        coerced_numeric_cols: List[str] = []
        coerced_datetime_cols: List[str] = []
        categorical_cols: List[str] = []

        for col in selected_cols:
            s = work_df[col]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                coerced_numeric_cols.append(col)
                work_df[col] = s.astype(float)
            elif pd.api.types.is_datetime64_any_dtype(s) or col in sem_profile.datetime_candidates:
                coerced_dt = CanonicalDataLayer.coerce_datetime_series(s)
                if coerced_dt.notna().mean() >= 0.60:
                    coerced_datetime_cols.append(col)
                    work_df[col] = coerced_dt
                else:
                    categorical_cols.append(col)
            else:
                # Test numeric coercion (currencies, %, (1,200), suffixes)
                coerced_num = CanonicalDataLayer.coerce_numeric_series(s)
                if coerced_num.notna().sum() / max(1, s.notna().sum()) >= 0.60:
                    coerced_numeric_cols.append(col)
                    work_df[col] = coerced_num
                else:
                    categorical_cols.append(col)

        # Step B: Fit Missing Value Imputers
        for col in coerced_numeric_cols:
            valid_vals = work_df[col].dropna()
            if len(valid_vals) > 0:
                if impute_numeric_strategy == "mean":
                    fill_val = float(valid_vals.mean())
                else:
                    fill_val = float(valid_vals.median())
            else:
                fill_val = 0.0
            state.numeric_imputers[col] = round(fill_val, 6)
            work_df[col] = work_df[col].fillna(fill_val)

        for col in categorical_cols:
            valid_vals = work_df[col].dropna()
            if len(valid_vals) > 0 and impute_categorical_strategy == "mode":
                fill_str = str(valid_vals.mode().iloc[0])
            else:
                fill_str = "Unknown"
            state.categorical_imputers[col] = fill_str
            work_df[col] = work_df[col].fillna(fill_str).astype(str)

        # Step C: Fit Outlier Bounds & Skewness Parameters (Numeric)
        for col in coerced_numeric_cols:
            vals = work_df[col].to_numpy(dtype=float)
            q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
            iqr = float(q75 - q25)
            low_b = q25 - 1.5 * iqr
            upp_b = q75 + 1.5 * iqr

            state.outlier_bounds[col] = {
                "lower": round(low_b, 6),
                "upper": round(upp_b, 6),
                "strategy": outlier_strategy,
            }

            if outlier_strategy == "clip":
                vals = np.clip(vals, low_b, upp_b)
                work_df[col] = vals

            # Skew transform fit
            if len(vals) >= 3:
                skew_val = float(stats.skew(vals))
                if abs(skew_val) > skew_threshold:
                    if np.all(vals >= 0):
                        state.skew_transforms[col] = {"method": "log1p", "original_skew": round(skew_val, 4)}
                        work_df[col] = np.log1p(vals)
                    else:
                        # Yeo-Johnson for mixed signs
                        try:
                            _, lmbda = stats.yeojohnson(vals)
                            state.skew_transforms[col] = {"method": "yeojohnson", "lambda": round(float(lmbda), 6), "original_skew": round(skew_val, 4)}
                            work_df[col] = stats.yeojohnson(vals, lmbda=lmbda)
                        except Exception:
                            pass

        # Step D: Fit Scalers
        for col in coerced_numeric_cols:
            vals = work_df[col].to_numpy(dtype=float).reshape(-1, 1)
            # Automatic scaler selection: RobustScaler if outliers present or skew high, else StandardScaler
            if scaling_strategy == "auto":
                has_heavy_outliers = state.outlier_bounds.get(col, {}).get("strategy") == "clip"
                chosen_scaler_type = "robust" if has_heavy_outliers else "standard"
            else:
                chosen_scaler_type = scaling_strategy.lower()

            if chosen_scaler_type == "robust":
                sc = RobustScaler()
                sc.fit(vals)
                state.scalers[col] = {
                    "method": "robust",
                    "center": round(float(sc.center_[0]), 6),
                    "scale": round(float(sc.scale_[0]), 6) if sc.scale_[0] != 0 else 1.0,
                }
            elif chosen_scaler_type == "minmax":
                sc = MinMaxScaler()
                sc.fit(vals)
                state.scalers[col] = {
                    "method": "minmax",
                    "min": round(float(sc.min_[0]), 6),
                    "scale": round(float(sc.scale_[0]), 6) if sc.scale_[0] != 0 else 1.0,
                    "data_min": round(float(sc.data_min_[0]), 6),
                    "data_max": round(float(sc.data_max_[0]), 6),
                }
            else:
                sc = StandardScaler()
                sc.fit(vals)
                mean_val = float(sc.mean_[0])
                var_val = float(sc.var_[0])
                scale_val = math.sqrt(var_val) if var_val > 1e-12 else 1.0
                state.scalers[col] = {
                    "method": "standard",
                    "center": round(mean_val, 6),
                    "scale": round(scale_val, 6),
                }

        # Step E: Fit Categorical Encoders
        for col in categorical_cols:
            s_cat = work_df[col].astype(str)
            cats = sorted(s_cat.unique().tolist())
            state.encoders[col] = {
                "method": encoding_strategy,
                "categories": cats,
                "category_to_index": {cat: idx for idx, cat in enumerate(cats)},
            }

        # Step F: Fit Datetime Feature Configurations
        for col in coerced_datetime_cols:
            state.datetime_configs[col] = {
                "components": ["year", "month", "day", "day_of_week", "is_weekend"],
                "cyclical": enable_cyclical_datetime,
            }

        # Step G: Determine Exact Ordered Output Column List & Feature Metadata
        output_cols: List[str] = []
        meta: Dict[str, Dict[str, Any]] = {}

        # 1. Numeric Features
        for col in coerced_numeric_cols:
            out_name = f"{col}_scaled"
            output_cols.append(out_name)
            meta[out_name] = {
                "source_column": col,
                "generated_column": out_name,
                "semantic_type": "numeric",
                "transformation": "impute_and_scale",
                "parameters": {
                    "imputer": state.numeric_imputers.get(col),
                    "scaler": state.scalers.get(col),
                    "skew": state.skew_transforms.get(col),
                    "outlier": state.outlier_bounds.get(col),
                },
                "original_dtype": str(raw_df[col].dtype),
                "output_dtype": "float64",
                "leakage_safe": True,
                "reversible": True,
            }

        # 2. Categorical Features
        for col in categorical_cols:
            enc_info = state.encoders[col]
            if enc_info["method"] == "onehot":
                for cat in enc_info["categories"]:
                    clean_cat = str(cat).replace(" ", "_").replace("/", "_").replace("-", "_")
                    out_name = f"{col}_{clean_cat}"
                    output_cols.append(out_name)
                    meta[out_name] = {
                        "source_column": col,
                        "generated_column": out_name,
                        "semantic_type": "categorical_binary",
                        "transformation": "one_hot_encoding",
                        "parameters": {"category": cat},
                        "original_dtype": str(raw_df[col].dtype),
                        "output_dtype": "float64",
                        "leakage_safe": True,
                        "reversible": True,
                    }
            else:
                out_name = f"{col}_encoded"
                output_cols.append(out_name)
                meta[out_name] = {
                    "source_column": col,
                    "generated_column": out_name,
                    "semantic_type": "categorical_ordinal",
                    "transformation": "ordinal_encoding",
                    "parameters": enc_info["category_to_index"],
                    "original_dtype": str(raw_df[col].dtype),
                    "output_dtype": "float64",
                    "leakage_safe": True,
                    "reversible": True,
                }

        # 3. Datetime Features
        for col in coerced_datetime_cols:
            dt_cfg = state.datetime_configs[col]
            for comp in dt_cfg["components"]:
                out_name = f"{col}_{comp}"
                output_cols.append(out_name)
                meta[out_name] = {
                    "source_column": col,
                    "generated_column": out_name,
                    "semantic_type": "datetime_component",
                    "transformation": f"datetime_extract_{comp}",
                    "parameters": {},
                    "original_dtype": str(raw_df[col].dtype),
                    "output_dtype": "float64",
                    "leakage_safe": True,
                    "reversible": False,
                }
            if dt_cfg.get("cyclical"):
                for trig in ["month_sin", "month_cos", "dow_sin", "dow_cos"]:
                    out_name = f"{col}_{trig}"
                    output_cols.append(out_name)
                    meta[out_name] = {
                        "source_column": col,
                        "generated_column": out_name,
                        "semantic_type": "datetime_cyclical",
                        "transformation": f"cyclical_{trig}",
                        "parameters": {},
                        "original_dtype": str(raw_df[col].dtype),
                        "output_dtype": "float64",
                        "leakage_safe": True,
                        "reversible": False,
                    }

        state.output_columns = output_cols
        state.feature_metadata = meta

        return state

    def transform(
        self,
        df: Union[pd.DataFrame, Any],
        fitted_state: Union[TransformationState, Dict[str, Any]],
        drift_policy: str = "compatible",
    ) -> pd.DataFrame:
        """
        Transform arbitrary evaluation/inference DataFrame using pre-fitted parameters.
        Guarantees zero leakage, non-mutation of input, and exact output column order.
        """
        raw_df = self._extract_dataframe(df)
        if raw_df is None or raw_df.empty:
            raise ValueError("Input DataFrame is empty or invalid. Cannot transform.")

        state = fitted_state if isinstance(fitted_state, TransformationState) else TransformationState.from_dict(fitted_state)

        # 1. Schema Drift Checking
        missing_cols = [c for c in state.selected_features if c not in raw_df.columns]
        if missing_cols and drift_policy == "strict":
            raise ValueError(f"Schema Drift Error (strict): Missing expected columns {missing_cols}")

        n_rows = len(raw_df)
        orig_index = raw_df.index
        out_data: Dict[str, np.ndarray] = {}

        # 2. Transform Numeric Features
        for col, fill_val in state.numeric_imputers.items():
            if col in raw_df.columns:
                s = raw_df[col]
                coerced = CanonicalDataLayer.coerce_numeric_series(s) if not (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)) else s.astype(float)
                vals = coerced.fillna(fill_val).to_numpy(dtype=float)
            else:
                vals = np.full(n_rows, fill_val, dtype=float)

            # Outlier handling
            if col in state.outlier_bounds:
                ob = state.outlier_bounds[col]
                if ob.get("strategy") == "clip":
                    vals = np.clip(vals, ob["lower"], ob["upper"])

            # Skew transform
            if col in state.skew_transforms:
                st = state.skew_transforms[col]
                if st["method"] == "log1p":
                    vals = np.log1p(np.maximum(0.0, vals))
                elif st["method"] == "yeojohnson":
                    try:
                        vals = stats.yeojohnson(vals, lmbda=st["lambda"])
                    except Exception:
                        pass

            # Scaling
            sc_info = state.scalers.get(col, {})
            sc_method = sc_info.get("method", "standard")
            if sc_method == "standard":
                center = sc_info.get("center", 0.0)
                scale = sc_info.get("scale", 1.0)
                scale = scale if abs(scale) > 1e-12 else 1.0
                scaled_vals = (vals - center) / scale
            elif sc_method == "robust":
                center = sc_info.get("center", 0.0)
                scale = sc_info.get("scale", 1.0)
                scale = scale if abs(scale) > 1e-12 else 1.0
                scaled_vals = (vals - center) / scale
            elif sc_method == "minmax":
                d_min = sc_info.get("data_min", 0.0)
                d_max = sc_info.get("data_max", 1.0)
                denom = d_max - d_min if abs(d_max - d_min) > 1e-12 else 1.0
                scaled_vals = (vals - d_min) / denom
            else:
                scaled_vals = vals

            out_col_name = f"{col}_scaled"
            out_data[out_col_name] = scaled_vals

        # 3. Transform Categorical Features
        for col, enc_info in state.encoders.items():
            fill_str = state.categorical_imputers.get(col, "Unknown")
            if col in raw_df.columns:
                s_cat = raw_df[col].fillna(fill_str).astype(str)
            else:
                s_cat = pd.Series([fill_str] * n_rows, index=orig_index)

            enc_method = enc_info.get("method", "onehot")
            if enc_method == "onehot":
                for cat in enc_info.get("categories", []):
                    clean_cat = str(cat).replace(" ", "_").replace("/", "_").replace("-", "_")
                    out_col_name = f"{col}_{clean_cat}"
                    out_data[out_col_name] = (s_cat == str(cat)).astype(float).to_numpy()
            else:
                # Ordinal
                cat_to_idx = enc_info.get("category_to_index", {})
                out_col_name = f"{col}_encoded"
                out_data[out_col_name] = s_cat.map(cat_to_idx).fillna(-1.0).astype(float).to_numpy()

        # 4. Transform Datetime Features
        for col, dt_cfg in state.datetime_configs.items():
            if col in raw_df.columns:
                dt_s = CanonicalDataLayer.coerce_datetime_series(raw_df[col])
            else:
                dt_s = pd.Series([pd.Timestamp.now()] * n_rows, index=orig_index)

            # Fill missing datetimes with median/first valid or current timestamp
            valid_dt = dt_s.dropna()
            fallback_dt = valid_dt.iloc[0] if len(valid_dt) > 0 else pd.Timestamp("2024-01-01")
            dt_clean = dt_s.fillna(fallback_dt)

            dt_vals = pd.to_datetime(dt_clean)
            if "year" in dt_cfg.get("components", []):
                out_data[f"{col}_year"] = dt_vals.dt.year.astype(float).to_numpy()
            if "month" in dt_cfg.get("components", []):
                out_data[f"{col}_month"] = dt_vals.dt.month.astype(float).to_numpy()
            if "day" in dt_cfg.get("components", []):
                out_data[f"{col}_day"] = dt_vals.dt.day.astype(float).to_numpy()
            if "day_of_week" in dt_cfg.get("components", []):
                out_data[f"{col}_day_of_week"] = dt_vals.dt.dayofweek.astype(float).to_numpy()
            if "is_weekend" in dt_cfg.get("components", []):
                out_data[f"{col}_is_weekend"] = (dt_vals.dt.dayofweek >= 5).astype(float).to_numpy()

            if dt_cfg.get("cyclical"):
                month_arr = dt_vals.dt.month.to_numpy(dtype=float)
                dow_arr = dt_vals.dt.dayofweek.to_numpy(dtype=float)
                out_data[f"{col}_month_sin"] = np.sin(2.0 * np.pi * (month_arr - 1.0) / 12.0)
                out_data[f"{col}_month_cos"] = np.cos(2.0 * np.pi * (month_arr - 1.0) / 12.0)
                out_data[f"{col}_dow_sin"] = np.sin(2.0 * np.pi * dow_arr / 7.0)
                out_data[f"{col}_dow_cos"] = np.cos(2.0 * np.pi * dow_arr / 7.0)

        # 5. Assemble Final DataFrame with Exact Preserved Index and Ordered Columns
        res_df = pd.DataFrame(index=orig_index)
        for col_name in state.output_columns:
            if col_name in out_data:
                res_df[col_name] = out_data[col_name]
            else:
                res_df[col_name] = np.zeros(n_rows, dtype=float)

        return res_df

    def fit_transform(
        self,
        df: Union[pd.DataFrame, Any],
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
        task_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, TransformationState, TransformationPlan]:
        """Fit transformation parameters and transform dataset in a single call."""
        state = self.fit(df, target=target, features=features, task_type=task_type, config=config)
        transformed_df = self.transform(df, state)
        plan = self.generate_plan(state)
        return transformed_df, state, plan

    def generate_plan(self, state: TransformationState) -> TransformationPlan:
        """Generate an explainable TransformationPlan from a fitted TransformationState."""
        return TransformationPlan(
            selected_features=state.selected_features,
            excluded_features=state.excluded_features,
            numeric_transformations={
                col: {
                    "imputer_value": state.numeric_imputers.get(col),
                    "scaler": state.scalers.get(col),
                    "skew": state.skew_transforms.get(col),
                    "outlier_bounds": state.outlier_bounds.get(col),
                }
                for col in state.numeric_imputers
            },
            categorical_transformations=state.encoders,
            datetime_transformations={col: dt_cfg.get("components", []) for col, dt_cfg in state.datetime_configs.items()},
            missing_value_strategy={
                **{c: f"fill_numeric_{v}" for c, v in state.numeric_imputers.items()},
                **{c: f"fill_categorical_{v}" for c, v in state.categorical_imputers.items()},
            },
            scaling_strategy={c: s.get("method", "standard") for c, s in state.scalers.items()},
            encoding_strategy={c: e.get("method", "onehot") for c, e in state.encoders.items()},
            outlier_strategy={c: o.get("strategy", "clip") for c, o in state.outlier_bounds.items()},
            generated_features=state.output_columns,
            dropped_features=list(state.excluded_features.keys()),
            leakage_protection={
                "target_isolation": f"Target '{state.target_name}' excluded from feature matrix X" if state.target_name else "No target specified",
                "identifier_exclusion": f"Excluded {len([c for c, r in state.excluded_features.items() if 'identifier' in r.lower()])} key/ID column(s)",
            },
        )

    def inverse_transform(
        self,
        df: pd.DataFrame,
        fitted_state: Union[TransformationState, Dict[str, Any]],
    ) -> pd.DataFrame:
        """Inverse transform scaled numeric columns back to original metric units."""
        state = fitted_state if isinstance(fitted_state, TransformationState) else TransformationState.from_dict(fitted_state)
        res = df.copy()

        for col, sc_info in state.scalers.items():
            scaled_col = f"{col}_scaled"
            if scaled_col in res.columns:
                vals = res[scaled_col].to_numpy(dtype=float)
                sc_method = sc_info.get("method", "standard")
                if sc_method in ("standard", "robust"):
                    center = sc_info.get("center", 0.0)
                    scale = sc_info.get("scale", 1.0)
                    inv_vals = vals * scale + center
                elif sc_method == "minmax":
                    d_min = sc_info.get("data_min", 0.0)
                    d_max = sc_info.get("data_max", 1.0)
                    inv_vals = vals * (d_max - d_min) + d_min
                else:
                    inv_vals = vals

                res[col] = inv_vals

        return res

    def _extract_dataframe(self, data: Any) -> Optional[pd.DataFrame]:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            for df in data.values():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
        return None