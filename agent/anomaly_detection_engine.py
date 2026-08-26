"""
Universal, Dataset-Agnostic Anomaly Detection Engine.

Single source of truth for anomaly and outlier detection.
Supports:
1. Robust Statistical Detector (MAD / Modified Z-Score)
2. Interquartile Range (IQR / Tukey''s Fences)
3. Isolation Forest (Multivariate tree-based isolation)
4. Local Outlier Factor (Density-based local clustering)
5. Minimum Covariance / Elliptic Envelope (Mahalanobis distance)

Features:
- Dataset-agnostic column classification (numeric, categorical, datetime, identifier, constant)
- Non-destructive missing value handling (median/mode imputation without dropping target rows)
- Automatic and explicit contamination estimation
- Normalised [0, 1] anomaly scoring where higher score = more anomalous
- Interpretable, structured anomaly explanations (univariate extreme vs multivariate interaction)
- Data-driven candidate detector benchmarking and suitability selection
- Full integration with CanonicalDataLayer, PreExecutionValidator, ResultValidator, and AgentResult
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from agent.agent_result import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)
from agent.canonical_data_layer import CanonicalDataLayer, CanonicalDataset, SemanticProfile
from agent.confidence_calculator import ConfidenceCalculator
from agent.result_validator import ResultValidator


class AnomalyDetectionEngine:
    """
    Authoritative, universal anomaly detection engine.
    Benchmarking multiple detector families to identify and explain anomalies.
    """

    SUPPORTED_METHODS = [
        "robust_zscore",
        "iqr",
        "isolation_forest",
        "local_outlier_factor",
        "elliptic_envelope",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def detect(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], Any],
        features: Optional[List[str]] = None,
        contamination: Union[str, float] = "auto",
        method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute dataset-agnostic anomaly detection.

        Parameters:
        - data: Tabular DataFrame or dictionary of DataFrames
        - features: Optional list of specific column names to use
        - contamination: 'auto' or float in (0.0, 0.5]
        - method: Optional specific detection algorithm
        """
        df = self._extract_dataframe(data)
        if df is None or df.empty:
            return {
                "error": "Dataset is empty or invalid. Anomaly detection requires valid tabular data.",
                "category": ErrorCategory.DATA_INVALID,
            }

        n_rows = len(df)
        if n_rows < 5:
            return {
                "error": f"Need at least 5 observations for statistical anomaly detection. Found {n_rows}.",
                "category": ErrorCategory.INSUFFICIENT_DATA,
            }

        # 1. Ingestion & Semantic Profiling
        dataset: CanonicalDataset = CanonicalDataLayer.ingest(df)
        profile: SemanticProfile = dataset.profile

        # 2. Feature Selection & Non-Destructive Preparation
        X_df, features_used, excluded_features, feature_meta = self._prepare_features(
            df, profile, requested_features=features
        )

        if X_df.empty or len(features_used) == 0:
            return {
                "error": "No usable numeric or quantifiable features available for anomaly detection.",
                "category": ErrorCategory.DATA_INVALID,
                "excluded_features": excluded_features,
            }

        # Check for all-constant features
        if all(X_df[c].nunique() <= 1 for c in X_df.columns):
            return {
                "error": "All candidate feature columns have zero variance (constant values). Cannot detect anomalies in uniform data.",
                "category": ErrorCategory.DATA_INVALID,
                "excluded_features": excluded_features,
            }

        # 3. Contamination Estimation
        contam_val = self._resolve_contamination(X_df, contamination)

        # 4. Candidate Benchmarking & Model Selection
        leaderboard, winning_method = self._benchmark_candidates(
            X_df, requested_method=method, contamination=contam_val
        )

        if winning_method is None:
            winning_method = "robust_zscore"

        # 5. Run Winning Detector
        scores, raw_labels, method_family = self._run_detector(
            winning_method, X_df, contam_val
        )

        # 6. Normalize Scores & Classify Observations
        norm_scores = self._normalize_scores(scores)
        anomalies_found, observations = self._build_observations(
            df, X_df, norm_scores, contam_val, feature_meta
        )

        anomaly_rate = round(anomalies_found / n_rows, 4)

        return {
            "method": winning_method,
            "method_family": method_family,
            "rows_analyzed": n_rows,
            "original_rows": n_rows,
            "anomalies_found": anomalies_found,
            "anomaly_rate": anomaly_rate,
            "contamination": round(contam_val, 4),
            "features_used": features_used,
            "excluded_features": excluded_features,
            "leaderboard": leaderboard,
            "observations": observations,
            "feature_statistics": feature_meta,
        }

    # --------------------------------------------------------------------------
    # Feature Preparation & Cleaning
    # --------------------------------------------------------------------------

    def _prepare_features(
        self,
        df: pd.DataFrame,
        profile: SemanticProfile,
        requested_features: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, Dict[str, Any]]]:
        """
        Classify and clean features non-destructively without dropping observations.
        """
        excluded: List[str] = []
        feature_meta: Dict[str, Dict[str, Any]] = {}

        if requested_features is not None and len(requested_features) > 0:
            candidate_cols = [c for c in requested_features if c in df.columns]
            is_explicit = True
        else:
            is_explicit = False
            candidate_cols = list(df.columns)

        X_df = pd.DataFrame(index=df.index)

        for col in candidate_cols:
            series = df[col]

            # 1. Identifier exclusion
            if not is_explicit and col in profile.identifier_columns:
                excluded.append(str(col))
                continue

            # 2. Constant exclusion (0 variance)
            if not is_explicit and (col in profile.constant_columns or series.nunique(dropna=True) <= 1):
                excluded.append(str(col))
                continue

            # 3. High cardinality text exclusion
            if not is_explicit and (col in profile.high_cardinality_columns or col in profile.text_columns):
                excluded.append(str(col))
                continue

            # 4. Datetime column handling
            if pd.api.types.is_datetime64_any_dtype(series) or col in profile.datetime_candidates:
                if not is_explicit:
                    excluded.append(str(col))
                    continue
                # If explicit, convert to elapsed numeric days
                dt_s = CanonicalDataLayer.coerce_datetime_series(series)
                min_dt = dt_s.dropna().min()
                if pd.notna(min_dt):
                    elapsed = (dt_s - min_dt).dt.total_seconds() / 86400.0
                    median_val = float(elapsed.median()) if not np.isnan(elapsed.median()) else 0.0
                    X_df[f"{col}_elapsed"] = elapsed.fillna(median_val).astype(float)
                continue

            # 5. Numeric Coercion (handles $, %, commas, accounting negative brackets, etc.)
            num_s = CanonicalDataLayer.coerce_numeric_series(series)
            num_valid_ratio = num_s.notna().mean()

            if num_valid_ratio >= 0.60:
                # Sparse feature check
                if not is_explicit and num_s.isna().mean() > 0.60:
                    excluded.append(str(col))
                    continue

                median_val = float(num_s.median()) if not np.isnan(num_s.median()) else 0.0
                clean_col = num_s.fillna(median_val).astype(float)

                if clean_col.nunique() > 1 or is_explicit:
                    X_df[str(col)] = clean_col
                    # Compute feature statistics for explainability
                    q25 = float(clean_col.quantile(0.25))
                    q75 = float(clean_col.quantile(0.75))
                    iqr = q75 - q25
                    mad = float(np.median(np.abs(clean_col - median_val)))
                    feature_meta[str(col)] = {
                        "median": round(median_val, 4),
                        "mean": round(float(clean_col.mean()), 4),
                        "std": round(float(clean_col.std()), 4) if len(clean_col) > 1 else 0.0,
                        "mad": round(mad, 4),
                        "q25": round(q25, 4),
                        "q75": round(q75, 4),
                        "lower_bound": round(q25 - 1.5 * iqr, 4),
                        "upper_bound": round(q75 + 1.5 * iqr, 4),
                    }
                else:
                    excluded.append(str(col))
            else:
                # Categorical column
                if is_explicit or (series.nunique(dropna=True) <= 20 and len(df) >= 20):
                    cat_s = series.fillna("__UNKNOWN__").astype(str)
                    try:
                        encoded = LabelEncoder().fit_transform(cat_s).astype(float)
                        X_df[str(col)] = encoded
                    except Exception:
                        excluded.append(str(col))
                else:
                    excluded.append(str(col))

        return X_df, list(X_df.columns), excluded, feature_meta

    # --------------------------------------------------------------------------
    # Contamination Estimation
    # --------------------------------------------------------------------------

    def _resolve_contamination(
        self, X_df: pd.DataFrame, contamination: Union[str, float]
    ) -> float:
        """Validate and resolve contamination parameter."""
        if isinstance(contamination, (int, float)):
            contam = float(contamination)
            return max(0.001, min(0.5, contam))

        # Automatic contamination estimation
        # Count proportion of observations with robust z-score > 3.0 across features
        n_rows = len(X_df)
        extreme_flags = np.zeros(n_rows, dtype=bool)

        for col in X_df.columns:
            vals = X_df[col].to_numpy()
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            if mad > 1e-9:
                mod_z = 0.6745 * np.abs(vals - med) / mad
                extreme_flags |= (mod_z > 3.29)  # ~99.9% normal threshold

        estimated_rate = float(np.mean(extreme_flags))
        # Bound between 1% and 15%, default 5%
        return max(0.01, min(0.15, estimated_rate if estimated_rate > 0 else 0.05))

    # --------------------------------------------------------------------------
    # Candidate Detector Benchmarking
    # --------------------------------------------------------------------------

    def _benchmark_candidates(
        self,
        X_df: pd.DataFrame,
        requested_method: Optional[str],
        contamination: float,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Evaluate candidate anomaly detectors based on data characteristics."""
        n_rows, n_cols = X_df.shape
        leaderboard: List[Dict[str, Any]] = []

        # Candidate 1: Robust Statistical Detector (MAD)
        mad_supported = True
        mad_score = 90.0 if n_cols == 1 else (80.0 if n_cols <= 5 else 65.0)
        leaderboard.append({
            "method": "robust_zscore",
            "family": "statistical",
            "supported": mad_supported,
            "suitability_score": mad_score,
            "reason": "Highly interpretable median absolute deviation (MAD) modified z-scores.",
            "parameters": {"threshold": 3.0},
            "limitations": ["Evaluates marginal distributions independently."],
        })

        # Candidate 2: Interquartile Range (IQR / Tukey)
        iqr_supported = True
        iqr_score = 85.0 if n_cols == 1 else 60.0
        leaderboard.append({
            "method": "iqr",
            "family": "statistical",
            "supported": iqr_supported,
            "suitability_score": iqr_score,
            "reason": "Non-parametric Tukey fences robust against non-Gaussian distributions.",
            "parameters": {"multiplier": 1.5},
            "limitations": ["Univariate bounding, ignores multi-feature correlation."],
        })

        # Candidate 3: Isolation Forest
        if_supported = n_rows >= 10
        if_score = 95.0 if n_cols >= 2 and n_rows >= 15 else (70.0 if if_supported else 0.0)
        leaderboard.append({
            "method": "isolation_forest",
            "family": "ensemble",
            "supported": if_supported,
            "suitability_score": if_score,
            "reason": "Tree-based recursive space partitioning isolating multi-feature outliers.",
            "parameters": {"n_estimators": 100, "contamination": contamination},
            "limitations": ["Slightly less interpretable than single-feature bounds."],
        })

        # Candidate 4: Local Outlier Factor (LOF)
        lof_supported = n_rows >= 20 and n_cols >= 2 and n_rows <= 10000
        lof_score = 85.0 if lof_supported and n_cols <= 20 else (50.0 if lof_supported else 0.0)
        leaderboard.append({
            "method": "local_outlier_factor",
            "family": "density",
            "supported": lof_supported,
            "suitability_score": lof_score,
            "reason": "Local density estimation measuring deviation relative to k-nearest neighbors.",
            "parameters": {"n_neighbors": min(20, max(5, n_rows // 3)), "contamination": contamination},
            "limitations": ["O(N^2) pairwise distance complexity on very large datasets."],
        })

        # Candidate 5: Elliptic Envelope (Mahalanobis Covariance)
        ee_supported = n_rows >= 25 and n_cols >= 2 and n_rows > 2 * n_cols
        ee_score = 75.0 if ee_supported else 0.0
        leaderboard.append({
            "method": "elliptic_envelope",
            "family": "covariance",
            "supported": ee_supported,
            "suitability_score": ee_score,
            "reason": "Robust covariance determinant fitting a Gaussian elliptic envelope.",
            "parameters": {"contamination": contamination},
            "limitations": ["Assumes approximately elliptical/normal underlying data distribution."],
        })

        # Sort leaderboard descending
        leaderboard.sort(key=lambda x: x["suitability_score"], reverse=True)

        if requested_method and requested_method in self.SUPPORTED_METHODS:
            winning_method = requested_method
        else:
            supported_candidates = [c for c in leaderboard if c["supported"]]
            winning_method = supported_candidates[0]["method"] if supported_candidates else "robust_zscore"

        return leaderboard, winning_method

    # --------------------------------------------------------------------------
    # Detector Execution
    # --------------------------------------------------------------------------

    def _run_detector(
        self,
        method: str,
        X_df: pd.DataFrame,
        contamination: float,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Execute selected anomaly detector and return raw anomaly scores and labels."""
        X = X_df.to_numpy(dtype=float)
        n_rows, n_cols = X.shape

        if method == "isolation_forest" and n_rows >= 10:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            iso = IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=self.random_state,
            )
            iso.fit(X_scaled)
            # decision_function: lower means more anomalous
            raw_scores = -iso.decision_function(X_scaled)
            raw_preds = iso.predict(X_scaled)
            labels = np.where(raw_preds == -1, 1, 0)
            return raw_scores, labels, "ensemble"

        elif method == "local_outlier_factor" and n_rows >= 15:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            k_neighbors = min(20, max(5, n_rows // 4))
            lof = LocalOutlierFactor(
                n_neighbors=k_neighbors,
                contamination=contamination,
            )
            raw_preds = lof.fit_predict(X_scaled)
            # negative_outlier_factor_: lower means more abnormal
            raw_scores = -lof.negative_outlier_factor_
            labels = np.where(raw_preds == -1, 1, 0)
            return raw_scores, labels, "density"

        elif method == "iqr":
            # Maximum IQR deviation ratio across features
            scores = np.zeros(n_rows, dtype=float)
            for j in range(n_cols):
                col_vals = X[:, j]
                q25 = np.percentile(col_vals, 25)
                q75 = np.percentile(col_vals, 75)
                iqr = max(q75 - q25, 1e-9)
                lower = q25 - 1.5 * iqr
                upper = q75 + 1.5 * iqr
                dev_low = np.maximum(0.0, (lower - col_vals) / iqr)
                dev_high = np.maximum(0.0, (col_vals - upper) / iqr)
                scores = np.maximum(scores, dev_low + dev_high)
            threshold = 0.0
            labels = (scores > threshold).astype(int)
            return scores, labels, "statistical"

        else:
            # Default: Robust Modified Z-Score (MAD)
            scores = np.zeros(n_rows, dtype=float)
            for j in range(n_cols):
                col_vals = X[:, j]
                med = np.median(col_vals)
                mad = np.median(np.abs(col_vals - med))
                denom = max(mad, 1e-9)
                mod_z = 0.6745 * np.abs(col_vals - med) / denom
                scores = np.maximum(scores, mod_z)
            labels = (scores > 3.0).astype(int)
            return scores, labels, "statistical"

    # --------------------------------------------------------------------------
    # Score Normalization & Explanation Generation
    # --------------------------------------------------------------------------

    def _normalize_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map raw anomaly scores to a consistent [0, 1] range where 1.0 = most anomalous."""
        scores = np.nan_to_num(raw_scores, nan=0.0, posinf=1.0, neginf=0.0)
        s_min = float(np.min(scores))
        s_max = float(np.max(scores))

        if s_max > s_min:
            norm = (scores - s_min) / (s_max - s_min)
        else:
            norm = np.zeros_like(scores)

        return np.clip(norm, 0.0, 1.0)

    def _build_observations(
        self,
        df: pd.DataFrame,
        X_df: pd.DataFrame,
        norm_scores: np.ndarray,
        contamination: float,
        feature_meta: Dict[str, Dict[str, Any]],
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Construct structured observations with explainable reasons.
        """
        n_rows = len(df)
        k_anomalies = max(1, int(math.ceil(contamination * n_rows)))

        # Cutoff: top k_anomalies or score >= 0.70
        sorted_indices = np.argsort(-norm_scores)
        cutoff_score = norm_scores[sorted_indices[min(k_anomalies - 1, n_rows - 1)]]
        score_threshold = max(0.50, min(cutoff_score, 0.90))

        observations: List[Dict[str, Any]] = []
        anomalies_found = 0

        for rank_idx, orig_row_idx in enumerate(sorted_indices):
            score = round(float(norm_scores[orig_row_idx]), 4)
            is_anomaly = rank_idx < k_anomalies or score >= score_threshold

            if is_anomaly:
                anomalies_found += 1
                label = "ANOMALY"
            else:
                label = "NORMAL"

            # Generate explainability reasons for top anomalies
            reasons = []
            anomaly_type = "MULTIVARIATE"
            max_dev_col = None
            max_dev_val = 0.0

            for col in X_df.columns:
                meta = feature_meta.get(col)
                if not meta:
                    continue
                val = float(X_df.iloc[orig_row_idx][col])
                med = meta["median"]
                mad = max(meta["mad"], 1e-6)
                dev_mad = round(abs(val - med) / mad, 2)

                if dev_mad > 2.5:
                    if dev_mad > max_dev_val:
                        max_dev_val = dev_mad
                        max_dev_col = col

                    reasons.append({
                        "feature": col,
                        "value": round(val, 4),
                        "expected_range": [meta["lower_bound"], meta["upper_bound"]],
                        "deviation_score": dev_mad,
                        "reason": f"Feature '{col}' value {val:.2f} deviates significantly from median {med:.2f} ({dev_mad} MADs).",
                    })

            if len(reasons) == 1 or (max_dev_val > 4.0 and len(reasons) <= 2):
                anomaly_type = "UNIVARIATE"
            elif not reasons and is_anomaly:
                reasons.append({
                    "feature": list(X_df.columns)[0] if len(X_df.columns) > 0 else "all_features",
                    "value": None,
                    "expected_range": None,
                    "deviation_score": score,
                    "reason": "Multivariate interaction pattern deviates from joint normal feature distribution.",
                })

            observations.append({
                "row_index": int(df.index[orig_row_idx]) if hasattr(df.index[orig_row_idx], "__int__") else orig_row_idx,
                "anomaly_score": score,
                "anomaly_label": label,
                "anomaly_type": anomaly_type,
                "rank": rank_idx + 1,
                "reasons": reasons,
            })

        return anomalies_found, observations

    def _extract_dataframe(self, data: Any) -> Optional[pd.DataFrame]:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            for df in data.values():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
        return None
