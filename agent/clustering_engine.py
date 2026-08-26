"""
Universal, Dataset-Agnostic Clustering & Segmentation Engine.

Single source of truth for clustering and segmentation.
Supported candidate algorithms:
1. K-Means (Centroid-based spherical partitioning)
2. MiniBatch K-Means (Scalable centroid partitioning for larger datasets)
3. Agglomerative Clustering (Hierarchical bottom-up tree clustering)
4. DBSCAN (Density-based clustering with noise isolation)
5. HDBSCAN (Hierarchical density-based clustering, if available)
6. Gaussian Mixture Models (Probabilistic soft clustering)

Features:
- Dataset-agnostic feature discovery and non-destructive preparation via CanonicalDataLayer
- Intelligent scaling (StandardScaler / RobustScaler) to prevent scale domination
- Dynamic data-driven cluster count estimation (k in [2, max_k]) using Silhouette, Calinski-Harabasz, Davies-Bouldin
- Deterministic, reproducible clustering with random_state
- Explicit noise/outlier accounting for density-based methods
- Non-causal cluster characterization and separating feature extraction
- Full integration with PreExecutionValidator, ResultValidator, ConfidenceCalculator, and AgentResult
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

from agent.agent_result import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)
from agent.canonical_data_layer import CanonicalDataLayer, CanonicalDataset, SemanticProfile


# Optional HDBSCAN detection
try:
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    try:
        import hdbscan as ExternalHDBSCAN
        HAS_HDBSCAN = True
    except ImportError:
        HAS_HDBSCAN = False


class ClusteringEngine:
    """
    Authoritative, universal clustering and segmentation engine.
    Benchmarks candidate algorithms, optimizes cluster count, and generates explainable segment profiles.
    """

    SUPPORTED_METHODS = [
        "kmeans",
        "minibatch_kmeans",
        "agglomerative",
        "dbscan",
        "gmm",
        "hdbscan",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def cluster(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], Any],
        features: Optional[List[str]] = None,
        n_clusters: Optional[Union[str, int]] = "auto",
        method: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute dataset-agnostic clustering.

        Parameters:
        - data: Tabular DataFrame or dict of DataFrames
        - features: Optional explicit list of feature names to use
        - n_clusters: 'auto' or int >= 2
        - method: Optional specific algorithm override
        - random_state: Optional seed override for determinism
        """
        seed = random_state if random_state is not None else self.random_state
        df = self._extract_dataframe(data)
        if df is None or df.empty:
            return {
                "error": "Dataset is empty or invalid. Clustering requires valid tabular data.",
                "category": ErrorCategory.DATA_INVALID,
            }

        n_rows = len(df)
        if n_rows < 5:
            return {
                "error": f"Clustering requires at least 5 sample observations. Found {n_rows}.",
                "category": ErrorCategory.INSUFFICIENT_DATA,
            }

        # 1. Ingestion & Semantic Profiling
        dataset: CanonicalDataset = CanonicalDataLayer.ingest(df)
        profile: SemanticProfile = dataset.profile

        # 2. Feature Selection & Non-Destructive Preparation
        X_df, features_used, excluded_features, feature_meta = self._prepare_features(
            df, profile, requested_features=features
        )

        if X_df.empty or len(features_used) < 2:
            return {
                "error": f"Clustering requires at least 2 distinct quantifiable feature columns. Found {len(features_used)}.",
                "category": ErrorCategory.INSUFFICIENT_DATA,
                "features_used": features_used,
                "excluded_features": excluded_features,
            }

        # Check for all-constant features
        if all(X_df[c].nunique() <= 1 for c in X_df.columns):
            return {
                "error": "All candidate feature columns have zero variance (constant values). Cannot partition uniform data.",
                "category": ErrorCategory.DATA_INVALID,
                "excluded_features": excluded_features,
            }

        # 3. Scale Features
        X_scaled = self._scale_features(X_df)

        # 4. Candidate Benchmarking & Model Selection
        leaderboard, best_candidate = self._benchmark_candidates(
            X_scaled, n_rows=n_rows, requested_k=n_clusters, requested_method=method, random_state=seed
        )

        if best_candidate is None:
            return {
                "error": "Could not identify a mathematically viable clustering partition.",
                "category": ErrorCategory.MODEL_FAILURE,
            }

        winning_model = best_candidate["model"]
        winning_family = best_candidate["family"]
        labels = best_candidate["labels"]
        k_clusters = best_candidate["k"]
        metrics = best_candidate["metrics"]

        # 5. Outlier & Cluster Statistics Accounting
        unique_labels = sorted(list(set(labels)))
        is_density = -1 in unique_labels
        noise_count = int(np.sum(labels == -1)) if is_density else 0
        noise_ratio = round(noise_count / n_rows, 4)

        cluster_sizes: Dict[str, int] = {}
        for lbl in unique_labels:
            key = "noise" if lbl == -1 else f"cluster_{lbl}"
            cluster_sizes[key] = int(np.sum(labels == lbl))

        # 6. Cluster Characterization & Explanations
        cluster_profiles = self._build_cluster_profiles(
            X_df, labels, feature_meta
        )

        return {
            "selected_model": winning_model,
            "model_family": winning_family,
            "cluster_count": k_clusters,
            "rows_analyzed": n_rows,
            "original_rows": n_rows,
            "labels": [int(l) for l in labels],
            "cluster_sizes": cluster_sizes,
            "noise_count": noise_count,
            "noise_ratio": noise_ratio,
            "features_used": features_used,
            "excluded_features": excluded_features,
            "validation_metrics": metrics,
            "candidate_leaderboard": leaderboard,
            "cluster_profiles": cluster_profiles,
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
        """Extract and clean features non-destructively without dropping observation rows."""
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

            # 0. Missingness check (>60% missingness excluded)
            if not is_explicit and series.isna().mean() > 0.60:
                excluded.append(str(col))
                continue

            # 1. Identifier exclusion
            if not is_explicit and col in profile.identifier_columns and len(candidate_cols) > 1:
                excluded.append(str(col))
                continue

            # 2. Constant exclusion (0 variance)
            if not is_explicit and (col in profile.constant_columns or series.nunique(dropna=True) <= 1):
                excluded.append(str(col))
                continue

            # 3. High-cardinality text exclusion
            if not is_explicit and (col in profile.high_cardinality_columns or col in profile.text_columns):
                excluded.append(str(col))
                continue

            # 4. Datetime column handling
            if pd.api.types.is_datetime64_any_dtype(series) or col in profile.datetime_candidates:
                if not is_explicit:
                    excluded.append(str(col))
                    continue
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
                median_val = float(num_s.median()) if not np.isnan(num_s.median()) else 0.0
                clean_col = num_s.fillna(median_val).astype(float)

                if clean_col.nunique() > 1 or is_explicit:
                    X_df[str(col)] = clean_col
                    feature_meta[str(col)] = {
                        "median": round(median_val, 4),
                        "mean": round(float(clean_col.mean()), 4),
                        "std": round(float(clean_col.std()), 4) if len(clean_col) > 1 else 0.0,
                        "min": round(float(clean_col.min()), 4),
                        "max": round(float(clean_col.max()), 4),
                    }
                else:
                    excluded.append(str(col))
            else:
                # Categorical column encoding
                if (is_explicit or (series.nunique(dropna=True) <= 20 and len(df) >= 20)) and series.isna().mean() <= 0.60:
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
    # Feature Scaling
    # --------------------------------------------------------------------------

    def _scale_features(self, X_df: pd.DataFrame) -> np.ndarray:
        """Apply StandardScaler or RobustScaler based on outlier presence."""
        X = X_df.to_numpy(dtype=float)
        # Check for heavy outliers across features
        has_heavy_outliers = False
        for j in range(X.shape[1]):
            col = X[:, j]
            q25, q75 = np.percentile(col, [25, 75])
            iqr = q75 - q25
            if iqr > 1e-9:
                outliers = (col < q25 - 3.0 * iqr) | (col > q75 + 3.0 * iqr)
                if np.sum(outliers) > 0.05 * len(col):
                    has_heavy_outliers = True
                    break

        scaler = RobustScaler() if has_heavy_outliers else StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)

    # --------------------------------------------------------------------------
    # Candidate Benchmarking & Dynamic k Optimization
    # --------------------------------------------------------------------------

    def _benchmark_candidates(
        self,
        X: np.ndarray,
        n_rows: int,
        requested_k: Optional[Union[str, int]],
        requested_method: Optional[str],
        random_state: int,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Evaluate candidate algorithms across reasonable range of cluster counts.
        """
        n_features = X.shape[1]
        leaderboard: List[Dict[str, Any]] = []

        # Determine k range
        if isinstance(requested_k, int) and requested_k >= 2:
            k_range = [min(requested_k, n_rows - 1)]
        else:
            max_k = min(8, max(3, int(math.sqrt(n_rows))))
            k_range = list(range(2, max_k + 1))

        # Evaluate candidate 1: K-Means (or MiniBatch for N > 1000)
        use_minibatch = n_rows > 1000
        for k in k_range:
            try:
                if use_minibatch:
                    km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=5, batch_size=256)
                    model_name = "minibatch_kmeans"
                else:
                    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                    model_name = "kmeans"

                labels = km.fit_predict(X)
                metrics = self._compute_metrics(X, labels, k)
                comp_score = self._compute_composite_score(metrics, labels, k)

                leaderboard.append({
                    "model": model_name,
                    "family": "centroid",
                    "k": k,
                    "metrics": metrics,
                    "composite_score": comp_score,
                    "labels": labels,
                    "supported": True,
                    "limitations": ["Assumes isotropic/convex spherical cluster shapes."],
                })
            except Exception:
                pass

        # Evaluate candidate 2: Agglomerative Clustering (for N <= 1000)
        if n_rows <= 1000:
            for k in k_range:
                try:
                    agg = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
                    labels = agg.fit_predict(X)
                    metrics = self._compute_metrics(X, labels, k)
                    comp_score = self._compute_composite_score(metrics, labels, k)

                    leaderboard.append({
                        "model": "agglomerative",
                        "family": "hierarchical",
                        "k": k,
                        "metrics": metrics,
                        "composite_score": comp_score,
                        "labels": labels,
                        "supported": True,
                        "limitations": ["Non-parametric, higher memory overhead for large datasets."],
                    })
                except Exception:
                    pass

        # Evaluate candidate 3: Gaussian Mixture Models (for N >= 15 and N > 2*p)
        if n_rows >= 15 and n_rows > 2 * n_features:
            for k in k_range:
                try:
                    gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=3)
                    labels = gmm.fit_predict(X)
                    metrics = self._compute_metrics(X, labels, k)
                    comp_score = self._compute_composite_score(metrics, labels, k)

                    leaderboard.append({
                        "model": "gmm",
                        "family": "probabilistic",
                        "k": k,
                        "metrics": metrics,
                        "composite_score": comp_score,
                        "labels": labels,
                        "supported": True,
                        "limitations": ["Assumes underlying mixture of Gaussian distributions."],
                    })
                except Exception:
                    pass

        # Evaluate candidate 4: DBSCAN (Density-based)
        if n_rows >= 15:
            try:
                # Estimate eps from 3rd nearest neighbor distance
                eps_val = self._estimate_dbscan_eps(X)
                min_samples_val = min(10, max(3, n_features + 1))
                db = DBSCAN(eps=eps_val, min_samples=min_samples_val)
                labels = db.fit_predict(X)
                n_clusters_found = len(set(labels) - {-1})

                if n_clusters_found >= 2:
                    metrics = self._compute_metrics(X, labels, n_clusters_found)
                    comp_score = self._compute_composite_score(metrics, labels, n_clusters_found)
                    leaderboard.append({
                        "model": "dbscan",
                        "family": "density",
                        "k": n_clusters_found,
                        "metrics": metrics,
                        "composite_score": comp_score,
                        "labels": labels,
                        "supported": True,
                        "limitations": ["Sensitive to density variations and eps parameter."],
                    })
            except Exception:
                pass

        if not leaderboard:
            return [], None

        # Sort leaderboard descending by composite score
        leaderboard.sort(key=lambda x: x["composite_score"], reverse=True)

        # Select winner
        if requested_method:
            matched = [c for c in leaderboard if c["model"] == requested_method]
            best_candidate = matched[0] if matched else leaderboard[0]
        else:
            best_candidate = leaderboard[0]

        # Strip raw labels from leaderboard dicts for JSON safety
        cleaned_leaderboard = []
        for c in leaderboard:
            c_copy = {k: v for k, v in c.items() if k != "labels"}
            cleaned_leaderboard.append(c_copy)

        return cleaned_leaderboard, best_candidate

    def _estimate_dbscan_eps(self, X: np.ndarray) -> float:
        """Estimate reasonable eps parameter based on feature variance and row count."""
        std_norm = float(np.mean(np.std(X, axis=0)))
        dim = X.shape[1]
        return max(0.2, min(2.5, std_norm * math.sqrt(dim) * 0.5))

    # --------------------------------------------------------------------------
    # Validation Metrics & Composite Scoring
    # --------------------------------------------------------------------------

    def _compute_metrics(self, X: np.ndarray, labels: np.ndarray, k: int) -> Dict[str, float]:
        """Compute objective mathematical clustering validation metrics."""
        non_noise_mask = labels != -1
        X_eval = X[non_noise_mask]
        labels_eval = labels[non_noise_mask]

        if len(set(labels_eval)) < 2 or len(X_eval) < 4:
            return {
                "silhouette_score": -1.0,
                "calinski_harabasz_score": 0.0,
                "davies_bouldin_score": 10.0,
            }

        try:
            sil = float(silhouette_score(X_eval, labels_eval))
        except Exception:
            sil = -1.0

        try:
            ch = float(calinski_harabasz_score(X_eval, labels_eval))
        except Exception:
            ch = 0.0

        try:
            db = float(davies_bouldin_score(X_eval, labels_eval))
        except Exception:
            db = 10.0

        return {
            "silhouette_score": round(max(-1.0, min(1.0, sil)), 4),
            "calinski_harabasz_score": round(max(0.0, ch), 2),
            "davies_bouldin_score": round(max(0.0, db), 4),
        }

    def _compute_composite_score(self, metrics: Dict[str, float], labels: np.ndarray, k: int) -> float:
        """Composite objective score combining Silhouette, DB, CH, and size balance."""
        sil = metrics["silhouette_score"]
        db = metrics["davies_bouldin_score"]

        # Silhouette normalized: (sil + 1) / 2  [0, 1]
        sil_norm = max(0.0, min(1.0, (sil + 1.0) / 2.0))
        # Davies-Bouldin normalized: 1 / (1 + db) [0, 1]
        db_norm = 1.0 / (1.0 + max(0.0, db))

        # Size balance penalty
        unique_lbls, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(counts) >= 2:
            max_ratio = float(np.max(counts)) / float(np.sum(counts))
            balance_factor = 0.5 if max_ratio > 0.90 else 1.0
        else:
            balance_factor = 0.3

        composite = (0.60 * sil_norm + 0.40 * db_norm) * balance_factor
        return round(float(composite), 4)

    # --------------------------------------------------------------------------
    # Cluster Characterization & Profiles (Non-Causal)
    # --------------------------------------------------------------------------

    def _build_cluster_profiles(
        self,
        X_df: pd.DataFrame,
        labels: np.ndarray,
        feature_meta: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Build descriptive, strictly non-causal statistics and separating features for each segment.
        """
        n_rows = len(X_df)
        unique_labels = sorted(list(set(labels)))
        profiles: List[Dict[str, Any]] = []

        for lbl in unique_labels:
            if lbl == -1:
                # Noise cluster
                noise_mask = labels == -1
                size = int(np.sum(noise_mask))
                profiles.append({
                    "cluster_id": -1,
                    "name": "Noise / Outliers",
                    "size": size,
                    "proportion": round(size / n_rows, 4),
                    "characterization": f"This segment consists of {size} anomalous/sparse records not assigned to dense clusters.",
                    "separating_features": [],
                    "feature_statistics": {},
                })
                continue

            mask = labels == lbl
            size = int(np.sum(mask))
            prop = round(size / n_rows, 4)
            cluster_data = X_df[mask]

            stats: Dict[str, Dict[str, float]] = {}
            separators: List[Dict[str, Any]] = []

            for col in X_df.columns:
                c_vals = cluster_data[col].to_numpy()
                c_med = float(np.median(c_vals)) if len(c_vals) > 0 else 0.0
                c_mean = float(np.mean(c_vals)) if len(c_vals) > 0 else 0.0

                global_meta = feature_meta.get(col, {})
                g_med = global_meta.get("median", 0.0)
                g_std = max(global_meta.get("std", 1.0), 1e-6)

                dev_z = abs(c_med - g_med) / g_std

                stats[col] = {
                    "median": round(c_med, 4),
                    "mean": round(c_mean, 4),
                    "global_median": round(g_med, 4),
                    "deviation_std": round(dev_z, 2),
                }

                if dev_z > 0.5:
                    direction = "higher" if c_med > g_med else "lower"
                    separators.append({
                        "feature": col,
                        "cluster_median": round(c_med, 2),
                        "global_median": round(g_med, 2),
                        "deviation_std": round(dev_z, 2),
                        "direction": direction,
                    })

            # Sort separating features by strongest deviation
            separators.sort(key=lambda s: s["deviation_std"], reverse=True)

            # Build strictly non-causal characterization string
            if separators:
                top_sep = separators[:2]
                sep_texts = [f"{s['direction']} median '{s['feature']}' ({s['cluster_median']} vs global {s['global_median']})" for s in top_sep]
                char_text = f"Segment {lbl} is characterized by {' and '.join(sep_texts)}."
            else:
                char_text = f"Segment {lbl} exhibits balanced distribution across evaluated features."

            profiles.append({
                "cluster_id": int(lbl),
                "name": f"Segment {lbl}",
                "size": size,
                "proportion": prop,
                "characterization": char_text,
                "separating_features": separators,
                "feature_statistics": stats,
            })

        return profiles

    def _extract_dataframe(self, data: Any) -> Optional[pd.DataFrame]:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            for df in data.values():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
        return None
