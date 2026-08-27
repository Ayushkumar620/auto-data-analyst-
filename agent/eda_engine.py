"""
Universal, Dataset-Agnostic EDA, Data Profiling & Data Quality Intelligence Engine.

Single source of truth for:
1. Universal Dataset Profiling & Schema Inference
2. Dirty Value & Coercion Intelligence (Currencies, %, Parens, Suffixes)
3. Missing Data Analysis & Severity Classification (Non-destructive)
4. Duplicate Row & Key Analysis
5. Numeric Distribution Analysis & Outlier Detection (Tukey IQR, Robust Z-Score, Non-causal)
6. Categorical Cardinality, Dominance, Rare Categories & Shannon Entropy Analysis
7. Datetime Span, Inferred Frequency & Regularity Analysis
8. Statistical Identifier & Sequence Detection
9. Decomposed Overall Data Quality Scoring [0.0, 1.0] (Completeness, Uniqueness, Validity, Consistency, Usability)
10. Actionable Data Quality Findings & Remediation Recommendations
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from agent.agent_result import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)
from agent.canonical_data_layer import CanonicalDataLayer, CanonicalDataset, SemanticProfile


class EDAEngine:
    """
    Authoritative, universal EDA, data profiling, and data quality intelligence engine.
    Profiles arbitrary tabular data without hardcoded column names or destructive drops.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def analyze(
        self,
        df: Union[pd.DataFrame, Dict[str, Any], Any],
        features: Optional[List[str]] = None,
        include_distributions: bool = True,
        include_quality: bool = True,
        include_relationships: bool = False,
        max_categories: int = 20,
        max_histogram_bins: int = 30,
    ) -> Dict[str, Any]:
        """
        Execute comprehensive dataset profiling and data quality assessment.
        Single source of truth for EDA.

        Parameters:
        - df: Tabular DataFrame or dictionary containing DataFrames
        - features: Optional column subset to profile
        - include_distributions: Include numeric and categorical distribution metrics
        - include_quality: Include decomposed data quality scoring
        - include_relationships: Include high-level correlation/association summary
        - max_categories: Top category frequency display count
        - max_histogram_bins: Maximum histogram bins for numeric features
        """
        raw_df = self._extract_dataframe(df)
        if raw_df is None or raw_df.empty:
            return {
                "error": "Dataset is empty or invalid. EDA requires tabular data.",
                "category": ErrorCategory.DATA_INVALID,
            }

        n_rows, n_cols = raw_df.shape
        if n_rows == 0:
            return {
                "error": "Dataset contains 0 rows. Cannot perform exploratory data analysis on an empty dataset.",
                "category": ErrorCategory.INSUFFICIENT_DATA,
            }
        if n_cols == 0:
            return {
                "error": "Dataset contains 0 columns. Cannot perform exploratory data analysis on an empty schema.",
                "category": ErrorCategory.DATA_INVALID,
            }

        # 1. Canonical Ingestion & Semantic Profiling
        dataset: CanonicalDataset = CanonicalDataLayer.ingest(raw_df)
        sem_profile: SemanticProfile = dataset.profile

        cols_to_profile = [c for c in features if c in raw_df.columns] if features else list(raw_df.columns)
        if not cols_to_profile:
            return {
                "error": "None of the requested columns were found in the dataset.",
                "category": ErrorCategory.DATA_INVALID,
            }

        # 2. Overall Structural Diagnostics
        dup_count = int(raw_df.duplicated().sum())
        dup_pct = round(float((dup_count / n_rows) * 100.0), 2) if n_rows > 0 else 0.0
        empty_rows = int(raw_df.isna().all(axis=1).sum())

        empty_cols: List[str] = []
        constant_cols: List[str] = []
        identifier_cols: List[str] = []
        high_card_cols: List[str] = []
        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        datetime_cols: List[str] = []
        boolean_cols: List[str] = []
        text_cols: List[str] = []
        parseable_numeric_cols: List[str] = []
        parseable_datetime_cols: List[str] = []

        # 3. Column-Level Deep Dive
        columns_profile: Dict[str, Dict[str, Any]] = {}
        dirty_coercion_summary: Dict[str, Any] = {"total_dirty_columns": 0, "columns": {}}

        for col in cols_to_profile:
            col_prof = self._profile_column(
                raw_df,
                col,
                sem_profile,
                max_categories=max_categories,
                include_distributions=include_distributions,
                max_histogram_bins=max_histogram_bins,
            )
            columns_profile[str(col)] = col_prof

            inferred_t = col_prof["inferred_type"]
            if inferred_t == "empty":
                empty_cols.append(str(col))
            if col_prof.get("is_constant") or col_prof.get("constant"):
                constant_cols.append(str(col))
            if col_prof.get("is_identifier"):
                identifier_cols.append(str(col))
            if col_prof.get("is_high_cardinality") or col_prof.get("high_cardinality"):
                high_card_cols.append(str(col))

            if inferred_t == "numeric":
                numeric_cols.append(str(col))
            elif inferred_t == "categorical":
                categorical_cols.append(str(col))
            elif inferred_t == "datetime":
                datetime_cols.append(str(col))
            elif inferred_t == "boolean":
                boolean_cols.append(str(col))
            elif inferred_t == "text":
                text_cols.append(str(col))

            if col_prof.get("parseability", {}).get("numeric"):
                parseable_numeric_cols.append(str(col))
            if col_prof.get("parseability", {}).get("datetime"):
                parseable_datetime_cols.append(str(col))

            if col_prof.get("dirty_coercion"):
                dirty_coercion_summary["total_dirty_columns"] += 1
                dirty_coercion_summary["columns"][str(col)] = col_prof["dirty_coercion"]

        # 4. Missing Data Deep Dive & Severity
        missing_analysis = self._analyze_missingness(raw_df, cols_to_profile, columns_profile)
        sparse_cols = missing_analysis["columns_by_severity"][">90%"] + missing_analysis["columns_by_severity"][">60-90%"] + missing_analysis["columns_by_severity"][">30-60%"]

        # 5. Duplicate Data Analysis
        duplicate_analysis = {
            "exact_duplicate_rows": dup_count,
            "duplicate_percentage": dup_pct,
            "has_duplicates": dup_count > 0,
            "empty_rows_count": empty_rows,
            "unique_rows_count": n_rows - dup_count,
        }

        # 6. Overall Data Quality Score & Component Breakdown
        quality_assessment = self._calculate_data_quality_score(
            n_rows=n_rows,
            n_cols=len(cols_to_profile),
            missing_analysis=missing_analysis,
            duplicate_analysis=duplicate_analysis,
            columns_profile=columns_profile,
            empty_cols=empty_cols,
            constant_cols=constant_cols,
            dirty_summary=dirty_coercion_summary,
        ) if include_quality else {}

        # 7. Actionable Data Quality Findings & Recommendations
        findings = self._generate_quality_findings(
            df=raw_df,
            n_rows=n_rows,
            columns_profile=columns_profile,
            missing_analysis=missing_analysis,
            duplicate_analysis=duplicate_analysis,
            quality_assessment=quality_assessment,
        )

        # 8. Memory Usage
        try:
            mem_bytes = int(raw_df[cols_to_profile].memory_usage(deep=True).sum())
            mem_mb = round(mem_bytes / (1024.0 * 1024.0), 2)
        except Exception:
            mem_bytes = 0
            mem_mb = 0.0

        # Summary structure with all canonical and alias keys
        summary_dict = {
            "row_count": n_rows,
            "column_count": len(cols_to_profile),
            "original_rows": n_rows,
            "original_columns": n_cols,
            "analyzed_rows": n_rows,
            "analyzed_columns": len(cols_to_profile),
            "total_dataset_columns": n_cols,
            "memory_usage_bytes": mem_bytes,
            "memory_usage_mb": mem_mb,
            "duplicate_rows": dup_count,
            "duplicate_percentage": dup_pct,
            "total_missing_cells": missing_analysis.get("total_missing_cells", 0),
            "missing_cell_percentage": missing_analysis.get("overall_missing_percentage", 0.0),
            "empty_rows": empty_rows,
            "empty_columns": empty_cols,
            "constant_columns": constant_cols,
            "identifier_columns": identifier_cols,
            "high_cardinality_columns": high_card_cols,
            "sparse_columns": sparse_cols,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "boolean_columns": boolean_cols,
            "text_columns": text_cols,
            "parseable_numeric_columns": parseable_numeric_cols,
            "parseable_datetime_columns": parseable_datetime_cols,
        }

        result_payload: Dict[str, Any] = {
            "task_type": "eda",
            "summary": summary_dict,
            "statistics": {
                "numeric": {c: columns_profile[c]["numeric_stats"] for c in numeric_cols if columns_profile[c].get("numeric_stats")},
                "categorical": {c: columns_profile[c]["categorical_stats"] for c in categorical_cols if columns_profile[c].get("categorical_stats")},
                "datetime": {c: columns_profile[c]["datetime_stats"] for c in datetime_cols if columns_profile[c].get("datetime_stats")},
                "text": {c: columns_profile[c]["text_stats"] for c in text_cols if columns_profile[c].get("text_stats")},
            },
            "columns": columns_profile,
            "missing_analysis": missing_analysis,
            "duplicate_analysis": duplicate_analysis,
            "dirty_data_analysis": dirty_coercion_summary,
            "data_quality": quality_assessment,
            "findings": findings,
            "warnings": [f["description"] for f in findings if f.get("severity") in ("CRITICAL", "HIGH")],
            "recommendations": [f["suggested_action"] for f in findings if f.get("suggested_action")],
            "assumptions": [
                "Profiling is computed across non-null pairwise observations without destructive row dropping.",
                "Semantic roles are inferred statistically from uniqueness, value distributions, and parseability.",
                "Identified outliers represent extreme values in distribution, not necessarily data errors.",
            ],
            "limitations": [
                "Observational statistics describe data distribution and do not establish causal mechanisms.",
                "Type inference is statistical; domain-specific interpretations may require user configuration.",
            ],
        }

        # Optional high-level relationship summary
        if include_relationships and len(numeric_cols) >= 2:
            try:
                corr_matrix = raw_df[numeric_cols].corr(method="pearson").round(4).to_dict()
                result_payload["relationships"] = {"pearson_correlation": corr_matrix}
            except Exception:
                pass

        return result_payload

    def profile(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], Any],
        selected_columns: Optional[List[str]] = None,
        max_categories: int = 10,
    ) -> Dict[str, Any]:
        """Backward-compatible alias for analyze."""
        return self.analyze(
            df=data,
            features=selected_columns,
            max_categories=max_categories,
        )

    # --------------------------------------------------------------------------
    # Column-Level Profiling
    # --------------------------------------------------------------------------

    def _profile_column(
        self,
        df: pd.DataFrame,
        col: str,
        sem_profile: SemanticProfile,
        max_categories: int = 20,
        include_distributions: bool = True,
        max_histogram_bins: int = 30,
    ) -> Dict[str, Any]:
        """Profile a single column with comprehensive statistical and semantic properties."""
        series = df[col]
        n_total = len(series)
        n_null = int(series.isna().sum())
        n_valid = n_total - n_null
        null_pct = round(float((n_null / n_total) * 100.0), 2) if n_total > 0 else 0.0
        n_uniq = int(series.nunique(dropna=True))
        uniq_ratio = round(float(n_uniq / n_valid), 4) if n_valid > 0 else 0.0
        uniq_pct = round(float(uniq_ratio * 100.0), 2)
        dup_count = max(0, n_valid - n_uniq)

        # Sample values (JSON-serializable)
        sample_vals: List[Any] = []
        for val in series.dropna().iloc[:5]:
            if pd.api.types.is_number(val) and not pd.isna(val):
                sample_vals.append(float(val) if not (math.isnan(val) or math.isinf(val)) else None)
            else:
                sample_vals.append(str(val))

        is_empty = n_valid == 0
        is_constant = n_uniq <= 1 and not is_empty

        # Check Datetime candidate FIRST before identifier
        is_dt_candidate = False
        dt_series: Optional[pd.Series] = None
        if not is_empty and not is_constant and (pd.api.types.is_datetime64_any_dtype(series) or col in sem_profile.datetime_candidates):
            dt_series = CanonicalDataLayer.coerce_datetime_series(series)
            if dt_series.notna().mean() >= 0.60:
                is_dt_candidate = True
        elif not is_empty and not is_constant and series.dtype == object:
            sample_str = series.dropna().astype(str).iloc[:30]
            if any("/" in s or "-" in s or ":" in s or "T" in s for s in sample_str):
                try:
                    dt_test = pd.to_datetime(sample_str, errors="coerce")
                    if dt_test.notna().mean() >= 0.70:
                        dt_series = CanonicalDataLayer.coerce_datetime_series(series)
                        if dt_series.notna().mean() >= 0.60:
                            is_dt_candidate = True
                except Exception:
                    pass

        # Statistical Identifier Detection
        id_likelihood, id_reason = self._detect_identifier_likelihood(series, n_total, n_valid, n_uniq, col, sem_profile)
        is_identifier = (not is_dt_candidate) and id_likelihood >= 0.75 and not is_constant

        # Check Numeric candidate & Dirty Coercion
        is_num_candidate = False
        num_series: Optional[pd.Series] = None
        dirty_info: Optional[Dict[str, Any]] = None

        if not is_dt_candidate and not is_empty:
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                is_num_candidate = True
                num_series = series.astype(float)
            else:
                # Attempt CanonicalDataLayer coercion (handles $, %, €, commas, (1,200), 1.5k, 2.4M)
                coerced = CanonicalDataLayer.coerce_numeric_series(series)
                valid_coerced = int(coerced.notna().sum())
                coercion_rate = valid_coerced / n_valid if n_valid > 0 else 0.0

                if coercion_rate >= 0.50 and not is_identifier:
                    is_num_candidate = True
                    num_series = coerced
                    dirty_count = int((series.notna() & coerced.notna() & (series.astype(str) != coerced.astype(str))).sum())
                    if dirty_count > 0 or coercion_rate < 1.0:
                        dirty_info = {
                            "original_dtype": str(series.dtype),
                            "detected_numeric": True,
                            "coercion_success_rate": round(float(coercion_rate), 4),
                            "values_cleaned_count": dirty_count,
                            "coercion_failure_count": n_valid - valid_coerced,
                            "remaining_invalid_count": n_valid - valid_coerced,
                            "transformation_applied": "currency_symbol_comma_percentage_and_bracket_stripping",
                        }

        # Near constant check (top category dominance >= 95%)
        is_near_constant = False
        if not is_empty and n_uniq > 1:
            top_val_count = int(series.dropna().value_counts().iloc[0])
            if (top_val_count / n_valid) >= 0.95:
                is_near_constant = True

        # Determine Inferred Type & Semantic Role
        if is_empty:
            inferred_type = "empty"
            semantic_role = "unusable"
        elif is_constant:
            inferred_type = "constant"
            semantic_role = "unusable"
        elif is_identifier:
            inferred_type = "identifier"
            semantic_role = "key"
        elif is_dt_candidate:
            inferred_type = "datetime"
            semantic_role = "temporal"
        elif is_num_candidate:
            inferred_type = "numeric"
            semantic_role = "measure"
        elif pd.api.types.is_bool_dtype(series) or (n_uniq == 2 and set(series.dropna().unique()).issubset({True, False, 0, 1, "0", "1", "true", "false", "True", "False"})):
            inferred_type = "boolean"
            semantic_role = "dimension"
        elif n_uniq > 50 and (n_uniq / n_valid > 0.60 if n_valid > 0 else False):
            inferred_type = "text"
            semantic_role = "attribute"
        else:
            inferred_type = "categorical"
            semantic_role = "dimension"

        is_high_card = bool(inferred_type in ("categorical", "text") and n_uniq > 50)

        # Build Specialized Statistics
        num_stats: Optional[Dict[str, Any]] = None
        cat_stats: Optional[Dict[str, Any]] = None
        dt_stats: Optional[Dict[str, Any]] = None
        txt_stats: Optional[Dict[str, Any]] = None

        if inferred_type == "numeric" and num_series is not None and include_distributions:
            num_stats = self._calculate_numeric_stats(num_series, max_histogram_bins=max_histogram_bins)
        elif inferred_type in ("categorical", "boolean", "constant") and include_distributions:
            cat_stats = self._calculate_categorical_stats(series, n_valid, n_uniq, max_categories)
        elif inferred_type == "text" and include_distributions:
            cat_stats = self._calculate_categorical_stats(series, n_valid, n_uniq, max_categories)
            txt_stats = self._calculate_text_stats(series)
        elif inferred_type == "datetime" and dt_series is not None and include_distributions:
            dt_stats = self._calculate_datetime_stats(dt_series)

        # Calculate Column-Level Quality Score [0.0, 1.0]
        col_comp_score = max(0.0, 1.0 - (null_pct / 100.0))
        col_valid_score = dirty_info["coercion_success_rate"] if dirty_info else 1.0
        col_distinct_score = 0.20 if is_constant else (0.50 if is_near_constant else 1.0)
        col_quality_score = round(0.40 * col_comp_score + 0.30 * col_valid_score + 0.30 * col_distinct_score, 4)

        return {
            "name": str(col),
            "column_name": str(col),
            "original_dtype": str(series.dtype),
            "inferred_type": inferred_type,
            "inferred_role": semantic_role,
            "semantic_role": semantic_role,
            "total_count": n_total,
            "row_count": n_total,
            "non_null_count": n_valid,
            "null_count": n_null,
            "null_percentage": null_pct,
            "missing_percentage": null_pct,
            "unique_count": n_uniq,
            "unique_percentage": uniq_pct,
            "uniqueness_ratio": uniq_ratio,
            "duplicate_count": dup_count,
            "is_constant": is_constant,
            "constant": is_constant,
            "is_near_constant": is_near_constant,
            "near_constant": is_near_constant,
            "is_identifier": is_identifier,
            "identifier_likelihood": round(float(id_likelihood), 2),
            "identifier_reason": id_reason if is_identifier else None,
            "is_high_cardinality": is_high_card,
            "high_cardinality": is_high_card,
            "sample_values": sample_vals,
            "dirty_coercion": dirty_info,
            "parseability": {
                "numeric": is_num_candidate,
                "datetime": is_dt_candidate,
            },
            "transformation": dirty_info.get("transformation_applied") if dirty_info else None,
            "quality_score": col_quality_score,
            "numeric_stats": num_stats,
            "categorical_stats": cat_stats,
            "datetime_stats": dt_stats,
            "text_stats": txt_stats,
        }

    # --------------------------------------------------------------------------
    # Specialized Statistical Calculators
    # --------------------------------------------------------------------------

    def _calculate_numeric_stats(self, s: pd.Series, max_histogram_bins: int = 30) -> Dict[str, Any]:
        """Compute exhaustive, robust non-causal numeric distribution statistics."""
        v = s.dropna().to_numpy(dtype=float)
        n = len(v)
        if n == 0:
            return {}

        v_mean = float(np.mean(v))
        v_std = float(np.std(v, ddof=1)) if n > 1 else 0.0
        v_var = float(np.var(v, ddof=1)) if n > 1 else 0.0
        v_min = float(np.min(v))
        v_max = float(np.max(v))

        q25 = float(np.percentile(v, 25))
        q50 = float(np.percentile(v, 50))  # median
        q75 = float(np.percentile(v, 75))
        iqr = float(q75 - q25)

        # Median Absolute Deviation (MAD)
        mad = float(np.median(np.abs(v - q50)))

        # Skewness and Kurtosis
        skew_val = float(stats.skew(v)) if n >= 3 and v_std > 1e-9 else 0.0
        kurt_val = float(stats.kurtosis(v)) if n >= 4 and v_std > 1e-9 else 0.0

        if math.isnan(skew_val) or math.isinf(skew_val):
            skew_val = 0.0
        if math.isnan(kurt_val) or math.isinf(kurt_val):
            kurt_val = 0.0

        # Distribution shape classification
        if abs(skew_val) < 0.5 and abs(kurt_val) < 1.0:
            dist_shape = "approximately_normal" if n >= 30 else "symmetric"
        elif skew_val >= 1.0:
            dist_shape = "right_skewed"
        elif skew_val <= -1.0:
            dist_shape = "left_skewed"
        elif kurt_val >= 2.0:
            dist_shape = "heavy_tailed"
        else:
            dist_shape = "symmetric"

        # Sign counts
        zero_cnt = int(np.sum(v == 0.0))
        neg_cnt = int(np.sum(v < 0.0))
        pos_cnt = int(np.sum(v > 0.0))
        if zero_cnt / n >= 0.30 and n >= 20:
            dist_shape = "zero_inflated"

        # Percentiles
        pcts = {
            "1%": round(float(np.percentile(v, 1)), 4),
            "5%": round(float(np.percentile(v, 5)), 4),
            "10%": round(float(np.percentile(v, 10)), 4),
            "25%": round(q25, 4),
            "50%": round(q50, 4),
            "75%": round(q75, 4),
            "90%": round(float(np.percentile(v, 90)), 4),
            "95%": round(float(np.percentile(v, 95)), 4),
            "99%": round(float(np.percentile(v, 99)), 4),
        }

        # Outlier Detection (Tukey IQR fences)
        lower_fence = q25 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr
        outlier_mask = (v < lower_fence) | (v > upper_fence)
        outlier_count = int(np.sum(outlier_mask))
        outlier_pct = round(float((outlier_count / n) * 100.0), 2)
        sample_outliers = [float(x) for x in v[outlier_mask][:5]]

        # Robust Z-score (MAD-based)
        if mad > 1e-9:
            robust_z = 0.6745 * (v - q50) / mad
            mad_outliers_cnt = int(np.sum(np.abs(robust_z) > 3.5))
        else:
            mad_outliers_cnt = 0

        # Histogram calculation
        n_bins = min(max_histogram_bins, max(5, int(np.sqrt(n))))
        counts, bin_edges = np.histogram(v, bins=n_bins)
        hist_bins = [
            {"bin_start": round(float(bin_edges[i]), 4), "bin_end": round(float(bin_edges[i + 1]), 4), "count": int(counts[i])}
            for i in range(len(counts))
        ]

        return {
            "count": n,
            "min": round(v_min, 4),
            "max": round(v_max, 4),
            "mean": round(v_mean, 4),
            "median": round(q50, 4),
            "std": round(v_std, 4),
            "variance": round(v_var, 4),
            "q1": round(q25, 4),
            "q3": round(q75, 4),
            "q25": round(q25, 4),
            "q75": round(q75, 4),
            "iqr": round(iqr, 4),
            "IQR": round(iqr, 4),
            "mad": round(mad, 4),
            "skewness": round(skew_val, 4),
            "kurtosis": round(kurt_val, 4),
            "distribution_shape": dist_shape,
            "zero_count": zero_cnt,
            "negative_count": neg_cnt,
            "positive_count": pos_cnt,
            "percentiles": pcts,
            "histogram": hist_bins,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_pct,
            "lower_bound": round(lower_fence, 4),
            "upper_bound": round(upper_fence, 4),
            "method": "tukey_iqr_1.5",
            "outliers": {
                "count": outlier_count,
                "percentage": outlier_pct,
                "lower_bound": round(lower_fence, 4),
                "upper_bound": round(upper_fence, 4),
                "sample_outliers": sample_outliers,
                "method": "tukey_iqr_1.5",
            },
            "robust_zscore_outliers": {
                "count": mad_outliers_cnt,
                "percentage": round(float((mad_outliers_cnt / n) * 100.0), 2),
                "threshold": 3.5,
            },
        }

    def _calculate_categorical_stats(
        self,
        s: pd.Series,
        n_valid: int,
        n_uniq: int,
        max_categories: int = 20,
    ) -> Dict[str, Any]:
        """Compute frequency, dominance, rare categories, and Shannon entropy for categorical series."""
        val_counts = s.dropna().astype(str).value_counts()
        cardinality_ratio = round(float(n_uniq / n_valid), 4) if n_valid > 0 else 0.0

        top_cats: List[Dict[str, Any]] = []
        top_val_dict: Dict[str, int] = {}
        for cat_name, count in val_counts.iloc[:max_categories].items():
            pct = round(float((count / n_valid) * 100.0), 2) if n_valid > 0 else 0.0
            top_cats.append({
                "value": str(cat_name),
                "count": int(count),
                "percentage": pct,
            })
            top_val_dict[str(cat_name)] = int(count)

        dominant_pct = top_cats[0]["percentage"] if top_cats else 0.0
        is_imbalanced = dominant_pct >= 85.0 and n_uniq > 1

        # Rare categories (count == 1 or < 1%)
        rare_count = int((val_counts <= max(1, int(n_valid * 0.01))).sum())

        # Shannon entropy in bits: -sum(p * log2(p))
        if n_valid > 0:
            probs = val_counts.to_numpy(dtype=float) / float(n_valid)
            entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        else:
            entropy = 0.0

        return {
            "unique_count": n_uniq,
            "cardinality_ratio": cardinality_ratio,
            "top_categories": top_cats,
            "top_values": [c["value"] for c in top_cats],
            "top_value_counts": top_val_dict,
            "dominant_category_percentage": round(dominant_pct, 2),
            "is_imbalanced": is_imbalanced,
            "rare_category_count": rare_count,
            "rare_categories_count": rare_count,
            "category_distribution": {c["value"]: c["percentage"] for c in top_cats},
            "entropy": round(float(entropy), 4),
        }

    def _calculate_text_stats(self, s: pd.Series) -> Dict[str, Any]:
        """Compute text length properties for free-form string columns."""
        str_series = s.dropna().astype(str)
        if str_series.empty:
            return {}
        lengths = str_series.str.len().to_numpy(dtype=float)
        return {
            "average_length": round(float(np.mean(lengths)), 2),
            "median_length": round(float(np.median(lengths)), 2),
            "min_length": int(np.min(lengths)),
            "max_length": int(np.max(lengths)),
            "empty_string_count": int((str_series.str.strip() == "").sum()),
            "unique_ratio": round(float(str_series.nunique() / len(str_series)), 4),
        }

    def _calculate_datetime_stats(self, dt_s: pd.Series) -> Dict[str, Any]:
        """Compute temporal span, timestamps uniqueness, and sequence regularity."""
        valid_dt = dt_s.dropna().sort_values()
        n_valid = len(valid_dt)
        if n_valid == 0:
            return {}

        min_dt = valid_dt.min()
        max_dt = valid_dt.max()
        span_days = round(float((max_dt - min_dt).total_seconds() / 86400.0), 2)
        n_uniq_dt = int(valid_dt.nunique())
        dup_timestamps = n_valid - n_uniq_dt

        # Infer frequency
        inferred_freq = None
        try:
            if n_uniq_dt >= 3:
                inferred_freq = pd.infer_freq(valid_dt.drop_duplicates())
        except Exception:
            inferred_freq = None

        return {
            "min_date": min_dt.isoformat(),
            "max_date": max_dt.isoformat(),
            "min_timestamp": min_dt.isoformat(),
            "max_timestamp": max_dt.isoformat(),
            "date_span": f"{span_days} days",
            "date_span_days": span_days,
            "unique_timestamps": n_uniq_dt,
            "duplicate_timestamps": dup_timestamps,
            "duplicate_timestamp_count": dup_timestamps,
            "missing_timestamp_count": 0,
            "inferred_frequency": str(inferred_freq) if inferred_freq else "irregular",
            "timezone": str(valid_dt.dt.tz) if hasattr(valid_dt.dt, "tz") and valid_dt.dt.tz else None,
        }

    # --------------------------------------------------------------------------
    # Statistical Identifier Detection
    # --------------------------------------------------------------------------

    def _detect_identifier_likelihood(
        self,
        s: pd.Series,
        n_total: int,
        n_valid: int,
        n_uniq: int,
        col_name: str,
        sem_profile: SemanticProfile,
    ) -> Tuple[float, str]:
        """Determine statistical likelihood that a column is a surrogate key / identifier."""
        if n_valid < 3:
            return 0.0, "Insufficient sample size"

        uniq_ratio = n_uniq / n_valid

        # 1. 100% unique string or sequence
        if uniq_ratio >= 0.999:
            # Check if sequential integer
            if pd.api.types.is_integer_dtype(s) or (pd.api.types.is_numeric_dtype(s) and (s.dropna() % 1 == 0).all()):
                diffs = s.dropna().diff().dropna()
                if (diffs == 1).mean() > 0.90:
                    return 0.98, "Sequential integer index"

            # String examination
            sample_str = s.dropna().astype(str).iloc[:20]

            # 1. String with UUID structure (e.g. 8-4-4-4-12)
            if any(len(x) in (32, 36) and "-" in x for x in sample_str):
                return 0.99, "UUID format matched"

            # 2. Free text with multiple spaces or paragraph text
            has_spaces = (sample_str.str.count(" ") >= 2).mean() > 0.40
            avg_len = sum(len(x) for x in sample_str) / max(1, len(sample_str))
            if has_spaces or avg_len > 60:
                return 0.10, "Free-form text attribute"

            if col_name in sem_profile.identifier_columns:
                return 0.95, "Identified as key by semantic profiler"
            if not pd.api.types.is_numeric_dtype(s):
                return 0.90, "100% unique non-numeric strings"

        if col_name in sem_profile.identifier_columns:
            return 0.85, "Profiled as identifier"

        return 0.0, "Distributed values"

    # --------------------------------------------------------------------------
    # Missingness & Severity Analysis
    # --------------------------------------------------------------------------

    def _analyze_missingness(
        self,
        df: pd.DataFrame,
        cols: List[str],
        col_profiles: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Classify missingness across standard severity bands without global row drops."""
        n_rows = len(df)
        total_cells = n_rows * len(cols) if n_rows > 0 else 0
        total_missing = int(df[cols].isna().sum().sum())
        missing_pct = round(float((total_missing / total_cells) * 100.0), 2) if total_cells > 0 else 0.0
        completeness_score = round(max(0.0, min(1.0, 1.0 - (total_missing / total_cells))), 4) if total_cells > 0 else 1.0

        bands: Dict[str, List[str]] = {
            "0%": [],
            ">0-10%": [],
            ">10-30%": [],
            ">30-60%": [],
            ">60-90%": [],
            ">90%": [],
        }

        col_missing_summary: Dict[str, Dict[str, Any]] = {}
        for c in cols:
            p = col_profiles[c]
            null_pct = p["null_percentage"]
            null_cnt = p["null_count"]
            col_missing_summary[c] = {
                "missing_count": null_cnt,
                "missing_percentage": null_pct,
                "completeness_score": round(1.0 - (null_pct / 100.0), 4),
            }

            if null_pct == 0.0:
                bands["0%"].append(c)
            elif null_pct <= 10.0:
                bands[">0-10%"].append(c)
            elif null_pct <= 30.0:
                bands[">10-30%"].append(c)
            elif null_pct <= 60.0:
                bands[">30-60%"].append(c)
            elif null_pct <= 90.0:
                bands[">60-90%"].append(c)
            else:
                bands[">90%"].append(c)

        # High missingness rows (>50% of columns missing)
        high_missing_rows = int((df[cols].isna().mean(axis=1) > 0.50).sum())

        return {
            "total_missing_cells": total_missing,
            "overall_missing_percentage": missing_pct,
            "completeness_score": completeness_score,
            "columns_by_severity": bands,
            "columns_summary": col_missing_summary,
            "high_missing_rows_count": high_missing_rows,
            "complete_columns_count": len(bands["0%"]),
            "sparse_columns_count": len(bands[">60-90%"]) + len(bands[">90%"]),
        }

    # --------------------------------------------------------------------------
    # Decomposed Data Quality Scoring
    # --------------------------------------------------------------------------

    def _calculate_data_quality_score(
        self,
        n_rows: int,
        n_cols: int,
        missing_analysis: Dict[str, Any],
        duplicate_analysis: Dict[str, Any],
        columns_profile: Dict[str, Dict[str, Any]],
        empty_cols: List[str],
        constant_cols: List[str],
        dirty_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute an objective, bounded [0.0, 1.0] data quality score with decomposable components."""
        # 1. Completeness Score [0, 1]
        comp_score = missing_analysis.get("completeness_score", 1.0)

        # 2. Validity / Cleanliness Score [0, 1]
        dirty_cols_cnt = dirty_summary.get("total_dirty_columns", 0)
        validity_score = max(0.0, 1.0 - (dirty_cols_cnt / max(1, n_cols) * 0.5))

        # 3. Uniqueness Score [0, 1]
        dup_pct = duplicate_analysis.get("duplicate_percentage", 0.0)
        uniqueness_score = max(0.0, 1.0 - (dup_pct / 100.0))

        # 4. Consistency / Non-Triviality Score [0, 1]
        trivial_cols = len(empty_cols) + len(constant_cols)
        consistency_score = max(0.0, 1.0 - (trivial_cols / max(1, n_cols)))

        # 5. Usability / Structure Score [0, 1]
        unusable_roles = sum(1 for p in columns_profile.values() if p.get("semantic_role") in ("unusable", "key"))
        structural_score = max(0.0, 1.0 - (unusable_roles / max(1, n_cols) * 0.5))

        # Weighted Composite Score
        overall = (
            0.35 * comp_score
            + 0.25 * validity_score
            + 0.20 * uniqueness_score
            + 0.10 * consistency_score
            + 0.10 * structural_score
        )
        overall = max(0.0, min(1.0, float(overall)))

        if overall >= 0.90:
            rating = "EXCELLENT"
        elif overall >= 0.75:
            rating = "GOOD"
        elif overall >= 0.60:
            rating = "MODERATE"
        elif overall >= 0.40:
            rating = "POOR"
        else:
            rating = "CRITICAL"

        return {
            "quality_score": round(overall, 4),
            "overall_quality_score": round(overall, 4),
            "completeness_score": round(float(comp_score), 4),
            "uniqueness_score": round(float(uniqueness_score), 4),
            "validity_score": round(float(validity_score), 4),
            "consistency_score": round(float(consistency_score), 4),
            "quality_rating": rating,
            "components": {
                "completeness": round(float(comp_score), 4),
                "validity": round(float(validity_score), 4),
                "uniqueness": round(float(uniqueness_score), 4),
                "consistency": round(float(consistency_score), 4),
                "structural_usability": round(float(structural_score), 4),
            },
            "explanations": {
                "completeness": f"{round(float(comp_score)*100, 1)}% complete (non-missing values).",
                "validity": f"{round(float(validity_score)*100, 1)}% valid formatting across columns.",
                "uniqueness": f"{round(float(uniqueness_score)*100, 1)}% unique non-duplicate observations.",
                "consistency": f"{round(float(consistency_score)*100, 1)}% informative non-constant columns.",
                "structural_usability": f"{round(float(structural_score)*100, 1)}% usable analytical features.",
            },
        }

    # --------------------------------------------------------------------------
    # Actionable Findings & Recommendations
    # --------------------------------------------------------------------------

    def _generate_quality_findings(
        self,
        df: pd.DataFrame,
        n_rows: int,
        columns_profile: Dict[str, Dict[str, Any]],
        missing_analysis: Dict[str, Any],
        duplicate_analysis: Dict[str, Any],
        quality_assessment: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate structured data quality findings with actionable remediation steps."""
        findings: List[Dict[str, Any]] = []

        # 1. Missing Data Findings
        sparse_cols = missing_analysis["columns_by_severity"][">90%"] + missing_analysis["columns_by_severity"][">60-90%"]
        if sparse_cols:
            findings.append({
                "code": "HIGH_MISSINGNESS",
                "category": "MISSING_DATA",
                "severity": "HIGH",
                "column": sparse_cols,
                "affected_columns": sparse_cols,
                "message": f"{len(sparse_cols)} column(s) contain over 60% missing values.",
                "description": f"{len(sparse_cols)} column(s) contain over 60% missing values.",
                "evidence": {"sparse_columns": sparse_cols},
                "suggested_action": "Consider imputing or dropping sparse columns before building supervised learning models.",
                "recommended_action": "Consider imputing or dropping sparse columns before building supervised learning models.",
            })

        # 2. Duplicate Rows Finding
        dup_cnt = duplicate_analysis.get("exact_duplicate_rows", 0)
        if dup_cnt > 0:
            findings.append({
                "code": "DUPLICATE_ROWS",
                "category": "DUPLICATES",
                "severity": "MEDIUM" if duplicate_analysis["duplicate_percentage"] < 10.0 else "HIGH",
                "column": None,
                "affected_columns": [],
                "message": f"Found {dup_cnt} exact duplicate rows ({duplicate_analysis['duplicate_percentage']}%) in the dataset.",
                "description": f"Found {dup_cnt} exact duplicate rows ({duplicate_analysis['duplicate_percentage']}%) in the dataset.",
                "evidence": {"duplicate_count": dup_cnt, "duplicate_percentage": duplicate_analysis["duplicate_percentage"]},
                "suggested_action": "Verify if repeated observations reflect legitimate multiple occurrences or data ingestion artifacts.",
                "recommended_action": "Verify if repeated observations reflect legitimate multiple occurrences or data ingestion artifacts.",
            })

        # 3. Constant Columns Finding
        const_cols = [c for c, p in columns_profile.items() if p.get("is_constant") or p.get("constant")]
        if const_cols:
            findings.append({
                "code": "CONSTANT_COLUMN",
                "category": "CONSTANT_COLUMN",
                "severity": "MEDIUM",
                "column": const_cols,
                "affected_columns": const_cols,
                "message": f"{len(const_cols)} column(s) have zero variance (single unique value).",
                "description": f"{len(const_cols)} column(s) have zero variance (single unique value).",
                "evidence": {"constant_columns": const_cols},
                "suggested_action": "Exclude zero-variance columns from downstream statistical analysis and predictive modeling.",
                "recommended_action": "Exclude zero-variance columns from downstream statistical analysis and predictive modeling.",
            })

        # 4. Dirty Data Finding
        dirty_cols = [c for c, p in columns_profile.items() if p.get("dirty_coercion")]
        for c in dirty_cols:
            dc = columns_profile[c]["dirty_coercion"]
            findings.append({
                "code": "DIRTY_NUMERIC_STRINGS",
                "category": "INVALID_VALUES",
                "severity": "LOW" if dc["coercion_success_rate"] >= 0.90 else "MEDIUM",
                "column": c,
                "affected_columns": [c],
                "message": f"Column '{c}' contains string-formatted numbers requiring cleaning (cleaned: {dc['values_cleaned_count']}).",
                "description": f"Column '{c}' contains string-formatted numbers requiring cleaning (cleaned: {dc['values_cleaned_count']}).",
                "evidence": dc,
                "suggested_action": "Apply automatic CanonicalDataLayer numeric coercion during preprocessing.",
                "recommended_action": "Apply automatic CanonicalDataLayer numeric coercion during preprocessing.",
            })

        # 5. Outliers Finding
        for c, p in columns_profile.items():
            num_st = p.get("numeric_stats")
            if num_st and num_st.get("outliers") and num_st["outliers"]["count"] > 0:
                out_info = num_st["outliers"]
                if out_info["percentage"] >= 5.0:
                    findings.append({
                        "code": "OUTLIERS_DETECTED",
                        "category": "OUTLIER",
                        "severity": "LOW",
                        "column": c,
                        "affected_columns": [c],
                        "message": f"Column '{c}' contains {out_info['count']} statistically extreme values ({out_info['percentage']}%).",
                        "description": f"Column '{c}' contains {out_info['count']} statistically extreme values ({out_info['percentage']}%).",
                        "evidence": {"outlier_count": out_info["count"], "lower_bound": out_info["lower_bound"], "upper_bound": out_info["upper_bound"]},
                        "suggested_action": "Inspect distribution skewness and consider robust scaling (e.g. RobustScaler) or rank-based transforms.",
                        "recommended_action": "Inspect distribution skewness and consider robust scaling (e.g. RobustScaler) or rank-based transforms.",
                    })

        # 6. Imbalance Finding
        for c, p in columns_profile.items():
            cat_st = p.get("categorical_stats")
            if cat_st and cat_st.get("is_imbalanced"):
                findings.append({
                    "code": "CATEGORY_IMBALANCE",
                    "category": "IMBALANCE",
                    "severity": "LOW",
                    "column": c,
                    "affected_columns": [c],
                    "message": f"Column '{c}' is heavily imbalanced (dominant category represents {cat_st['dominant_category_percentage']}% of records).",
                    "description": f"Column '{c}' is heavily imbalanced (dominant category represents {cat_st['dominant_category_percentage']}% of records).",
                    "evidence": {"dominant_percentage": cat_st["dominant_category_percentage"]},
                    "suggested_action": "Consider stratified sampling or class rebalancing if using this column as a classification target.",
                    "recommended_action": "Consider stratified sampling or class rebalancing if using this column as a classification target.",
                })

        return findings

    def _extract_dataframe(self, data: Any) -> Optional[pd.DataFrame]:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            for df in data.values():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
        return None