"""
Universal, Dataset-Agnostic Statistical Relationship & Dependency Analysis Engine.

Single source of truth for:
1. Numeric <-> Numeric dependency (Pearson, Spearman, Kendall tau, Outlier Sensitivity)
2. Numeric <-> Categorical dependency (Point-biserial, One-Way ANOVA, Kruskal-Wallis, Eta-squared effect size)
3. Categorical <-> Categorical association (Chi-Square Test of Independence, Cramer's V, Fisher's Exact Test)
4. Time-aware Trend and Chronological Association
5. Multiple Testing Correction (Benjamini-Hochberg False Discovery Rate)
6. Pairwise Non-Destructive Missing Data Masking (Zero global row loss)
7. Strictly Non-Causal Grounding and Effect Size vs Significance Disambiguation
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


class StatisticalAnalysisEngine:
    """
    Authoritative, universal statistical relationship and dependency analysis engine.
    Discovers, measures, tests, and ranks bivariate and multivariate feature associations.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_pairs: int = 250,
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.max_pairs = max_pairs
        self.random_state = random_state

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def analyze(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], Any],
        features: Optional[List[str]] = None,
        target: Optional[str] = None,
        alpha: Optional[float] = None,
        max_pairs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute comprehensive dataset-agnostic statistical relationship analysis.

        Parameters:
        - data: Tabular DataFrame or dictionary of DataFrames
        - features: Optional subset of feature column names to evaluate
        - target: Optional specific column to focus relationships against
        - alpha: Significance threshold (default 0.05)
        - max_pairs: Combinatorial evaluation limit
        """
        sig_alpha = alpha if alpha is not None else self.alpha
        pair_limit = max_pairs if max_pairs is not None else self.max_pairs

        df = self._extract_dataframe(data)
        if df is None or df.empty:
            return {
                "error": "Dataset is empty or invalid. Statistical analysis requires tabular data.",
                "category": ErrorCategory.DATA_INVALID,
            }

        n_rows = len(df)
        if n_rows < 3:
            return {
                "error": f"Statistical relationship analysis requires at least 3 sample observations. Found {n_rows}.",
                "category": ErrorCategory.INSUFFICIENT_DATA,
            }

        # 1. Ingestion & Semantic Profiling
        dataset: CanonicalDataset = CanonicalDataLayer.ingest(df)
        profile: SemanticProfile = dataset.profile

        # 2. Feature Discovery & Partitioning
        numeric_cols, cat_cols, dt_cols, excluded_cols = self._discover_features(
            df, profile, requested_features=features, target=target
        )

        all_usable = numeric_cols + cat_cols + dt_cols
        if len(all_usable) < 2:
            return {
                "error": f"Statistical relationship analysis requires at least 2 distinct feature columns. Found {len(all_usable)}.",
                "category": ErrorCategory.INSUFFICIENT_DATA,
                "features_used": all_usable,
                "excluded_features": excluded_cols,
            }

        # Check for all-constant features
        if all(df[c].nunique(dropna=True) <= 1 for c in all_usable):
            return {
                "error": "All candidate feature columns have zero variance (constant values). Cannot compute statistical relationships across invariant columns.",
                "category": ErrorCategory.DATA_INVALID,
                "excluded_features": excluded_cols,
            }

        # 3. Generate Candidate Pairs (target-focused or all-pairwise)
        candidate_pairs, is_limited = self._generate_candidate_pairs(
            numeric_cols, cat_cols, dt_cols, target=target, max_pairs=pair_limit
        )

        if not candidate_pairs:
            return {
                "error": "No valid variable pairs could be formed for statistical relationship analysis.",
                "category": ErrorCategory.INSUFFICIENT_DATA,
                "excluded_features": excluded_cols,
            }

        # 4. Evaluate Each Variable Pair Non-Destructively
        raw_relationships: List[Dict[str, Any]] = []
        for feat_x, feat_y, pair_type in candidate_pairs:
            rel = self._evaluate_pair(df, feat_x, feat_y, pair_type, sig_alpha)
            if rel is not None:
                raw_relationships.append(rel)

        if not raw_relationships:
            return {
                "error": "Could not compute statistically valid relationship metrics for candidate variable pairs.",
                "category": ErrorCategory.MODEL_FAILURE,
                "excluded_features": excluded_cols,
            }

        # 5. Multiple Testing Correction (Benjamini-Hochberg FDR)
        relationships = self._apply_multiple_testing_correction(raw_relationships, sig_alpha)

        # 6. Rank Relationships & Build Leaderboard
        ranked_relationships = self._rank_relationships(relationships)

        # 7. Subgroup Consistency & Weak-Global-Strong-Subgroup Analysis
        subgroup_cols = self._detect_subgroup_dimensions(df, profile, set(numeric_cols + dt_cols))
        subgroup_analysis = self._analyze_subgroups(df, ranked_relationships[:10], subgroup_cols, sig_alpha)

        # 8. Summary Diagnostics & Correlation Matrix
        corr_matrix = self._build_correlation_matrix(df, numeric_cols)

        return {
            "task_type": "statistical_analysis",
            "rows_analyzed": n_rows,
            "original_rows": n_rows,
            "features_considered": len(df.columns),
            "features_used": all_usable,
            "numeric_features": numeric_cols,
            "categorical_features": cat_cols,
            "datetime_features": dt_cols,
            "excluded_features": excluded_cols,
            "pairs_evaluated": len(relationships),
            "significance_threshold": sig_alpha,
            "analysis_limited_for_performance": is_limited,
            "relationships": relationships,
            "ranked_relationships": ranked_relationships,
            "top_relationships": ranked_relationships[:10],
            "correlation_matrix": corr_matrix,
            "subgroup_analysis": subgroup_analysis,
            "warnings": ["All reported statistical relationships reflect observational associations and do not prove causal influence."],
            "assumptions": [
                "Pearson correlation measures linear association between continuous variables.",
                "Spearman and Kendall measure monotonic rank association.",
                "ANOVA and Kruskal-Wallis measure group-mean/rank differences.",
                "Chi-square measures independence in categorical contingency tables.",
                "Subgroup analyses evaluate relationship stability across low-cardinality cohorts.",
            ],
            "limitations": [
                "Observational statistical association does not establish cause and effect.",
                "Unobserved confounding variables may explain observed associations.",
            ],
        }

    # --------------------------------------------------------------------------
    # Feature Discovery & Partitioning
    # --------------------------------------------------------------------------

    def _discover_features(
        self,
        df: pd.DataFrame,
        profile: SemanticProfile,
        requested_features: Optional[List[str]] = None,
        target: Optional[str] = None,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Identify and categorize usable features, excluding constants, identifiers, and high-cardinality text."""
        numeric_cols: List[str] = []
        cat_cols: List[str] = []
        dt_cols: List[str] = []
        excluded: List[str] = []

        if requested_features is not None and len(requested_features) > 0:
            candidate_cols = [c for c in requested_features if c in df.columns]
            is_explicit = True
        else:
            is_explicit = False
            candidate_cols = list(df.columns)

        if target is not None and target in df.columns and target not in candidate_cols:
            candidate_cols.append(target)

        for col in candidate_cols:
            series = df[col]

            # 0. Missingness (>80% missing excluded unless explicit)
            if not is_explicit and series.isna().mean() > 0.80:
                excluded.append(str(col))
                continue

            # 1. Identifier exclusion
            if not is_explicit and (col in profile.identifier_columns or (series.nunique(dropna=True) == len(df) and len(df) >= 5 and not pd.api.types.is_numeric_dtype(series))):
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

            # 4. Datetime detection
            if pd.api.types.is_datetime64_any_dtype(series) or col in profile.datetime_candidates:
                dt_cols.append(str(col))
                continue

            # 5. Numeric detection & coercion
            num_s = CanonicalDataLayer.coerce_numeric_series(series)
            num_valid_ratio = num_s.notna().mean()

            if num_valid_ratio >= 0.50 and num_s.nunique(dropna=True) > 1:
                numeric_cols.append(str(col))
            else:
                # Categorical candidate (cardinality between 2 and 50 and not 100% unique key)
                n_uniq = series.nunique(dropna=True)
                if 2 <= n_uniq <= 50 and (n_uniq < len(df) or len(df) < 4):
                    cat_cols.append(str(col))
                else:
                    excluded.append(str(col))

        return numeric_cols, cat_cols, dt_cols, excluded

    # --------------------------------------------------------------------------
    # Candidate Pair Generation
    # --------------------------------------------------------------------------

    def _generate_candidate_pairs(
        self,
        numeric_cols: List[str],
        cat_cols: List[str],
        dt_cols: List[str],
        target: Optional[str] = None,
        max_pairs: int = 250,
    ) -> Tuple[List[Tuple[str, str, str]], bool]:
        """Generate pairwise evaluation tuples: (feat_x, feat_y, pair_type)."""
        pairs: List[Tuple[str, str, str]] = []

        if target is not None:
            # Target-focused mode
            all_others = [c for c in numeric_cols + cat_cols + dt_cols if c != target]
            target_is_num = target in numeric_cols
            target_is_cat = target in cat_cols
            target_is_dt = target in dt_cols

            for other in all_others:
                if target_is_num and other in numeric_cols:
                    pairs.append((target, other, "numeric_numeric"))
                elif target_is_num and other in cat_cols:
                    pairs.append((target, other, "numeric_categorical"))
                elif target_is_cat and other in numeric_cols:
                    pairs.append((other, target, "numeric_categorical"))
                elif target_is_cat and other in cat_cols:
                    pairs.append((target, other, "categorical_categorical"))
                elif target_is_num and other in dt_cols:
                    pairs.append((target, other, "numeric_datetime"))
                elif target_is_dt and other in numeric_cols:
                    pairs.append((other, target, "numeric_datetime"))
        else:
            # All-pairwise mode
            # A. Numeric <-> Numeric
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    pairs.append((numeric_cols[i], numeric_cols[j], "numeric_numeric"))

            # B. Numeric <-> Categorical
            for num in numeric_cols:
                for cat in cat_cols:
                    pairs.append((num, cat, "numeric_categorical"))

            # C. Categorical <-> Categorical
            for i in range(len(cat_cols)):
                for j in range(i + 1, len(cat_cols)):
                    pairs.append((cat_cols[i], cat_cols[j], "categorical_categorical"))

            # D. Numeric <-> Datetime
            for num in numeric_cols:
                for dt in dt_cols:
                    pairs.append((num, dt, "numeric_datetime"))

        is_limited = len(pairs) > max_pairs
        if is_limited:
            pairs = pairs[:max_pairs]

        return pairs, is_limited

    # --------------------------------------------------------------------------
    # Pairwise Non-Destructive Evaluation
    # --------------------------------------------------------------------------

    def _evaluate_pair(
        self,
        df: pd.DataFrame,
        feat_x: str,
        feat_y: str,
        pair_type: str,
        alpha: float,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate relationship between two features non-destructively on pairwise-valid observations."""
        s_x = df[feat_x]
        s_y = df[feat_y]

        if pair_type == "numeric_numeric":
            return self._eval_numeric_numeric(s_x, s_y, feat_x, feat_y, alpha)
        elif pair_type == "numeric_categorical":
            return self._eval_numeric_categorical(s_x, s_y, feat_x, feat_y, alpha)
        elif pair_type == "categorical_categorical":
            return self._eval_categorical_categorical(s_x, s_y, feat_x, feat_y, alpha)
        elif pair_type == "numeric_datetime":
            return self._eval_numeric_datetime(s_x, s_y, feat_x, feat_y, alpha)
        return None

    # 1. Numeric <-> Numeric (Pearson, Spearman, Kendall, Outlier Sensitivity)
    def _eval_numeric_numeric(
        self,
        s_x: pd.Series,
        s_y: pd.Series,
        name_x: str,
        name_y: str,
        alpha: float,
    ) -> Optional[Dict[str, Any]]:
        num_x = CanonicalDataLayer.coerce_numeric_series(s_x)
        num_y = CanonicalDataLayer.coerce_numeric_series(s_y)

        # Pairwise non-destructive mask
        valid_mask = num_x.notna() & num_y.notna()
        n_valid = int(valid_mask.sum())

        if n_valid < 3:
            return None

        vx = num_x[valid_mask].to_numpy(dtype=float)
        vy = num_y[valid_mask].to_numpy(dtype=float)

        # Variance check
        if np.std(vx) < 1e-9 or np.std(vy) < 1e-9:
            return None

        # A. Pearson correlation
        try:
            r_val, p_pearson = stats.pearsonr(vx, vy)
            r_val = float(r_val) if not math.isnan(r_val) else 0.0
            p_pearson = float(p_pearson) if not math.isnan(p_pearson) else 1.0
        except Exception:
            r_val, p_pearson = 0.0, 1.0

        # Pearson 95% Confidence Interval (Fisher z-transform)
        ci_lower, ci_upper = None, None
        if n_valid > 3 and abs(r_val) < 0.9999:
            try:
                z = np.arctanh(r_val)
                se = 1.0 / math.sqrt(n_valid - 3)
                z_crit = 1.96
                ci_lower = round(float(np.tanh(z - z_crit * se)), 4)
                ci_upper = round(float(np.tanh(z + z_crit * se)), 4)
            except Exception:
                pass

        # B. Spearman rank correlation
        try:
            rho_val, p_spearman = stats.spearmanr(vx, vy)
            rho_val = float(rho_val) if not math.isnan(rho_val) else 0.0
            p_spearman = float(p_spearman) if not math.isnan(p_spearman) else 1.0
        except Exception:
            rho_val, p_spearman = 0.0, 1.0

        # C. Kendall tau (for N <= 2000)
        tau_val, p_kendall = None, None
        if n_valid <= 2000:
            try:
                t_res = stats.kendalltau(vx, vy)
                tau_val = round(float(t_res.statistic), 4) if not math.isnan(t_res.statistic) else 0.0
                p_kendall = round(float(t_res.pvalue), 6) if not math.isnan(t_res.pvalue) else 1.0
            except Exception:
                pass

        # Outlier sensitivity detection (difference between Pearson and Spearman rank correlation)
        r_vs_rho_delta = round(float(abs(r_val - rho_val)), 4)
        outlier_sensitivity = bool(r_vs_rho_delta > 0.20)

        # Select primary method: Robust Spearman if outlier sensitivity is high; else Pearson
        primary_method = "spearman" if outlier_sensitivity else "pearson"
        primary_stat = rho_val if outlier_sensitivity else r_val
        primary_p = p_spearman if outlier_sensitivity else p_pearson
        effect_size = abs(primary_stat)

        strength = self._classify_correlation_strength(effect_size)
        direction = "positive" if primary_stat > 0 else "negative" if primary_stat < 0 else "neutral"

        # Strictly non-causal interpretation
        if abs(primary_stat) >= 0.10:
            interp = f"Features '{name_x}' and '{name_y}' exhibit a {strength} {direction} {primary_method} association (statistic = {primary_stat:.3f}, p = {primary_p:.4f})."
        else:
            interp = f"Features '{name_x}' and '{name_y}' exhibit negligible statistical association (statistic = {primary_stat:.3f})."

        return {
            "feature_x": name_x,
            "feature_y": name_y,
            "pair_type": "numeric_numeric",
            "primary_method": primary_method,
            "statistic": round(primary_stat, 4),
            "p_value": round(primary_p, 6),
            "effect_size": round(effect_size, 4),
            "strength": strength,
            "direction": direction,
            "valid_rows": n_valid,
            "missing_x": int((~num_x.notna()).sum()),
            "missing_y": int((~num_y.notna()).sum()),
            "outlier_sensitivity": outlier_sensitivity,
            "r_vs_rho_delta": r_vs_rho_delta,
            "pearson": {
                "r": round(r_val, 4),
                "p_value": round(p_pearson, 6),
                "confidence_interval": [ci_lower, ci_upper] if ci_lower is not None else None,
            },
            "spearman": {
                "rho": round(rho_val, 4),
                "p_value": round(p_spearman, 6),
            },
            "kendall": {
                "tau": tau_val,
                "p_value": p_kendall,
            } if tau_val is not None else None,
            "interpretation": interp,
            "limitations": [
                "Pearson measures linear relationship; Spearman measures monotonic relationship.",
                "Statistical association does not establish causal dependency.",
            ],
        }

    # 2. Numeric <-> Categorical (Point-Biserial, ANOVA, Kruskal-Wallis, Eta-squared)
    def _eval_numeric_categorical(
        self,
        s_num: pd.Series,
        s_cat: pd.Series,
        name_num: str,
        name_cat: str,
        alpha: float,
    ) -> Optional[Dict[str, Any]]:
        num_clean = CanonicalDataLayer.coerce_numeric_series(s_num)
        cat_clean = s_cat.astype(str).replace({"nan": None, "None": None, "<NA>": None})

        valid_mask = num_clean.notna() & cat_clean.notna()
        n_valid = int(valid_mask.sum())

        if n_valid < 4:
            return None

        v_num = num_clean[valid_mask].to_numpy(dtype=float)
        v_cat = cat_clean[valid_mask].to_numpy()

        unique_groups = [g for g in np.unique(v_cat) if g is not None]
        k_groups = len(unique_groups)

        if k_groups < 2:
            return None

        groups_data = [v_num[v_cat == g] for g in unique_groups]
        # Filter groups with at least 1 observation
        groups_data = [g for g in groups_data if len(g) > 0]
        if len(groups_data) < 2:
            return None

        # Point-biserial if exactly 2 groups
        pb_r, pb_p = 0.0, 1.0
        if k_groups == 2:
            try:
                bin_indicator = (v_cat == unique_groups[0]).astype(int)
                pb_res = stats.pointbiserialr(bin_indicator, v_num)
                pb_r = float(pb_res.statistic) if not math.isnan(pb_res.statistic) else 0.0
                pb_p = float(pb_res.pvalue) if not math.isnan(pb_res.pvalue) else 1.0
            except Exception:
                pb_r, pb_p = 0.0, 1.0

        # One-Way ANOVA F-test
        try:
            f_stat, p_anova = stats.f_oneway(*groups_data)
            f_stat = float(f_stat) if not math.isnan(f_stat) else 0.0
            p_anova = float(p_anova) if not math.isnan(p_anova) else 1.0
        except Exception:
            f_stat, p_anova = 0.0, 1.0

        # Non-parametric Kruskal-Wallis H-test
        try:
            h_stat, p_kruskal = stats.kruskal(*groups_data)
            h_stat = float(h_stat) if not math.isnan(h_stat) else 0.0
            p_kruskal = float(p_kruskal) if not math.isnan(p_kruskal) else 1.0
        except Exception:
            h_stat, p_kruskal = 0.0, 1.0

        # Calculate Eta-squared effect size (SS_between / SS_total)
        grand_mean = float(np.mean(v_num))
        ss_total = float(np.sum((v_num - grand_mean) ** 2))
        ss_between = float(sum(len(g) * ((np.mean(g) - grand_mean) ** 2) for g in groups_data))
        eta_sq = ss_between / ss_total if ss_total > 1e-9 else 0.0
        eta_sq = max(0.0, min(1.0, float(eta_sq)))

        strength = self._classify_eta_squared_strength(eta_sq)
        primary_method = "point_biserial" if k_groups == 2 else "anova"
        primary_p = pb_p if k_groups == 2 else p_anova
        primary_stat = pb_r if k_groups == 2 else f_stat

        group_summaries = {}
        for g_name, g_vals in zip(unique_groups, groups_data):
            group_summaries[str(g_name)] = {
                "count": len(g_vals),
                "mean": round(float(np.mean(g_vals)), 4),
                "median": round(float(np.median(g_vals)), 4),
                "std": round(float(np.std(g_vals)), 4) if len(g_vals) > 1 else 0.0,
            }

        interp = f"Values of '{name_num}' exhibit {strength} differences across categories of '{name_cat}' (eta_squared = {eta_sq:.3f}, p = {primary_p:.4f})."

        return {
            "feature_x": name_num,
            "feature_y": name_cat,
            "pair_type": "numeric_categorical",
            "primary_method": primary_method,
            "statistic": round(primary_stat, 4),
            "p_value": round(primary_p, 6),
            "effect_size": round(eta_sq, 4),
            "strength": strength,
            "valid_rows": n_valid,
            "missing_x": int((~num_clean.notna()).sum()),
            "missing_y": int((~cat_clean.notna()).sum()),
            "group_count": k_groups,
            "group_summaries": group_summaries,
            "anova": {
                "f_statistic": round(f_stat, 4),
                "p_value": round(p_anova, 6),
                "eta_squared": round(eta_sq, 4),
            },
            "kruskal_wallis": {
                "h_statistic": round(h_stat, 4),
                "p_value": round(p_kruskal, 6),
            },
            "point_biserial": {
                "r": round(pb_r, 4),
                "p_value": round(pb_p, 6),
            } if k_groups == 2 else None,
            "interpretation": interp,
            "limitations": [
                "ANOVA assumes normality and homoscedasticity across groups.",
                "Group differences do not prove the categorical factor causes numerical variation.",
            ],
        }

    # 3. Categorical <-> Categorical (Chi-Square, Cramer's V, Fisher's Exact)
    def _eval_categorical_categorical(
        self,
        s_x: pd.Series,
        s_y: pd.Series,
        name_x: str,
        name_y: str,
        alpha: float,
    ) -> Optional[Dict[str, Any]]:
        c_x = s_x.astype(str).replace({"nan": None, "None": None, "<NA>": None})
        c_y = s_y.astype(str).replace({"nan": None, "None": None, "<NA>": None})

        valid_mask = c_x.notna() & c_y.notna()
        n_valid = int(valid_mask.sum())

        if n_valid < 5:
            return None

        vx = c_x[valid_mask]
        vy = c_y[valid_mask]

        if vx.nunique() < 2 or vy.nunique() < 2:
            return None

        # Contingency table
        contingency = pd.crosstab(vx, vy)
        r, c = contingency.shape

        if r < 2 or c < 2:
            return None

        try:
            chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
            chi2 = float(chi2) if not math.isnan(chi2) else 0.0
            p_val = float(p_val) if not math.isnan(p_val) else 1.0
        except Exception:
            chi2, p_val, dof, expected = 0.0, 1.0, 0, np.zeros((r, c))

        # Expected frequency assumption check: no more than 20% of cells < 5
        sparse_cells_pct = float(np.mean(expected < 5.0)) * 100.0
        sparse_warning = sparse_cells_pct > 20.0

        # Cramer's V
        min_dim = min(r - 1, c - 1)
        if min_dim > 0 and n_valid > 0:
            cramers_v = math.sqrt(chi2 / (n_valid * min_dim))
            cramers_v = max(0.0, min(1.0, float(cramers_v)))
        else:
            cramers_v = 0.0

        strength = self._classify_cramers_v_strength(cramers_v)

        # Fisher's exact test for 2x2 if expected frequencies are low
        fisher_p = None
        if r == 2 and c == 2:
            try:
                _, fisher_p = stats.fisher_exact(contingency)
                fisher_p = float(fisher_p) if not math.isnan(fisher_p) else None
            except Exception:
                pass

        final_p = fisher_p if (fisher_p is not None and sparse_warning) else p_val
        interp = f"Categories of '{name_x}' and '{name_y}' exhibit {strength} association (Cramer's V = {cramers_v:.3f}, p = {final_p:.4f})."

        return {
            "feature_x": name_x,
            "feature_y": name_y,
            "pair_type": "categorical_categorical",
            "primary_method": "chi_square",
            "statistic": round(chi2, 4),
            "p_value": round(final_p, 6),
            "effect_size": round(cramers_v, 4),
            "strength": strength,
            "valid_rows": n_valid,
            "missing_x": int((~c_x.notna()).sum()),
            "missing_y": int((~c_y.notna()).sum()),
            "contingency_shape": [r, c],
            "chi_square": {
                "statistic": round(chi2, 4),
                "degrees_of_freedom": int(dof),
                "p_value": round(p_val, 6),
                "cramers_v": round(cramers_v, 4),
                "sparse_cells_pct": round(sparse_cells_pct, 1),
                "sparse_warning": sparse_warning,
            },
            "fisher_exact_p_value": round(fisher_p, 6) if fisher_p is not None else None,
            "interpretation": interp,
            "limitations": [
                "Chi-square tests categorical independence; large samples may produce low p-values for weak associations.",
                "Categorical co-occurrence does not imply a causal relationship.",
            ],
        }

    # 4. Numeric <-> Datetime (Trend Association)
    def _eval_numeric_datetime(
        self,
        s_num: pd.Series,
        s_dt: pd.Series,
        name_num: str,
        name_dt: str,
        alpha: float,
    ) -> Optional[Dict[str, Any]]:
        num_clean = CanonicalDataLayer.coerce_numeric_series(s_num)
        dt_clean = CanonicalDataLayer.coerce_datetime_series(s_dt)

        valid_mask = num_clean.notna() & dt_clean.notna()
        n_valid = int(valid_mask.sum())

        if n_valid < 4:
            return None

        v_num = num_clean[valid_mask].to_numpy(dtype=float)
        v_dt = dt_clean[valid_mask]

        # Convert datetime to elapsed days
        min_dt = v_dt.min()
        elapsed_days = (v_dt - min_dt).dt.total_seconds() / 86400.0
        v_days = elapsed_days.to_numpy(dtype=float)

        if np.std(v_days) < 1e-9 or np.std(v_num) < 1e-9:
            return None

        try:
            r_val, p_val = stats.pearsonr(v_days, v_num)
            r_val = float(r_val) if not math.isnan(r_val) else 0.0
            p_val = float(p_val) if not math.isnan(p_val) else 1.0
        except Exception:
            r_val, p_val = 0.0, 1.0

        try:
            rho_val, p_rho = stats.spearmanr(v_days, v_num)
            rho_val = float(rho_val) if not math.isnan(rho_val) else 0.0
        except Exception:
            rho_val = 0.0

        effect_size = abs(r_val)
        strength = self._classify_correlation_strength(effect_size)
        direction = "upward" if r_val > 0 else "downward" if r_val < 0 else "flat"

        interp = f"Feature '{name_num}' displays a {strength} {direction} chronological trend over '{name_dt}' (r = {r_val:.3f}, p = {p_val:.4f})."

        return {
            "feature_x": name_num,
            "feature_y": name_dt,
            "pair_type": "numeric_datetime",
            "primary_method": "temporal_trend",
            "statistic": round(r_val, 4),
            "p_value": round(p_val, 6),
            "effect_size": round(effect_size, 4),
            "strength": strength,
            "trend_direction": direction,
            "valid_rows": n_valid,
            "pearson_r": round(r_val, 4),
            "spearman_rho": round(rho_val, 4),
            "interpretation": interp,
            "limitations": [
                "Temporal trend indicates chronological co-movement, not direct temporal causation.",
            ],
        }

    # --------------------------------------------------------------------------
    # Multiple Testing Correction (Benjamini-Hochberg FDR)
    # --------------------------------------------------------------------------

    def _apply_multiple_testing_correction(
        self,
        relationships: List[Dict[str, Any]],
        alpha: float,
    ) -> List[Dict[str, Any]]:
        """Apply Benjamini-Hochberg FDR correction to control false discovery rate across evaluated pairs."""
        m = len(relationships)
        if m == 0:
            return []

        # Sort indices by raw p-value
        p_vals = [r["p_value"] for r in relationships]
        sorted_indices = sorted(range(m), key=lambda i: p_vals[i])

        # Compute BH adjusted p-values
        adj_p = [0.0] * m
        min_p = 1.0

        # Step-up calculation in reverse sorted order
        for rank_idx in range(m - 1, -1, -1):
            orig_idx = sorted_indices[rank_idx]
            rank = rank_idx + 1  # 1-indexed
            raw_p = p_vals[orig_idx]
            curr_adj = min(1.0, (m / rank) * raw_p)
            min_p = min(min_p, curr_adj)
            adj_p[orig_idx] = round(float(min_p), 6)

        for i, rel in enumerate(relationships):
            adjusted = adj_p[i]
            rel["adjusted_p_value"] = adjusted
            rel["is_significant"] = bool(adjusted < alpha)
            rel["significance_level"] = "statistically_significant" if rel["is_significant"] else "not_significant"

        return relationships

    # --------------------------------------------------------------------------
    # Relationship Ranking & Correlation Matrix
    # --------------------------------------------------------------------------

    def _rank_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank relationships by composite score balancing effect size, adjusted significance, and sample support."""
        for rel in relationships:
            eff = rel.get("effect_size", 0.0)
            adj_p = rel.get("adjusted_p_value", 1.0)
            n_val = rel.get("valid_rows", 10)

            # Reliability factor from adjusted p-value [0, 1]
            p_factor = 1.0 - min(1.0, adj_p)
            # Sample adequacy factor [0.5, 1.0]
            n_factor = min(1.0, max(0.5, math.sqrt(n_val) / 10.0))

            # Composite ranking score: 70% effect size, 20% significance reliability, 10% sample power
            score = (0.70 * eff + 0.20 * p_factor + 0.10 * n_factor)
            rel["ranking_score"] = round(float(score), 4)

        ranked = sorted(relationships, key=lambda r: r["ranking_score"], reverse=True)
        return ranked

    def _build_correlation_matrix(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
        """Build pairwise Pearson correlation matrix for numeric features."""
        corr_matrix: Dict[str, Dict[str, Optional[float]]] = {}
        for c1 in numeric_cols:
            corr_matrix[c1] = {}
            for c2 in numeric_cols:
                if c1 == c2:
                    corr_matrix[c1][c2] = 1.0
                else:
                    s1 = CanonicalDataLayer.coerce_numeric_series(df[c1])
                    s2 = CanonicalDataLayer.coerce_numeric_series(df[c2])
                    mask = s1.notna() & s2.notna()
                    if mask.sum() >= 3 and s1[mask].std() > 1e-9 and s2[mask].std() > 1e-9:
                        try:
                            r, _ = stats.pearsonr(s1[mask], s2[mask])
                            corr_matrix[c1][c2] = round(float(r), 4) if not math.isnan(r) else None
                        except Exception:
                            corr_matrix[c1][c2] = None
                    else:
                        corr_matrix[c1][c2] = None
        return corr_matrix

    # --------------------------------------------------------------------------
    # Strength Classifiers
    # --------------------------------------------------------------------------

    @staticmethod
    def _classify_correlation_strength(effect_size: float) -> str:
        if effect_size >= 0.70:
            return "very_strong"
        elif effect_size >= 0.50:
            return "strong"
        elif effect_size >= 0.30:
            return "moderate"
        elif effect_size >= 0.10:
            return "weak"
        return "negligible"

    @staticmethod
    def _classify_eta_squared_strength(eta_sq: float) -> str:
        if eta_sq >= 0.14:
            return "large"
        elif eta_sq >= 0.06:
            return "medium"
        elif eta_sq >= 0.01:
            return "small"
        return "negligible"

    @staticmethod
    def _classify_cramers_v_strength(v: float) -> str:
        if v >= 0.50:
            return "strong"
        elif v >= 0.30:
            return "moderate"
        elif v >= 0.10:
            return "weak"
        return "negligible"

    # --------------------------------------------------------------------------
    # Subgroup Analysis & Heterogeneity Detection
    # --------------------------------------------------------------------------

    def _detect_subgroup_dimensions(
        self,
        df: pd.DataFrame,
        profile: SemanticProfile,
        exclude_cols: Set[str],
    ) -> List[str]:
        """Automatically detect suitable low-cardinality categorical dimensions for subgroup analysis."""
        candidate_cols: List[Tuple[str, int, int]] = []
        semantic_keywords = (
            "segment", "group", "zone", "region", "tier", "category", "type",
            "channel", "plan", "class", "status", "country", "state",
            "department", "division", "cluster", "cohort", "market",
        )

        for col in df.columns:
            if col in exclude_cols or col in profile.identifier_columns or col in profile.constant_columns:
                continue
            series = df[col]
            n_unique = series.nunique(dropna=True)
            if 2 <= n_unique <= 15:
                val_counts = series.value_counts(dropna=True)
                if len(val_counts) >= 2 and val_counts.iloc[1] >= 3:
                    score = 0
                    col_lower = str(col).lower()
                    for kw in semantic_keywords:
                        if kw in col_lower:
                            score += 10
                    if 2 <= n_unique <= 6:
                        score += 5
                    candidate_cols.append((str(col), score, n_unique))

        candidate_cols.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        return [c[0] for c in candidate_cols[:4]]

    def _analyze_subgroups(
        self,
        df: pd.DataFrame,
        top_relationships: List[Dict[str, Any]],
        subgroup_cols: List[str],
        alpha: float,
    ) -> Dict[str, Any]:
        """
        Evaluate relationship consistency across subgroups, detecting weak-global/strong-subgroup
        heterogeneity and checking for Simpson's paradox mathematically.
        """
        if not subgroup_cols or not top_relationships:
            return {
                "dimensions_evaluated": [],
                "subgroup_relationships": [],
                "weak_global_strong_subgroup_findings": [],
                "simpsons_paradox_findings": [],
                "subgroup_consistency_summary": [],
            }

        subgroup_relationships: List[Dict[str, Any]] = []
        weak_global_findings: List[Dict[str, Any]] = []
        simpsons_findings: List[Dict[str, Any]] = []
        consistency_summaries: List[Dict[str, Any]] = []

        for rel in top_relationships:
            if rel.get("pair_type") != "numeric_numeric":
                continue

            fx = rel["feature_x"]
            fy = rel["feature_y"]
            global_r = rel.get("pearson", {}).get("r", rel.get("statistic", 0.0))
            global_p = rel.get("pearson", {}).get("p_value", rel.get("p_value", 1.0))
            global_rho = rel.get("spearman", {}).get("rho", 0.0)

            num_x = CanonicalDataLayer.coerce_numeric_series(df[fx])
            num_y = CanonicalDataLayer.coerce_numeric_series(df[fy])

            for s_col in subgroup_cols:
                if s_col == fx or s_col == fy:
                    continue

                sub_entries: List[Dict[str, Any]] = []
                sub_r_list: List[float] = []

                s_series = df[s_col]
                unique_vals = s_series.dropna().unique()

                for val in unique_vals:
                    mask = (s_series == val) & num_x.notna() & num_y.notna()
                    n_sub = int(mask.sum())

                    if n_sub < 5:
                        continue

                    vx = num_x[mask].to_numpy(dtype=float)
                    vy = num_y[mask].to_numpy(dtype=float)

                    if np.std(vx) < 1e-9 or np.std(vy) < 1e-9:
                        continue

                    try:
                        r_sub, p_sub = stats.pearsonr(vx, vy)
                        r_sub = float(r_sub) if not math.isnan(r_sub) else 0.0
                        p_sub = float(p_sub) if not math.isnan(p_sub) else 1.0
                    except Exception:
                        r_sub, p_sub = 0.0, 1.0

                    try:
                        rho_sub, _ = stats.spearmanr(vx, vy)
                        rho_sub = float(rho_sub) if not math.isnan(rho_sub) else 0.0
                    except Exception:
                        rho_sub = 0.0

                    sub_strength = self._classify_correlation_strength(abs(r_sub))
                    sub_direction = "positive" if r_sub > 0 else "negative" if r_sub < 0 else "neutral"

                    entry = {
                        "feature_x": fx,
                        "feature_y": fy,
                        "subgroup_dimension": s_col,
                        "subgroup_value": str(val),
                        "valid_rows": n_sub,
                        "pearson_r": round(r_sub, 4),
                        "p_value": round(p_sub, 6),
                        "spearman_rho": round(rho_sub, 4),
                        "strength": sub_strength,
                        "direction": sub_direction,
                        "global_r": round(global_r, 4),
                    }
                    sub_entries.append(entry)
                    subgroup_relationships.append(entry)
                    sub_r_list.append(r_sub)

                    # Check for Weak Global but Strong Within Subgroup
                    # Condition: Global |r| < 0.35 and Subgroup |r| >= 0.50 (or |r_sub| - |global_r| >= 0.30) with n >= 8 and p_sub < 0.05
                    is_weak_global = abs(global_r) < 0.35
                    is_strong_sub = (abs(r_sub) >= 0.50 or (abs(r_sub) - abs(global_r) >= 0.30)) and p_sub < 0.05 and n_sub >= 8
                    if is_weak_global and is_strong_sub:
                        weak_global_findings.append({
                            "feature_x": fx,
                            "feature_y": fy,
                            "subgroup_dimension": s_col,
                            "subgroup_value": str(val),
                            "global_r": round(global_r, 4),
                            "global_strength": self._classify_correlation_strength(abs(global_r)),
                            "subgroup_r": round(r_sub, 4),
                            "subgroup_p_value": round(p_sub, 6),
                            "subgroup_strength": sub_strength,
                            "subgroup_valid_rows": n_sub,
                            "finding": f"Relationship between '{fx}' and '{fy}' is weak overall (r = {global_r:.3f}), but becomes {sub_strength} within {s_col} = '{val}' (r = {r_sub:.3f}, p = {p_sub:.4f}, n = {n_sub}).",
                        })

                # Check for Simpson's Paradox (Mathematical sign flip with statistical support)
                if len(sub_r_list) >= 2:
                    neg_subgroups = [r for r in sub_r_list if r < -0.15]
                    pos_subgroups = [r for r in sub_r_list if r > 0.15]
                    if global_r > 0.20 and global_p < 0.05 and len(neg_subgroups) >= max(2, len(sub_r_list) // 2):
                        simpsons_findings.append({
                            "feature_x": fx,
                            "feature_y": fy,
                            "subgroup_dimension": s_col,
                            "global_r": round(global_r, 4),
                            "subgroup_correlations": [round(r, 3) for r in sub_r_list],
                            "explanation": f"Demonstrated Simpson's Paradox: '{fx}' and '{fy}' exhibit positive correlation globally (r = {global_r:.3f}), but reverse to negative association across subgroups of '{s_col}'.",
                        })
                    elif global_r < -0.20 and global_p < 0.05 and len(pos_subgroups) >= max(2, len(sub_r_list) // 2):
                        simpsons_findings.append({
                            "feature_x": fx,
                            "feature_y": fy,
                            "subgroup_dimension": s_col,
                            "global_r": round(global_r, 4),
                            "subgroup_correlations": [round(r, 3) for r in sub_r_list],
                            "explanation": f"Demonstrated Simpson's Paradox: '{fx}' and '{fy}' exhibit negative correlation globally (r = {global_r:.3f}), but reverse to positive association across subgroups of '{s_col}'.",
                        })

                if sub_entries:
                    all_same_sign = all(r > 0 for r in sub_r_list) or all(r < 0 for r in sub_r_list)
                    min_r = min(sub_r_list)
                    max_r = max(sub_r_list)
                    consistency_summaries.append({
                        "feature_x": fx,
                        "feature_y": fy,
                        "subgroup_dimension": s_col,
                        "is_directionally_consistent": all_same_sign,
                        "r_range": [round(min_r, 4), round(max_r, 4)],
                        "subgroups_evaluated": len(sub_entries),
                    })

                    if "subgroups" not in rel:
                        rel["subgroups"] = {}
                    rel["subgroups"][s_col] = sub_entries

        return {
            "dimensions_evaluated": subgroup_cols,
            "subgroup_relationships": subgroup_relationships,
            "weak_global_strong_subgroup_findings": weak_global_findings,
            "simpsons_paradox_findings": simpsons_findings,
            "subgroup_consistency_summary": consistency_summaries,
        }

    def _extract_dataframe(self, data: Any) -> Optional[pd.DataFrame]:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            for df in data.values():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
        return None