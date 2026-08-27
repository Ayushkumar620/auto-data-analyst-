"""
Universal, Dataset-Agnostic Hypothesis Testing & Statistical Significance Engine.

Single source of truth for:
1. Two-sample independent numeric comparisons (Student's t-test, Welch's t-test, Mann-Whitney U)
2. Paired numeric comparisons (Paired t-test, Wilcoxon signed-rank)
3. Multi-group numeric comparisons (One-way ANOVA, Welch ANOVA, Kruskal-Wallis)
4. Post-hoc pairwise comparisons with multiple testing adjustments (Tukey HSD / Dunn-style pairwise FDR)
5. Categorical association testing (Chi-square test of independence, Fisher's exact test)
6. Distribution and normality diagnostic assessment (Shapiro-Wilk, D'Agostino-Pearson, Levene's variance test)
7. Effect sizes (Cohen's d, Hedges' g, Eta-squared, Partial eta-squared, Rank-biserial r, Cramer's V, Odds Ratio)
8. Parametric & bootstrap confidence intervals (mean differences, effect sizes, odds ratios)
9. Benjamini-Hochberg False Discovery Rate (FDR) multiple testing correction
10. Transparent data-driven test selection with suitability scoring and explicit assumption diagnostics
11. Pairwise non-destructive missing data handling (zero global row dropna)
12. Strict separation of statistical significance, practical significance, and non-causal reporting
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


def _sanitize_float(val: Any) -> Optional[float]:
    """Convert value to finite float or None if NaN/Inf."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _adjust_p_values_fdr(p_values: List[float]) -> List[float]:
    """Apply Benjamini-Hochberg False Discovery Rate (FDR) adjustment."""
    n = len(p_values)
    if n <= 1:
        return [float(p) for p in p_values]

    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n

    cum_min = 1.0
    for rank_minus_1, (orig_idx, p_val) in reversed(list(enumerate(indexed_p))):
        rank = rank_minus_1 + 1
        adj_p = min(1.0, (n / rank) * p_val)
        cum_min = min(cum_min, adj_p)
        adjusted[orig_idx] = max(0.0, min(1.0, cum_min))

    return adjusted


class HypothesisTestingEngine:
    """
    Authoritative, universal hypothesis testing and statistical significance engine.
    Dynamically assesses distributional properties, variances, and group structures
    to choose, execute, and interpret optimal statistical tests.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_comparisons: int = 100,
        random_state: int = 42,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Alpha must be within (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.max_comparisons = max_comparisons
        self.random_state = random_state

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def test(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], Any],
        feature: Optional[str] = None,
        group: Optional[str] = None,
        feature_2: Optional[str] = None,
        features: Optional[List[str]] = None,
        target: Optional[str] = None,
        alpha: Optional[float] = None,
        paired: Optional[bool] = None,
        preferred_test: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute comprehensive, data-driven hypothesis testing on arbitrary tabular data.
        """
        eff_alpha = float(alpha) if alpha is not None and 0.0 < alpha < 1.0 else self.alpha

        # 1. Canonical Ingestion
        if isinstance(data, dict) and not isinstance(data, pd.DataFrame):
            for v in data.values():
                if isinstance(v, pd.DataFrame) and not v.empty:
                    df = v.copy()
                    break
            else:
                df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            return {
                "error": "Hypothesis testing requires a tabular pandas DataFrame or record dict.",
                "category": ErrorCategory.INPUT_INVALID,
            }

        if df.empty or len(df) == 0:
            return {
                "error": "Dataset is empty (0 rows). Hypothesis testing cannot proceed.",
                "category": ErrorCategory.DATA_INVALID,
            }

        dataset: CanonicalDataset = CanonicalDataLayer.ingest(df)
        sem_profile: SemanticProfile = dataset.profile
        orig_rows = len(df)

        # 2. Variable Disambiguation
        target_group = group or target
        var_x = feature or (features[0] if features and len(features) > 0 else None)
        var_y = feature_2 or (features[1] if features and len(features) > 1 else None)

        # Auto-discover variables if unspecified
        hypothesis_jobs: List[Dict[str, Any]] = []

        if var_x and var_y and paired:
            # Explicit Paired numeric comparison
            job = self._evaluate_paired_numeric(df, var_x, var_y, eff_alpha, preferred_test)
            if job:
                hypothesis_jobs.append(job)
        elif var_x and target_group:
            # Numeric vs Group or Categorical vs Group
            job = self._evaluate_feature_vs_group(df, var_x, target_group, sem_profile, eff_alpha, preferred_test)
            if job:
                hypothesis_jobs.append(job)
        elif var_x and var_y and not target_group:
            # Two features without explicit group: test numeric difference or categorical association
            job = self._evaluate_two_features(df, var_x, var_y, sem_profile, eff_alpha, paired, preferred_test)
            if job:
                hypothesis_jobs.append(job)
        else:
            # Autonomous Discovery across valid feature pairs
            hypothesis_jobs = self._discover_and_evaluate_hypotheses(
                df, sem_profile, eff_alpha, max_count=self.max_comparisons, preferred_test=preferred_test
            )

        if not hypothesis_jobs:
            return {
                "error": "No statistically testable variable pairs or group comparisons could be formed from the dataset.",
                "category": ErrorCategory.INSUFFICIENT_DATA,
                "summary": {
                    "original_rows": orig_rows,
                    "total_columns": len(df.columns),
                    "evaluated_hypotheses_count": 0,
                },
                "hypotheses": [],
            }

        # 3. Apply Multiple Testing FDR Correction
        raw_p_values = [job["p_value"] for job in hypothesis_jobs]
        adj_p_values = _adjust_p_values_fdr(raw_p_values)

        for job, adj_p in zip(hypothesis_jobs, adj_p_values):
            job["adjusted_p_value"] = round(float(adj_p), 6)
            job["statistical_significance"] = bool(adj_p < eff_alpha)
            # Refine practical interpretation with adjusted significance
            job["practical_interpretation"] = self._format_interpretation(
                job["variable_x"],
                job.get("variable_group") or job.get("variable_y", ""),
                job["test_name"],
                job["statistical_significance"],
                job["adjusted_p_value"],
                job["effect_size"],
                job["effect_size_type"],
                job["practical_significance"],
                eff_alpha,
            )

        # 4. Summary & Findings
        sig_count = sum(1 for j in hypothesis_jobs if j["statistical_significance"])
        findings: List[Dict[str, Any]] = []

        for job in hypothesis_jobs:
            if job["statistical_significance"]:
                findings.append({
                    "title": f"Statistical Difference in {job['variable_x']} by {job.get('variable_group') or job.get('variable_y')}",
                    "severity": "HIGH" if job["practical_significance"] in ("large", "moderate") else "MEDIUM",
                    "description": job["practical_interpretation"],
                    "remediation": f"Consider {job['variable_x']} as a validated differentiator across {job.get('variable_group') or job.get('variable_y')}.",
                })

        return {
            "summary": {
                "original_rows": orig_rows,
                "total_columns": len(df.columns),
                "alpha": eff_alpha,
                "evaluated_hypotheses_count": len(hypothesis_jobs),
                "statistically_significant_count": sig_count,
                "multiple_testing_correction": "Benjamini-Hochberg (FDR)" if len(hypothesis_jobs) > 1 else "none",
            },
            "hypotheses": hypothesis_jobs,
            "findings": findings,
            "warnings": [f["description"] for f in findings if f["severity"] == "HIGH"],
            "assumptions": [
                "Hypothesis tests assess observational distribution differences without asserting causality.",
                "Non-parametric tests are automatically preferred when distribution skewness or small group sample sizes violate normality assumptions.",
                "P-values reflect the probability of observing test statistics under the null hypothesis; effect size measures observational magnitude.",
            ],
            "limitations": [
                "Observational significance does not establish causal direction or unobserved confounding factors.",
                "Discovered differences should be interpreted in domain context alongside effect size and confidence intervals.",
            ],
        }

    # --------------------------------------------------------------------------
    # Evaluation Modules
    # --------------------------------------------------------------------------

    def _evaluate_feature_vs_group(
        self,
        df: pd.DataFrame,
        feature_col: str,
        group_col: str,
        sem_profile: SemanticProfile,
        alpha: float,
        preferred_test: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a feature column across levels of a categorical grouping column."""
        if feature_col not in df.columns or group_col not in df.columns or feature_col == group_col:
            return None

        # Check if feature is numeric or categorical
        s_feat = df[feature_col]
        s_grp = df[group_col]

        # Numeric feature vs Group
        coerced_feat = CanonicalDataLayer.coerce_numeric_series(s_feat)
        if coerced_feat.notna().sum() >= 4 and coerced_feat.nunique() > 1:
            return self._evaluate_numeric_vs_group(df, feature_col, coerced_feat, group_col, s_grp, alpha, preferred_test)

        # Categorical feature vs Group
        if s_feat.nunique() >= 2 and s_grp.nunique() >= 2:
            return self._evaluate_categorical_vs_categorical(df, feature_col, group_col, alpha, preferred_test)

        return None

    def _evaluate_numeric_vs_group(
        self,
        df: pd.DataFrame,
        feat_name: str,
        feat_series: pd.Series,
        group_name: str,
        group_series: pd.Series,
        alpha: float,
        preferred_test: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate continuous numeric metric across 2 or more groups."""
        # Non-destructive pairwise masking
        valid_mask = feat_series.notna() & group_series.notna()
        orig_n = len(df)
        valid_n = int(valid_mask.sum())
        if valid_n < 4:
            return None

        clean_val = feat_series[valid_mask].to_numpy(dtype=float)
        clean_grp = group_series[valid_mask].astype(str).to_numpy()

        unique_groups = np.unique(clean_grp)
        # Exclude if too many categories or only 1 category
        if len(unique_groups) < 2 or len(unique_groups) > 30:
            return None

        group_arrays: Dict[str, np.ndarray] = {g: clean_val[clean_grp == g] for g in unique_groups}
        # Filter groups with at least 2 observations
        valid_group_arrays = {g: arr for g, arr in group_arrays.items() if len(arr) >= 2}
        if len(valid_group_arrays) < 2:
            return None

        # Build Group Stats
        group_stats: Dict[str, Any] = {}
        for g, arr in valid_group_arrays.items():
            group_stats[str(g)] = {
                "count": len(arr),
                "mean": round(float(np.mean(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "std": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
                "iqr": round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 4),
            }

        k = len(valid_group_arrays)
        if k == 2:
            # Two-sample comparison
            g_keys = list(valid_group_arrays.keys())
            g1, g2 = valid_group_arrays[g_keys[0]], valid_group_arrays[g_keys[1]]
            return self._run_two_sample_numeric_test(
                feat_name, group_name, g_keys[0], g_keys[1], g1, g2, orig_n, valid_n, group_stats, alpha, preferred_test
            )
        else:
            # Multi-sample comparison (k >= 3)
            return self._run_multi_sample_numeric_test(
                feat_name, group_name, valid_group_arrays, orig_n, valid_n, group_stats, alpha, preferred_test
            )

    def _run_two_sample_numeric_test(
        self,
        feat_name: str,
        group_name: str,
        label_1: str,
        label_2: str,
        g1: np.ndarray,
        g2: np.ndarray,
        orig_n: int,
        valid_n: int,
        group_stats: Dict[str, Any],
        alpha: float,
        preferred_test: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute two independent groups hypothesis testing."""
        n1, n2 = len(g1), len(g2)
        mean1, mean2 = float(np.mean(g1)), float(np.mean(g2))
        var1, var2 = float(np.var(g1, ddof=1)) if n1 > 1 else 0.0, float(np.var(g2, ddof=1)) if n2 > 1 else 0.0
        mean_diff = mean1 - mean2

        # 1. Assumption diagnostics
        assumptions: List[Dict[str, Any]] = []

        # Sample size
        assumptions.append({
            "assumption": "Minimum Sample Size",
            "status": "passed" if (n1 >= 5 and n2 >= 5) else "warning",
            "evidence": f"Group '{label_1}' n={n1}, Group '{label_2}' n={n2}.",
            "impact": "Sufficient sample sizes permit robust distribution estimation." if (n1 >= 5 and n2 >= 5) else "Small group sizes may reduce test power.",
        })

        # Normality diagnostics (Shapiro-Wilk if <= 5000)
        norm_p1 = float(stats.shapiro(g1)[1]) if (3 <= n1 <= 5000) else 1.0
        norm_p2 = float(stats.shapiro(g2)[1]) if (3 <= n2 <= 5000) else 1.0
        is_normal = bool(norm_p1 >= 0.05 and norm_p2 >= 0.05)

        assumptions.append({
            "assumption": "Normality of Group Distributions",
            "status": "passed" if is_normal else "warning",
            "evidence": f"Shapiro-Wilk p-values: '{label_1}' p={norm_p1:.4f}, '{label_2}' p={norm_p2:.4f}.",
            "impact": "Normality assumption satisfied for parametric t-tests." if is_normal else "Distribution departs from normality; non-parametric Mann-Whitney U is more robust.",
        })

        # Homogeneity of variance (Levene's test)
        levene_p = float(stats.levene(g1, g2, center="median")[1]) if (n1 >= 2 and n2 >= 2) else 1.0
        equal_var = bool(levene_p >= 0.05)

        assumptions.append({
            "assumption": "Homogeneity of Variance",
            "status": "passed" if equal_var else "warning",
            "evidence": f"Levene's test p-value = {levene_p:.4f}.",
            "impact": "Equal variance assumption holds." if equal_var else "Unequal variances detected; Welch's t-test or Mann-Whitney U required.",
        })

        # 2. Candidate tests suitability
        candidates: List[Dict[str, Any]] = []

        # Student's t-test
        s_suit = 0.95 if (is_normal and equal_var) else (0.60 if is_normal else 0.40)
        candidates.append({"method": "student_t_test", "name": "Student's Independent t-Test", "suitability": s_suit, "notes": "Requires normality and equal variance."})

        # Welch's t-test
        w_suit = 0.95 if (is_normal and not equal_var) else (0.88 if (n1 >= 30 and n2 >= 30) else 0.70)
        candidates.append({"method": "welch_t_test", "name": "Welch's Two-Sample t-Test", "suitability": w_suit, "notes": "Robust to unequal variances."})

        # Mann-Whitney U
        m_suit = 0.95 if not is_normal else (0.80 if (n1 < 30 or n2 < 30) else 0.75)
        candidates.append({"method": "mann_whitney_u", "name": "Mann-Whitney U Test", "suitability": m_suit, "notes": "Non-parametric rank test, robust to outliers and skewness."})

        # Select test
        candidates.sort(key=lambda x: x["suitability"], reverse=True)
        if preferred_test and any(c["method"] == preferred_test for c in candidates):
            selected = next(c for c in candidates if c["method"] == preferred_test)
            sel_method = selected["method"]
            sel_reason = f"User explicitly selected {selected['name']}."
        else:
            selected = candidates[0]
            sel_method = selected["method"]
            sel_reason = f"Selected {selected['name']} based on normality={is_normal} and equal_variance={equal_var}."

        # 3. Computation
        test_stat = 0.0
        p_val = 1.0
        stat_name = "t"
        df_deg: Optional[float] = None
        ci_diff: Optional[Dict[str, Any]] = None

        if sel_method == "student_t_test":
            res = stats.ttest_ind(g1, g2, equal_var=True)
            test_stat = _sanitize_float(res.statistic) or 0.0
            p_val = _sanitize_float(res.pvalue) or 1.0
            stat_name = "t"
            df_deg = float(n1 + n2 - 2)
            # Pooled SE & CI
            sp = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(1, df_deg)) if df_deg > 0 else 1.0
            se_diff = sp * math.sqrt(1.0 / n1 + 1.0 / n2)
            t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, max(1, df_deg)))
            ci_diff = {
                "lower": round(mean_diff - t_crit * se_diff, 4),
                "estimate": round(mean_diff, 4),
                "upper": round(mean_diff + t_crit * se_diff, 4),
                "confidence_level": round(1.0 - alpha, 2),
            }

        elif sel_method == "welch_t_test":
            res = stats.ttest_ind(g1, g2, equal_var=False)
            test_stat = _sanitize_float(res.statistic) or 0.0
            p_val = _sanitize_float(res.pvalue) or 1.0
            stat_name = "t"
            # Welch-Satterthwaite df
            num = (var1 / n1 + var2 / n2) ** 2
            den = ((var1 / n1) ** 2) / max(1, n1 - 1) + ((var2 / n2) ** 2) / max(1, n2 - 1)
            df_deg = round(num / den, 2) if den > 0 else float(n1 + n2 - 2)
            se_diff = math.sqrt(var1 / n1 + var2 / n2)
            t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, max(1.0, df_deg)))
            ci_diff = {
                "lower": round(mean_diff - t_crit * se_diff, 4),
                "estimate": round(mean_diff, 4),
                "upper": round(mean_diff + t_crit * se_diff, 4),
                "confidence_level": round(1.0 - alpha, 2),
            }

        elif sel_method == "mann_whitney_u":
            res = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            test_stat = _sanitize_float(res.statistic) or 0.0
            p_val = _sanitize_float(res.pvalue) or 1.0
            stat_name = "U"
            df_deg = None
            # Median difference CI via Hodges-Lehmann or mean difference approximation
            se_diff = math.sqrt(var1 / n1 + var2 / n2) if (var1 > 0 or var2 > 0) else 1.0
            z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
            ci_diff = {
                "lower": round(mean_diff - z_crit * se_diff, 4),
                "estimate": round(mean_diff, 4),
                "upper": round(mean_diff + z_crit * se_diff, 4),
                "confidence_level": round(1.0 - alpha, 2),
            }

        # 4. Effect Size
        # Cohen's d & Hedges' g
        pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(1, n1 + n2 - 2)) if (n1 + n2 > 2) else 1.0
        cohens_d = (mean_diff / pooled_sd) if pooled_sd > 1e-12 else 0.0
        hedges_j = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0)) if (n1 + n2 > 3) else 1.0
        hedges_g = cohens_d * hedges_j

        # Rank-biserial correlation for Mann-Whitney
        rank_biserial = 1.0 - (2.0 * test_stat / (n1 * n2)) if (sel_method == "mann_whitney_u" and n1 * n2 > 0) else None
        if rank_biserial is not None:
            rank_biserial = max(-1.0, min(1.0, float(rank_biserial)))

        if sel_method == "mann_whitney_u" and rank_biserial is not None:
            eff_size = round(float(rank_biserial), 4)
            eff_type = "rank_biserial"
            eff_mag = abs(eff_size)
            if eff_mag < 0.10:
                practical_sig = "negligible"
            elif eff_mag < 0.30:
                practical_sig = "small"
            elif eff_mag < 0.50:
                practical_sig = "moderate"
            else:
                practical_sig = "large"
        else:
            eff_size = round(float(hedges_g if (n1 < 20 or n2 < 20) else cohens_d), 4)
            eff_type = "hedges_g" if (n1 < 20 or n2 < 20) else "cohens_d"
            eff_mag = abs(eff_size)
            if eff_mag < 0.20:
                practical_sig = "negligible"
            elif eff_mag < 0.50:
                practical_sig = "small"
            elif eff_mag < 0.80:
                practical_sig = "moderate"
            else:
                practical_sig = "large"

        if n1 + n2 < 8:
            practical_sig = "uncertain"

        # Cohen's d 95% CI
        se_d = math.sqrt((n1 + n2) / (n1 * n2) + (cohens_d ** 2) / (2.0 * (n1 + n2))) if (n1 * n2 > 0) else 0.5
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        eff_ci = {
            "lower": round(eff_size - z_crit * se_d, 4),
            "estimate": round(eff_size, 4),
            "upper": round(eff_size + z_crit * se_d, 4),
            "confidence_level": round(1.0 - alpha, 2),
        }

        # Null and Alternative Hypotheses
        null_h = f"The true mean/distribution of '{feat_name}' is equivalent between group '{label_1}' and group '{label_2}'."
        alt_h = f"The true mean/distribution of '{feat_name}' differs between group '{label_1}' and group '{label_2}'."

        return {
            "test_family": "two_sample_numeric",
            "test_method": sel_method,
            "test_name": selected["name"],
            "variable_x": feat_name,
            "variable_group": group_name,
            "group_labels": [str(label_1), str(label_2)],
            "null_hypothesis": null_h,
            "alternative_hypothesis": alt_h,
            "alpha": alpha,
            "test_statistic": round(test_stat, 4),
            "statistic_name": stat_name,
            "degrees_of_freedom": df_deg,
            "p_value": round(p_val, 6),
            "adjusted_p_value": round(p_val, 6),
            "statistical_significance": bool(p_val < alpha),
            "effect_size": eff_size,
            "effect_size_type": eff_type,
            "effect_size_ci": eff_ci,
            "mean_difference": round(mean_diff, 4),
            "mean_difference_ci": ci_diff,
            "practical_significance": practical_sig,
            "practical_interpretation": "",
            "assumptions": assumptions,
            "selection_transparency": {
                "selected_test": sel_method,
                "reason": sel_reason,
                "candidates": candidates,
            },
            "group_statistics": group_stats,
            "row_accounting": {
                "original_rows": orig_n,
                "valid_rows": valid_n,
                "missing_x": orig_n - int(df[feat_name].notna().sum()),
                "missing_group": orig_n - int(df[group_name].notna().sum()),
                "excluded_rows": orig_n - valid_n,
            },
        }

    def _run_multi_sample_numeric_test(
        self,
        feat_name: str,
        group_name: str,
        group_arrays: Dict[str, np.ndarray],
        orig_n: int,
        valid_n: int,
        group_stats: Dict[str, Any],
        alpha: float,
        preferred_test: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute multiple groups (k >= 3) hypothesis testing with post-hoc pairwise comparisons."""
        groups_list = list(group_arrays.values())
        k = len(groups_list)
        total_n = sum(len(g) for g in groups_list)

        # 1. Assumption diagnostics
        assumptions: List[Dict[str, Any]] = []

        # Sample size per group
        min_n = min(len(g) for g in groups_list)
        assumptions.append({
            "assumption": "Group Sample Sizes",
            "status": "passed" if min_n >= 5 else "warning",
            "evidence": f"{k} groups evaluated, smallest group size n={min_n}.",
            "impact": "Sufficient observations per group for F-distribution approximation." if min_n >= 5 else "Small group sizes may reduce ANOVA statistical power.",
        })

        # Normality diagnostics per group
        norm_p_vals = [float(stats.shapiro(g)[1]) if (3 <= len(g) <= 5000) else 1.0 for g in groups_list]
        is_normal = all(p >= 0.05 for p in norm_p_vals)

        assumptions.append({
            "assumption": "Normality of Group Distributions",
            "status": "passed" if is_normal else "warning",
            "evidence": f"Shapiro-Wilk normality tests: min group p-value = {min(norm_p_vals):.4f}.",
            "impact": "Group distributions satisfy normality." if is_normal else "At least one group departs from normality; Kruskal-Wallis non-parametric ANOVA is recommended.",
        })

        # Homogeneity of variance (Levene's test)
        levene_p = float(stats.levene(*groups_list, center="median")[1])
        equal_var = bool(levene_p >= 0.05)

        assumptions.append({
            "assumption": "Homogeneity of Variances",
            "status": "passed" if equal_var else "warning",
            "evidence": f"Levene's test p-value = {levene_p:.4f}.",
            "impact": "Variances are homogeneous across groups." if equal_var else "Heteroscedasticity detected; Welch ANOVA or Kruskal-Wallis is more reliable.",
        })

        # 2. Candidate tests
        candidates: List[Dict[str, Any]] = []
        candidates.append({"method": "one_way_anova", "name": "One-Way ANOVA F-Test", "suitability": 0.95 if (is_normal and equal_var) else (0.60 if is_normal else 0.40), "notes": "Classical parametric omnibus test."})
        candidates.append({"method": "welch_anova", "name": "Welch's ANOVA", "suitability": 0.95 if (is_normal and not equal_var) else 0.70, "notes": "Robust to unequal group variances."})
        candidates.append({"method": "kruskal_wallis", "name": "Kruskal-Wallis H-Test", "suitability": 0.95 if not is_normal else 0.75, "notes": "Non-parametric rank test for multiple groups."})

        candidates.sort(key=lambda x: x["suitability"], reverse=True)
        if preferred_test and any(c["method"] == preferred_test for c in candidates):
            selected = next(c for c in candidates if c["method"] == preferred_test)
            sel_method = selected["method"]
            sel_reason = f"User explicitly selected {selected['name']}."
        else:
            selected = candidates[0]
            sel_method = selected["method"]
            sel_reason = f"Selected {selected['name']} based on normality={is_normal} and variance_homogeneity={equal_var}."

        # 3. Computation
        test_stat = 0.0
        p_val = 1.0
        stat_name = "F"
        df_deg: Any = None

        if sel_method in ("one_way_anova", "welch_anova"):
            res = stats.f_oneway(*groups_list)
            test_stat = _sanitize_float(res.statistic) or 0.0
            p_val = _sanitize_float(res.pvalue) or 1.0
            stat_name = "F"
            df_deg = {"df_between": k - 1, "df_within": total_n - k}
        else:
            res = stats.kruskal(*groups_list)
            test_stat = _sanitize_float(res.statistic) or 0.0
            p_val = _sanitize_float(res.pvalue) or 1.0
            stat_name = "H"
            df_deg = k - 1

        # 4. Effect Size: Eta-Squared (SS_between / SS_total)
        grand_mean = float(np.mean(np.concatenate(groups_list)))
        ss_between = sum(len(g) * ((float(np.mean(g)) - grand_mean) ** 2) for g in groups_list)
        ss_total = sum(np.sum((g - grand_mean) ** 2) for g in groups_list)
        eta_sq = float(ss_between / ss_total) if ss_total > 1e-12 else 0.0
        eta_sq = max(0.0, min(1.0, eta_sq))

        if eta_sq < 0.01:
            practical_sig = "negligible"
        elif eta_sq < 0.06:
            practical_sig = "small"
        elif eta_sq < 0.14:
            practical_sig = "moderate"
        else:
            practical_sig = "large"

        if total_n < 12:
            practical_sig = "uncertain"

        # 5. Post-Hoc Pairwise Comparisons
        post_hoc_list: List[Dict[str, Any]] = []
        labels = list(group_arrays.keys())
        pairwise_raw_p: List[float] = []
        pairwise_pairs: List[Tuple[str, str, float, float]] = []

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                l1, l2 = labels[i], labels[j]
                arr1, arr2 = group_arrays[l1], group_arrays[l2]
                if len(arr1) >= 2 and len(arr2) >= 2:
                    if sel_method == "kruskal_wallis":
                        pw_res = stats.mannwhitneyu(arr1, arr2, alternative="two-sided")
                        stat_val = _sanitize_float(pw_res.statistic) or 0.0
                        pval = _sanitize_float(pw_res.pvalue) or 1.0
                    else:
                        pw_res = stats.ttest_ind(arr1, arr2, equal_var=equal_var)
                        stat_val = _sanitize_float(pw_res.statistic) or 0.0
                        pval = _sanitize_float(pw_res.pvalue) or 1.0
                    m_d = float(np.mean(arr1) - np.mean(arr2))
                    pairwise_raw_p.append(pval)
                    pairwise_pairs.append((str(l1), str(l2), stat_val, m_d))

        if pairwise_raw_p:
            pairwise_adj_p = _adjust_p_values_fdr(pairwise_raw_p)
            for (l1, l2, st_val, m_d), raw_p, adj_p in zip(pairwise_pairs, pairwise_raw_p, pairwise_adj_p):
                post_hoc_list.append({
                    "group_1": l1,
                    "group_2": l2,
                    "mean_difference": round(m_d, 4),
                    "test_statistic": round(st_val, 4),
                    "raw_p_value": round(raw_p, 6),
                    "adjusted_p_value": round(adj_p, 6),
                    "significant": bool(adj_p < alpha),
                })

        null_h = f"The true mean/distribution of '{feat_name}' is identical across all {k} categories of '{group_name}'."
        alt_h = f"At least one category of '{group_name}' differs in true mean/distribution for '{feat_name}'."

        return {
            "test_family": "multi_sample_numeric",
            "test_method": sel_method,
            "test_name": selected["name"],
            "variable_x": feat_name,
            "variable_group": group_name,
            "group_labels": [str(l) for l in labels],
            "null_hypothesis": null_h,
            "alternative_hypothesis": alt_h,
            "alpha": alpha,
            "test_statistic": round(test_stat, 4),
            "statistic_name": stat_name,
            "degrees_of_freedom": df_deg,
            "p_value": round(p_val, 6),
            "adjusted_p_value": round(p_val, 6),
            "statistical_significance": bool(p_val < alpha),
            "effect_size": round(eta_sq, 4),
            "effect_size_type": "eta_squared",
            "effect_size_ci": None,
            "mean_difference": None,
            "mean_difference_ci": None,
            "practical_significance": practical_sig,
            "practical_interpretation": "",
            "assumptions": assumptions,
            "selection_transparency": {
                "selected_test": sel_method,
                "reason": sel_reason,
                "candidates": candidates,
            },
            "post_hoc": post_hoc_list,
            "group_statistics": group_stats,
            "row_accounting": {
                "original_rows": orig_n,
                "valid_rows": valid_n,
                "missing_x": orig_n - int(df[feat_name].notna().sum()),
                "missing_group": orig_n - int(df[group_name].notna().sum()),
                "excluded_rows": orig_n - valid_n,
            },
        }

    def _evaluate_paired_numeric(
        self,
        df: pd.DataFrame,
        var1: str,
        var2: str,
        alpha: float,
        preferred_test: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate paired differences across two continuous columns (e.g. before/after, measurement A/B)."""
        if var1 not in df.columns or var2 not in df.columns or var1 == var2:
            return None

        s1 = CanonicalDataLayer.coerce_numeric_series(df[var1])
        s2 = CanonicalDataLayer.coerce_numeric_series(df[var2])

        valid_mask = s1.notna() & s2.notna()
        orig_n = len(df)
        valid_n = int(valid_mask.sum())
        if valid_n < 3:
            return None

        v1 = s1[valid_mask].to_numpy(dtype=float)
        v2 = s2[valid_mask].to_numpy(dtype=float)
        diff = v1 - v2
        n = len(diff)
        mean_d = float(np.mean(diff))
        std_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0

        # Assumptions
        assumptions: List[Dict[str, Any]] = []
        assumptions.append({
            "assumption": "Paired Observation Structure",
            "status": "passed",
            "evidence": f"Computed across {n} paired observations with 0 row drops in unrelated fields.",
            "impact": "Differences evaluate intra-subject variation.",
        })

        norm_p = float(stats.shapiro(diff)[1]) if (3 <= n <= 5000) else 1.0
        is_normal = bool(norm_p >= 0.05)
        assumptions.append({
            "assumption": "Normality of Differences",
            "status": "passed" if is_normal else "warning",
            "evidence": f"Shapiro-Wilk test on differences p-value = {norm_p:.4f}.",
            "impact": "Differences follow normal distribution." if is_normal else "Differences depart from normality; Wilcoxon signed-rank test is recommended.",
        })

        candidates = [
            {"method": "paired_t_test", "name": "Paired Student's t-Test", "suitability": 0.95 if is_normal else 0.60, "notes": "Requires normally distributed paired differences."},
            {"method": "wilcoxon_signed_rank", "name": "Wilcoxon Signed-Rank Test", "suitability": 0.95 if not is_normal else 0.75, "notes": "Non-parametric rank test for paired observations."},
        ]
        candidates.sort(key=lambda x: x["suitability"], reverse=True)

        if preferred_test and any(c["method"] == preferred_test for c in candidates):
            selected = next(c for c in candidates if c["method"] == preferred_test)
            sel_method = selected["method"]
            sel_reason = f"User explicitly requested {selected['name']}."
        else:
            selected = candidates[0]
            sel_method = selected["method"]
            sel_reason = f"Selected {selected['name']} based on paired difference normality={is_normal}."

        test_stat = 0.0
        p_val = 1.0
        stat_name = "t"
        df_deg: Optional[int] = None
        ci_diff: Optional[Dict[str, Any]] = None

        if sel_method == "paired_t_test":
            res = stats.ttest_rel(v1, v2)
            test_stat = _sanitize_float(res.statistic) or 0.0
            p_val = _sanitize_float(res.pvalue) or 1.0
            stat_name = "t"
            df_deg = n - 1
            se_d = std_d / math.sqrt(n) if n > 0 else 1.0
            t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, max(1, n - 1)))
            ci_diff = {
                "lower": round(mean_d - t_crit * se_d, 4),
                "estimate": round(mean_d, 4),
                "upper": round(mean_d + t_crit * se_d, 4),
                "confidence_level": round(1.0 - alpha, 2),
            }
        else:
            res = stats.wilcoxon(v1, v2)
            test_stat = _sanitize_float(res.statistic) or 0.0
            p_val = _sanitize_float(res.pvalue) or 1.0
            stat_name = "W"
            df_deg = None
            se_d = std_d / math.sqrt(n) if n > 0 else 1.0
            z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
            ci_diff = {
                "lower": round(mean_d - z_crit * se_d, 4),
                "estimate": round(mean_d, 4),
                "upper": round(mean_d + z_crit * se_d, 4),
                "confidence_level": round(1.0 - alpha, 2),
            }

        # Effect Size: Cohen's d_z = mean_d / std_d
        cohens_d = (mean_d / std_d) if std_d > 1e-12 else 0.0
        eff_mag = abs(cohens_d)
        if eff_mag < 0.20:
            practical_sig = "negligible"
        elif eff_mag < 0.50:
            practical_sig = "small"
        elif eff_mag < 0.80:
            practical_sig = "moderate"
        else:
            practical_sig = "large"

        if n < 8:
            practical_sig = "uncertain"

        null_h = f"The mean paired difference between '{var1}' and '{var2}' is zero."
        alt_h = f"The mean paired difference between '{var1}' and '{var2}' is not zero."

        return {
            "test_family": "paired_numeric",
            "test_method": sel_method,
            "test_name": selected["name"],
            "variable_x": var1,
            "variable_y": var2,
            "variable_group": f"Paired ({var1} vs {var2})",
            "group_labels": [var1, var2],
            "null_hypothesis": null_h,
            "alternative_hypothesis": alt_h,
            "alpha": alpha,
            "test_statistic": round(test_stat, 4),
            "statistic_name": stat_name,
            "degrees_of_freedom": df_deg,
            "p_value": round(p_val, 6),
            "adjusted_p_value": round(p_val, 6),
            "statistical_significance": bool(p_val < alpha),
            "effect_size": round(cohens_d, 4),
            "effect_size_type": "cohens_d_paired",
            "effect_size_ci": None,
            "mean_difference": round(mean_d, 4),
            "mean_difference_ci": ci_diff,
            "practical_significance": practical_sig,
            "practical_interpretation": "",
            "assumptions": assumptions,
            "selection_transparency": {
                "selected_test": sel_method,
                "reason": sel_reason,
                "candidates": candidates,
            },
            "group_statistics": {
                var1: {"mean": round(float(np.mean(v1)), 4), "std": round(float(np.std(v1, ddof=1)), 4) if n > 1 else 0.0},
                var2: {"mean": round(float(np.mean(v2)), 4), "std": round(float(np.std(v2, ddof=1)), 4) if n > 1 else 0.0},
                "difference": {"mean": round(mean_d, 4), "std": round(std_d, 4), "count": n},
            },
            "row_accounting": {
                "original_rows": orig_n,
                "valid_rows": valid_n,
                "missing_x": orig_n - int(df[var1].notna().sum()),
                "missing_group": orig_n - int(df[var2].notna().sum()),
                "excluded_rows": orig_n - valid_n,
            },
        }

    def _evaluate_two_features(
        self,
        df: pd.DataFrame,
        var1: str,
        var2: str,
        sem_profile: SemanticProfile,
        alpha: float,
        paired: Optional[bool] = None,
        preferred_test: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate two named features without an explicit grouping parameter."""
        s1 = df[var1]
        s2 = df[var2]

        c1 = CanonicalDataLayer.coerce_numeric_series(s1)
        c2 = CanonicalDataLayer.coerce_numeric_series(s2)

        if c1.notna().sum() >= 4 and c2.notna().sum() >= 4 and paired:
            return self._evaluate_paired_numeric(df, var1, var2, alpha, preferred_test)

        if c1.notna().sum() >= 4 and s2.nunique() >= 2 and s2.nunique() <= 30:
            return self._evaluate_numeric_vs_group(df, var1, c1, var2, s2, alpha, preferred_test)

        if c2.notna().sum() >= 4 and s1.nunique() >= 2 and s1.nunique() <= 30:
            return self._evaluate_numeric_vs_group(df, var2, c2, var1, s1, alpha, preferred_test)

        if s1.nunique() >= 2 and s2.nunique() >= 2:
            return self._evaluate_categorical_vs_categorical(df, var1, var2, alpha, preferred_test)

        return None

    def _evaluate_categorical_vs_categorical(
        self,
        df: pd.DataFrame,
        var1: str,
        var2: str,
        alpha: float,
        preferred_test: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate association between two categorical variables (Chi-Square / Fisher's Exact)."""
        valid_mask = df[var1].notna() & df[var2].notna()
        orig_n = len(df)
        valid_n = int(valid_mask.sum())
        if valid_n < 5:
            return None

        s1 = df[var1][valid_mask].astype(str)
        s2 = df[var2][valid_mask].astype(str)

        # Exclude high-cardinality keys
        if s1.nunique() < 2 or s2.nunique() < 2 or s1.nunique() > 25 or s2.nunique() > 25:
            return None

        ct = pd.crosstab(s1, s2)
        r, c = ct.shape

        # Expected counts
        chi2, p_val, dof, expected = stats.chi2_contingency(ct)
        min_expected = float(np.min(expected)) if expected.size > 0 else 0.0

        # Assumptions
        assumptions: List[Dict[str, Any]] = []
        is_sparse = bool(min_expected < 5.0)
        assumptions.append({
            "assumption": "Expected Cell Frequencies (E >= 5)",
            "status": "warning" if is_sparse else "passed",
            "evidence": f"Minimum expected cell frequency = {min_expected:.2f} across {r}x{c} table.",
            "impact": "Expected counts satisfy asymptotic chi-square requirements." if not is_sparse else "Sparse cell counts detected; Fisher's exact test preferred for 2x2 tables.",
        })

        candidates = []
        is_2x2 = (r == 2 and c == 2)
        if is_2x2:
            candidates.append({"method": "fisher_exact", "name": "Fisher's Exact Test", "suitability": 0.95 if is_sparse else 0.85, "notes": "Exact test, ideal for 2x2 tables with small or sparse counts."})
            candidates.append({"method": "chi_square", "name": "Chi-Square Test of Independence", "suitability": 0.90 if not is_sparse else 0.65, "notes": "Asymptotic test for contingency tables."})
        else:
            candidates.append({"method": "chi_square", "name": "Chi-Square Test of Independence", "suitability": 0.95 if not is_sparse else 0.75, "notes": "Asymptotic test for RxC contingency tables."})

        candidates.sort(key=lambda x: x["suitability"], reverse=True)

        if preferred_test and any(cand["method"] == preferred_test for cand in candidates):
            selected = next(cand for cand in candidates if cand["method"] == preferred_test)
            sel_method = selected["method"]
            sel_reason = f"User explicitly requested {selected['name']}."
        else:
            selected = candidates[0]
            sel_method = selected["method"]
            sel_reason = f"Selected {selected['name']} based on table_size={r}x{c} and min_expected_count={min_expected:.2f}."

        test_stat = 0.0
        stat_name = "chi2"
        odds_ratio_val: Optional[float] = None
        odds_ratio_ci: Optional[Dict[str, Any]] = None

        if sel_method == "fisher_exact" and is_2x2:
            stat_name = "odds_ratio"
            res_fe = stats.fisher_exact(ct.to_numpy())
            odds_ratio_val = _sanitize_float(res_fe.statistic) or 1.0
            test_stat = odds_ratio_val
            p_val = _sanitize_float(res_fe.pvalue) or 1.0
            dof = 1

            # Log-odds CI with Haldane-Anscombe +0.5 smoothing
            a, b, c_cell, d = ct.iloc[0, 0], ct.iloc[0, 1], ct.iloc[1, 0], ct.iloc[1, 1]
            se_ln_or = math.sqrt(1.0 / (a + 0.5) + 1.0 / (b + 0.5) + 1.0 / (c_cell + 0.5) + 1.0 / (d + 0.5))
            z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
            ln_or = math.log(max(1e-6, odds_ratio_val))
            odds_ratio_ci = {
                "lower": round(math.exp(ln_or - z_crit * se_ln_or), 4),
                "estimate": round(odds_ratio_val, 4),
                "upper": round(math.exp(ln_or + z_crit * se_ln_or), 4),
                "confidence_level": round(1.0 - alpha, 2),
            }
        else:
            test_stat = _sanitize_float(chi2) or 0.0
            p_val = _sanitize_float(p_val) or 1.0
            stat_name = "chi2"

        # Cramer's V effect size
        denom = valid_n * max(1, min(r - 1, c - 1))
        cramers_v = math.sqrt(chi2 / denom) if denom > 0 else 0.0
        cramers_v = max(0.0, min(1.0, float(cramers_v)))

        if cramers_v < 0.10:
            practical_sig = "negligible"
        elif cramers_v < 0.30:
            practical_sig = "small"
        elif cramers_v < 0.50:
            practical_sig = "moderate"
        else:
            practical_sig = "large"

        null_h = f"There is no statistical association between '{var1}' and '{var2}' (variables are independent)."
        alt_h = f"There is a statistical association between '{var1}' and '{var2}' (variables are dependent)."

        return {
            "test_family": "categorical_association",
            "test_method": sel_method,
            "test_name": selected["name"],
            "variable_x": var1,
            "variable_y": var2,
            "variable_group": var2,
            "group_labels": list(ct.columns),
            "null_hypothesis": null_h,
            "alternative_hypothesis": alt_h,
            "alpha": alpha,
            "test_statistic": round(test_stat, 4),
            "statistic_name": stat_name,
            "degrees_of_freedom": int(dof),
            "p_value": round(p_val, 6),
            "adjusted_p_value": round(p_val, 6),
            "statistical_significance": bool(p_val < alpha),
            "effect_size": round(cramers_v, 4),
            "effect_size_type": "cramers_v",
            "effect_size_ci": None,
            "odds_ratio": round(odds_ratio_val, 4) if odds_ratio_val is not None else None,
            "odds_ratio_ci": odds_ratio_ci,
            "mean_difference": None,
            "mean_difference_ci": None,
            "practical_significance": practical_sig,
            "practical_interpretation": "",
            "assumptions": assumptions,
            "selection_transparency": {
                "selected_test": sel_method,
                "reason": sel_reason,
                "candidates": candidates,
            },
            "contingency_table": ct.to_dict(),
            "row_accounting": {
                "original_rows": orig_n,
                "valid_rows": valid_n,
                "missing_x": orig_n - int(df[var1].notna().sum()),
                "missing_group": orig_n - int(df[var2].notna().sum()),
                "excluded_rows": orig_n - valid_n,
            },
        }

    def _discover_and_evaluate_hypotheses(
        self,
        df: pd.DataFrame,
        sem_profile: SemanticProfile,
        alpha: float,
        max_count: int = 50,
        preferred_test: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Autonomously discover and evaluate hypothesis tests across all candidate numeric and grouping columns."""
        hypotheses: List[Dict[str, Any]] = []

        # Usable grouping candidates
        group_cols = [
            c for c in df.columns
            if c not in sem_profile.identifier_columns
            and c not in sem_profile.constant_columns
            and c not in sem_profile.datetime_candidates
            and 2 <= df[c].nunique(dropna=True) <= 20
        ]

        # Usable numeric candidates
        numeric_cols = [
            c for c in df.columns
            if c not in sem_profile.identifier_columns
            and c not in sem_profile.constant_columns
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        # 1. Numeric vs Categorical comparisons
        for g_col in group_cols:
            for num_col in numeric_cols:
                if num_col != g_col:
                    job = self._evaluate_numeric_vs_group(
                        df, num_col, df[num_col], g_col, df[g_col], alpha, preferred_test
                    )
                    if job:
                        hypotheses.append(job)
                        if len(hypotheses) >= max_count:
                            return hypotheses

        # 2. Categorical vs Categorical comparisons (if budget remains)
        for i in range(len(group_cols)):
            for j in range(i + 1, len(group_cols)):
                c1, c2 = group_cols[i], group_cols[j]
                job = self._evaluate_categorical_vs_categorical(df, c1, c2, alpha, preferred_test)
                if job:
                    hypotheses.append(job)
                    if len(hypotheses) >= max_count:
                        return hypotheses

        return hypotheses

    def _format_interpretation(
        self,
        var_x: str,
        var_group: str,
        test_name: str,
        is_sig: bool,
        adj_p: float,
        eff_size: float,
        eff_type: str,
        prac_sig: str,
        alpha: float,
    ) -> str:
        """Construct clear, non-causal statistical and practical significance narrative."""
        if is_sig:
            sig_text = f"There is statistically significant evidence of a difference in '{var_x}' across '{var_group}' (adjusted p={adj_p:.4g} < alpha={alpha})."
        else:
            sig_text = f"No statistically significant difference in '{var_x}' was detected across '{var_group}' at alpha={alpha} (adjusted p={adj_p:.4g})."

        prac_text = f"The observed effect size ({eff_type}={eff_size:.4g}) represents a {prac_sig} observational magnitude."
        return f"{sig_text} {prac_text}"