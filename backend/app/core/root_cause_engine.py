"""Root-Cause & Counterfactual Decomposition Engine (What-If Scenario Modeling).

Provides mathematical variance decomposition (Volume, Price, Mix, and Segment effects)
to explain why KPIs changed across time/cohorts, and provides deterministic Counterfactual
"What-If" simulation modeling with sensitivity and elasticity analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class SegmentContribution:
    """Individual segment's contribution to total metric variance."""
    segment_name: str
    baseline_value: float
    comparison_value: float
    absolute_delta: float
    percentage_delta: float
    share_of_total_delta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_name": str(self.segment_name),
            "baseline_value": round(float(self.baseline_value), 2),
            "comparison_value": round(float(self.comparison_value), 2),
            "absolute_delta": round(float(self.absolute_delta), 2),
            "percentage_delta": round(float(self.percentage_delta), 2),
            "share_of_total_delta": round(float(self.share_of_total_delta), 2),
        }


@dataclass
class VarianceDecompositionReport:
    """Mathematical decomposition of KPI variance between two time periods or cohorts."""
    metric_name: str
    dimension_name: str
    baseline_total: float
    comparison_total: float
    total_delta: float
    total_delta_pct: float
    volume_effect: float
    price_rate_effect: float
    mix_effect: float
    top_positive_drivers: List[SegmentContribution]
    top_negative_drivers: List[SegmentContribution]
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "dimension_name": self.dimension_name,
            "baseline_total": round(float(self.baseline_total), 2),
            "comparison_total": round(float(self.comparison_total), 2),
            "total_delta": round(float(self.total_delta), 2),
            "total_delta_pct": round(float(self.total_delta_pct), 2),
            "volume_effect": round(float(self.volume_effect), 2),
            "price_rate_effect": round(float(self.price_rate_effect), 2),
            "mix_effect": round(float(self.mix_effect), 2),
            "top_positive_drivers": [d.to_dict() for d in self.top_positive_drivers],
            "top_negative_drivers": [d.to_dict() for d in self.top_negative_drivers],
            "duration_ms": round(float(self.duration_ms), 3),
        }


@dataclass
class CounterfactualSimulationResult:
    """Deterministic output of a What-If scenario simulation."""
    scenario_description: str
    target_metric: str
    lever_feature: str
    transformation_applied: str
    baseline_value: float
    simulated_value: float
    absolute_impact: float
    percentage_impact: float
    segment_sensitivities: Dict[str, float]
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_description": self.scenario_description,
            "target_metric": self.target_metric,
            "lever_feature": self.lever_feature,
            "transformation_applied": self.transformation_applied,
            "baseline_value": round(float(self.baseline_value), 2),
            "simulated_value": round(float(self.simulated_value), 2),
            "absolute_impact": round(float(self.absolute_impact), 2),
            "percentage_impact": round(float(self.percentage_impact), 2),
            "segment_sensitivities": {k: round(float(v), 2) for k, v in self.segment_sensitivities.items()},
            "duration_ms": round(float(self.duration_ms), 3),
        }


class RootCauseDecompositionEngine:
    """Engine for Root-Cause Variance Decomposition and What-If Counterfactual Analysis."""

    def decompose_variance(
        self,
        df: pd.DataFrame,
        metric: str,
        dimension: Optional[str] = None,
        date_col: Optional[str] = None,
        volume_col: Optional[str] = None,
    ) -> VarianceDecompositionReport:
        """
        Decompose KPI variance into Volume, Rate, Mix, and Segment drivers.
        Splits data chronologically or into top cohorts if no dates present.
        """
        start_t = time.time()
        
        # 1. Identify Dimension and Metric
        if metric not in df.columns:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            metric = num_cols[0] if num_cols else df.columns[0]

        if not dimension or dimension not in df.columns:
            cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
            dimension = cat_cols[0] if cat_cols else (df.columns[1] if len(df.columns) > 1 else df.columns[0])

        # 2. Split into Baseline and Comparison cohorts
        if date_col and date_col in df.columns:
            sorted_df = df.sort_values(by=date_col)
            mid = len(sorted_df) // 2
            df_base = sorted_df.iloc[:mid]
            df_comp = sorted_df.iloc[mid:]
        else:
            mid = len(df) // 2
            df_base = df.iloc[:mid]
            df_comp = df.iloc[mid:]

        base_total = float(df_base[metric].sum()) if len(df_base) > 0 else 0.0
        comp_total = float(df_comp[metric].sum()) if len(df_comp) > 0 else 0.0
        total_delta = comp_total - base_total
        delta_pct = (total_delta / base_total * 100.0) if base_total != 0 else 0.0

        # 3. Volume and Price / Rate Decomposition
        # If explicit volume column present, use standard bridge math: Delta = Volume_Effect + Price_Effect + Mix_Effect
        if volume_col and volume_col in df.columns and volume_col != metric:
            v_base = float(df_base[volume_col].sum()) or 1.0
            v_comp = float(df_comp[volume_col].sum()) or 1.0
            p_base = base_total / v_base
            p_comp = comp_total / v_comp

            vol_effect = (v_comp - v_base) * p_base
            price_effect = (p_comp - p_base) * v_comp
            mix_effect = total_delta - (vol_effect + price_effect)
        else:
            # Synthetic volume vs rate split based on transaction counts
            v_base = len(df_base) or 1
            v_comp = len(df_comp) or 1
            avg_base = base_total / v_base
            avg_comp = comp_total / v_comp

            vol_effect = (v_comp - v_base) * avg_base
            price_effect = (avg_comp - avg_base) * v_comp
            mix_effect = total_delta - (vol_effect + price_effect)

        # 4. Segment Attribution Breakdown
        base_by_seg = df_base.groupby(dimension, observed=True)[metric].sum()
        comp_by_seg = df_comp.groupby(dimension, observed=True)[metric].sum()
        all_segs = sorted(list(set(base_by_seg.index).union(set(comp_by_seg.index))))

        contributions: List[SegmentContribution] = []
        for seg in all_segs:
            b_val = float(base_by_seg.get(seg, 0.0))
            c_val = float(comp_by_seg.get(seg, 0.0))
            d_val = c_val - b_val
            d_pct = (d_val / b_val * 100.0) if b_val != 0 else (100.0 if c_val > 0 else 0.0)
            share_total = (d_val / total_delta * 100.0) if total_delta != 0 else 0.0

            contributions.append(SegmentContribution(
                segment_name=str(seg),
                baseline_value=b_val,
                comparison_value=c_val,
                absolute_delta=d_val,
                percentage_delta=d_pct,
                share_of_total_delta=share_total,
            ))

        # Sort positive and negative drivers
        pos_drivers = sorted([c for c in contributions if c.absolute_delta > 0], key=lambda x: x.absolute_delta, reverse=True)
        neg_drivers = sorted([c for c in contributions if c.absolute_delta < 0], key=lambda x: x.absolute_delta)

        duration = (time.time() - start_t) * 1000

        return VarianceDecompositionReport(
            metric_name=metric,
            dimension_name=dimension,
            baseline_total=base_total,
            comparison_total=comp_total,
            total_delta=total_delta,
            total_delta_pct=delta_pct,
            volume_effect=vol_effect,
            price_rate_effect=price_effect,
            mix_effect=mix_effect,
            top_positive_drivers=pos_drivers[:5],
            top_negative_drivers=neg_drivers[:5],
            duration_ms=duration,
        )

    def simulate_what_if(
        self,
        df: pd.DataFrame,
        target_metric: str,
        lever_feature: str,
        percentage_change: float = 10.0,
        dimension: Optional[str] = None,
    ) -> CounterfactualSimulationResult:
        """
        Simulate a counterfactual scenario where a lever feature is modified by a given percentage.
        Calculates resulting impact on the target metric and segment-level elasticity.
        """
        start_t = time.time()
        sim_df = df.copy()

        if target_metric not in sim_df.columns:
            num_cols = sim_df.select_dtypes(include=[np.number]).columns.tolist()
            target_metric = num_cols[0] if num_cols else sim_df.columns[0]

        if lever_feature not in sim_df.columns:
            num_cols = sim_df.select_dtypes(include=[np.number]).columns.tolist()
            lever_feature = num_cols[1] if len(num_cols) > 1 else num_cols[0]

        base_val = float(sim_df[target_metric].sum())

        # Determine relationship / elasticity between lever and target
        # e.g., if lever is Discount, increasing discount usually reduces revenue/profit
        # If lever is Price/Revenue itself, direct multiplier
        if lever_feature == target_metric:
            factor = 1.0 + (percentage_change / 100.0)
            sim_df[target_metric] = sim_df[target_metric] * factor
            trans = f"Direct {percentage_change:+.1f}% shift to {target_metric}"
        elif "discount" in lever_feature.lower():
            # Higher discount -> lower margin/revenue
            delta_disc = percentage_change / 100.0
            sim_df[lever_feature] = sim_df[lever_feature] * (1.0 + delta_disc)
            sim_df[target_metric] = sim_df[target_metric] * (1.0 - (delta_disc * 0.5))
            trans = f"Simulated {percentage_change:+.1f}% adjustment to {lever_feature} (with -0.5x margin drag)"
        elif "cost" in lever_feature.lower() and "profit" in target_metric.lower():
            delta_cost = (percentage_change / 100.0)
            cost_shift = sim_df[lever_feature] * delta_cost
            sim_df[target_metric] = sim_df[target_metric] - cost_shift
            trans = f"Shifted {lever_feature} by {percentage_change:+.1f}%"
        else:
            # Proportional elasticity model
            corr = float(sim_df[[lever_feature, target_metric]].dropna().corr().iloc[0, 1]) if len(sim_df) > 1 else 1.0
            if np.isnan(corr):
                corr = 0.5
            elasticity = corr
            factor = 1.0 + (percentage_change / 100.0 * elasticity)
            sim_df[target_metric] = sim_df[target_metric] * factor
            trans = f"Adjusted {lever_feature} by {percentage_change:+.1f}% (model elasticity: {elasticity:.2f})"

        sim_val = float(sim_df[target_metric].sum())
        abs_impact = sim_val - base_val
        pct_impact = (abs_impact / base_val * 100.0) if base_val != 0 else 0.0

        # Segment sensitivities
        segment_sens = {}
        if dimension and dimension in df.columns:
            base_by_seg = df.groupby(dimension, observed=True)[target_metric].sum()
            sim_by_seg = sim_df.groupby(dimension, observed=True)[target_metric].sum()
            for seg in base_by_seg.index:
                b_s = float(base_by_seg[seg])
                s_s = float(sim_by_seg.get(seg, 0.0))
                segment_sens[str(seg)] = (s_s - b_s)

        duration = (time.time() - start_t) * 1000

        desc = f"Simulating {percentage_change:+.1f}% change in '{lever_feature}' on target '{target_metric}'"

        return CounterfactualSimulationResult(
            scenario_description=desc,
            target_metric=target_metric,
            lever_feature=lever_feature,
            transformation_applied=trans,
            baseline_value=base_val,
            simulated_value=sim_val,
            absolute_impact=abs_impact,
            percentage_impact=pct_impact,
            segment_sensitivities=segment_sens,
            duration_ms=duration,
        )


# Global singleton instance
global_root_cause_engine = RootCauseDecompositionEngine()
