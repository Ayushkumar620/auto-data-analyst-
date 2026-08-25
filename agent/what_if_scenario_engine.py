"""
What-If Counterfactual Scenario Engine.

Executes deterministic scenario simulations (percentage adjustments, segment shocks,
fixed value shifts, optimistic/expected/pessimistic multi-scenario matrices) with strict
epistemic non-causal attribution protection and mathematical evidence tracking.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from agent.forecasting_schemas import (
    ScenarioComparison,
    ScenarioResult,
    WhatIfRequest,
)
from agent.schemas import ClaimType, Evidence


class WhatIfScenarioEngine:
    """
    Simulates counterfactual What-If scenarios against datasets and baseline models.
    """

    def simulate_scenario(self, request: WhatIfRequest) -> ScenarioResult:
        """Run single scenario simulation."""
        df = request.dataset
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return ScenarioResult(
                scenario_name=request.scenario_name,
                target_metric=request.target or "target",
                baseline_value=0.0,
                scenario_value=0.0,
                absolute_difference=0.0,
                percentage_difference=0.0,
                limitations=["No active dataset available to simulate counterfactual scenarios."],
            )

        target_col = request.target
        if not target_col or target_col not in df.columns:
            num_cols = df.select_dtypes(include=[np.number]).columns
            target_col = num_cols[0] if len(num_cols) > 0 else "metric"

        baseline_val = float(df[target_col].sum()) if target_col in df.columns else 100.0

        # Apply Perturbation
        sim_df = df.copy()
        scenario_val = baseline_val
        assumptions = list(request.assumptions)
        assumptions.append("All unperturbed features and external environmental factors remain constant.")

        changed_vars = request.changed_variables or {}

        # 1. Segment-Specific Shock (e.g. {"segment": "North", "pct": 0.15, "dimension": "region"})
        if "segment" in changed_vars:
            seg_name = changed_vars.get("segment")
            pct = float(changed_vars.get("pct", 0.0))
            dim_col = changed_vars.get("dimension")

            # Locate dimension column if not specified
            if not dim_col:
                for c in df.select_dtypes(include=["object", "category", "string"]).columns:
                    if seg_name in df[c].values:
                        dim_col = c
                        break

            if dim_col and dim_col in df.columns and seg_name:
                mask = sim_df[dim_col] == seg_name
                sim_df.loc[mask, target_col] = sim_df.loc[mask, target_col] * (1.0 + pct)
                scenario_val = float(sim_df[target_col].sum())
                assumptions.append(f"Segment '{seg_name}' in dimension '{dim_col}' adjusted by {pct * 100:+.1f}%.")
            else:
                scenario_val = baseline_val * (1.0 + pct)
                assumptions.append(f"Global adjustment of {pct * 100:+.1f}% applied across segments.")

        # 2. Direct Target Percentage Shift (e.g. {"pct": 0.10} or {"target_pct": 0.10})
        elif "pct" in changed_vars:
            pct = float(changed_vars["pct"])
            scenario_val = baseline_val * (1.0 + pct)
            assumptions.append(f"Target metric '{target_col}' adjusted by {pct * 100:+.1f}%.")

        # 3. Variable Elasticity / Co-movement (e.g. {"price": 0.10})
        elif changed_vars:
            var_name, change_spec = next(iter(changed_vars.items()))
            if isinstance(change_spec, (int, float)):
                pct = float(change_spec)
            elif isinstance(change_spec, dict):
                pct = float(change_spec.get("pct", change_spec.get("value", 0.0)))
            else:
                pct = 0.0

            # If lever variable exists in dataframe
            if var_name in df.columns and pd.api.types.is_numeric_dtype(df[var_name]):
                sim_df[var_name] = sim_df[var_name] * (1.0 + pct)
                # Compute empirical correlation to determine target elasticity
                corr = float(df[var_name].corr(df[target_col])) if len(df) > 1 else 1.0
                if np.isnan(corr):
                    corr = 1.0
                target_pct_impact = pct * corr
                scenario_val = baseline_val * (1.0 + target_pct_impact)
                assumptions.append(f"Lever variable '{var_name}' shifted by {pct * 100:+.1f}% with empirical elasticity factor of {corr:.2f}.")
            else:
                scenario_val = baseline_val * (1.0 + pct)
                assumptions.append(f"Hypothetical variable shift of {pct * 100:+.1f}% applied.")

        abs_diff = scenario_val - baseline_val
        pct_diff = (abs_diff / baseline_val * 100.0) if baseline_val != 0 else 0.0

        limitations = [
            "Simulation is based on learned predictive co-movement and does not constitute proven causal intervention.",
            "General equilibrium effects, competitor reactions, and nonlinear saturation are not modeled.",
        ]

        evidence_obj = Evidence(
            source="WhatIfScenarioEngine.simulation",
            method="counterfactual_elasticity_simulation",
            confidence=0.90,
            claim_type=ClaimType.INFERENCE,
            computation_details={
                "scenario_name": request.scenario_name,
                "baseline_value": baseline_val,
                "scenario_value": scenario_val,
                "absolute_diff": abs_diff,
                "percentage_diff": pct_diff,
            },
        )

        return ScenarioResult(
            scenario_name=request.scenario_name,
            target_metric=target_col,
            baseline_value=baseline_val,
            scenario_value=scenario_val,
            absolute_difference=abs_diff,
            percentage_difference=pct_diff,
            assumptions=assumptions,
            limitations=limitations,
            evidence=[evidence_obj],
            confidence=0.90,
        )

    def compare_scenarios(
        self,
        df: pd.DataFrame,
        target: Optional[str],
        scenarios_spec: Dict[str, Dict[str, Any]],
    ) -> ScenarioComparison:
        """Run and rank multiple What-If scenarios (e.g. Optimistic, Expected, Pessimistic)."""
        results: List[ScenarioResult] = []
        all_evidence: List[Evidence] = []
        
        target_col = target
        if not target_col or target_col not in df.columns:
            num_cols = df.select_dtypes(include=[np.number]).columns
            target_col = num_cols[0] if len(num_cols) > 0 else "target"

        baseline_val = float(df[target_col].sum()) if target_col in df.columns else 100.0

        for name, params in scenarios_spec.items():
            req = WhatIfRequest(
                dataset=df,
                target=target_col,
                scenario_name=name,
                changed_variables=params,
            )
            res = self.simulate_scenario(req)
            results.append(res)
            all_evidence.extend(res.evidence)

        # Rank scenarios from highest to lowest simulated outcome
        ranked = sorted(results, key=lambda s: s.scenario_value, reverse=True)

        summary = (
            f"Evaluated {len(results)} scenarios for target '{target}' against baseline ({baseline_val:,.2f}). "
            f"Best outcome: '{ranked[0].scenario_name}' ({ranked[0].scenario_value:,.2f}, {ranked[0].percentage_difference:+.1f}%), "
            f"Worst outcome: '{ranked[-1].scenario_name}' ({ranked[-1].scenario_value:,.2f}, {ranked[-1].percentage_difference:+.1f}%)."
        )

        return ScenarioComparison(
            target_metric=target,
            baseline_value=baseline_val,
            scenarios=results,
            ranked_scenarios=ranked,
            summary=summary,
            evidence=all_evidence,
        )
