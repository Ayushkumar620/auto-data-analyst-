"""
Analysis Discovery Agent for Autonomous Analytics.

Inspects DatasetKnowledge and UserIntent, evaluates data viability,
prioritizes eligible analytical candidates, and generates a structured ExecutionPlan.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from agent.autonomous_analysis_schemas import (
    AnalysisCandidate,
    AnalysisDepth,
    AutonomousAnalysisRequest,
)
from agent.base import BaseAgent
from agent.dataset_knowledge import DatasetKnowledge
from agent.intent import UserIntent
from agent.schemas import AgentResult, AgentStatus, ClaimType, Evidence


class AnalysisDiscoveryAgent(BaseAgent):
    """
    Autonomous Discovery Engine that evaluates dataset capabilities and user intent
    to determine WHAT analyses need to happen and in WHAT priority order.
    """
    name = "Analysis Discovery Agent"
    role = "lead_data_discovery_analyst"
    description = "Discovers, prioritizes, and plans evidence-based analyses tailored to dataset structure and user intent."

    def __init__(self, data: Optional[Any] = None):
        super().__init__(data=data)

    def discover_analyses(
        self,
        df: pd.DataFrame,
        knowledge: Optional[DatasetKnowledge] = None,
        user_intent: Optional[UserIntent] = None,
        depth: AnalysisDepth = AnalysisDepth.STANDARD,
        max_steps: int = 10,
    ) -> List[AnalysisCandidate]:
        """
        Evaluate dataset properties and user intent to identify and rank candidate analyses.
        """
        candidates: List[AnalysisCandidate] = []
        n_rows = len(df)
        cols = list(df.columns)

        # 1. Column Role Extraction
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(include=["object", "category", "string", "str"]).columns)
        date_cols = []
        for c in cols:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_cols.append(c)
            elif "date" in c.lower() or "time" in c.lower() or "year" in c.lower() or "month" in c.lower():
                try:
                    pd.to_datetime(df[c].dropna().head(10))
                    date_cols.append(c)
                except Exception:
                    pass

        # Cross-reference with knowledge if available
        if knowledge:
            num_cols = knowledge.numerical_columns or num_cols
            cat_cols = knowledge.categorical_columns or cat_cols
            date_cols = knowledge.date_columns or date_cols

        intent_metrics = user_intent.metrics if user_intent else []
        intent_dims = user_intent.dimensions if user_intent else []
        intent_text = (user_intent.original_command or user_intent.objective or "").lower() if user_intent else ""

        is_why_question = any(q in intent_text for q in ("why", "driver", "cause", "reason", "decline", "drop", "fell", "decrease"))

        # ----------------------------------------------------------------------
        # Rule 1: Data Quality & Profiling
        # ----------------------------------------------------------------------
        if n_rows > 0:
            candidates.append(
                AnalysisCandidate(
                    analysis_type="data_quality",
                    objective="Assess dataset completeness, missingness, duplicates, and health quality.",
                    required_inputs=cols[:10],
                    priority=1,
                    expected_value=0.90,
                    computational_cost="low",
                    confidence=0.98,
                    reason="Fundamental baseline required to establish evidentiary reliability.",
                )
            )

        # ----------------------------------------------------------------------
        # Rule 2: Descriptive Statistics
        # ----------------------------------------------------------------------
        if num_cols:
            target_nums = [c for c in intent_metrics if c in num_cols] or num_cols[:5]
            candidates.append(
                AnalysisCandidate(
                    analysis_type="descriptive_statistics",
                    objective=f"Compute central tendencies, dispersion, and statistical distributions for {target_nums}.",
                    required_inputs=target_nums,
                    priority=2,
                    expected_value=0.88,
                    computational_cost="low",
                    confidence=0.95,
                    reason="Identifies numerical ranges, means, medians, and spreads.",
                )
            )

        # ----------------------------------------------------------------------
        # Rule 3: Trend & Temporal Analysis
        # ----------------------------------------------------------------------
        if date_cols and num_cols:
            primary_date = date_cols[0]
            primary_metric = intent_metrics[0] if intent_metrics and intent_metrics[0] in num_cols else num_cols[0]
            candidates.append(
                AnalysisCandidate(
                    analysis_type="trend_analysis",
                    objective=f"Track temporal progression, growth rates, and seasonality for '{primary_metric}' across '{primary_date}'.",
                    required_inputs=[primary_date, primary_metric],
                    priority=3 if not is_why_question else 1,
                    expected_value=0.95,
                    computational_cost="medium",
                    confidence=0.92,
                    reason="Temporal patterns reveal acceleration, decline, and seasonal peaks.",
                )
            )

        # ----------------------------------------------------------------------
        # Rule 4: Segmentation & Group Breakdown
        # ----------------------------------------------------------------------
        if cat_cols and num_cols:
            primary_dim = intent_dims[0] if intent_dims and intent_dims[0] in cat_cols else cat_cols[0]
            primary_metric = intent_metrics[0] if intent_metrics and intent_metrics[0] in num_cols else num_cols[0]
            candidates.append(
                AnalysisCandidate(
                    analysis_type="segmentation",
                    objective=f"Segment '{primary_metric}' by '{primary_dim}' to identify top performers and disparity.",
                    required_inputs=[primary_dim, primary_metric],
                    priority=4 if not is_why_question else 2,
                    expected_value=0.92,
                    computational_cost="medium",
                    confidence=0.93,
                    reason="Isolates key performing subgroups, category shares, and structural imbalances.",
                )
            )

        # ----------------------------------------------------------------------
        # Rule 5: Correlation Analysis
        # ----------------------------------------------------------------------
        if len(num_cols) >= 2:
            candidates.append(
                AnalysisCandidate(
                    analysis_type="correlation_analysis",
                    objective=f"Evaluate statistical associations and co-movements across {num_cols[:6]}.",
                    required_inputs=num_cols[:6],
                    priority=5,
                    expected_value=0.82,
                    computational_cost="low",
                    confidence=0.90,
                    reason="Reveals relationships and multi-collinearity across numerical indicators.",
                )
            )

        # ----------------------------------------------------------------------
        # Rule 6: Anomaly & Outlier Detection
        # ----------------------------------------------------------------------
        if num_cols and n_rows >= 15:
            target_metric = intent_metrics[0] if intent_metrics and intent_metrics[0] in num_cols else num_cols[0]
            candidates.append(
                AnalysisCandidate(
                    analysis_type="anomaly_detection",
                    objective=f"Detect statistical outliers and anomalous spikes in '{target_metric}' using IQR / Z-score.",
                    required_inputs=[target_metric],
                    priority=6 if not is_why_question else 3,
                    expected_value=0.85,
                    computational_cost="medium",
                    confidence=0.88,
                    reason="Flags rare events and extreme observations that skew aggregate metrics.",
                )
            )

        # ----------------------------------------------------------------------
        # Rule 7: Concentration & Pareto Analysis
        # ----------------------------------------------------------------------
        if (cat_cols or len(cols) > 1) and num_cols and n_rows >= 10:
            target_dim = cat_cols[0] if cat_cols else cols[0]
            target_metric = intent_metrics[0] if intent_metrics and intent_metrics[0] in num_cols else num_cols[0]
            candidates.append(
                AnalysisCandidate(
                    analysis_type="concentration_analysis",
                    objective=f"Quantify revenue/metric concentration and Pareto 80/20 distribution for '{target_metric}' across '{target_dim}'.",
                    required_inputs=[target_dim, target_metric],
                    priority=7,
                    expected_value=0.84,
                    computational_cost="low",
                    confidence=0.91,
                    reason="Measures reliance on top accounts, categories, or high-volume entities.",
                )
            )

        # ----------------------------------------------------------------------
        # Rule 8: Root Cause / Business Driver Investigation
        # ----------------------------------------------------------------------
        if is_why_question and num_cols and (cat_cols or date_cols):
            target_metric = intent_metrics[0] if intent_metrics and intent_metrics[0] in num_cols else num_cols[0]
            candidates.insert(
                0,
                AnalysisCandidate(
                    analysis_type="business_driver_investigation",
                    objective=f"Investigate underlying segment drivers and contributing factors for '{target_metric}' shifts.",
                    required_inputs=[target_metric] + (cat_cols[:2] if cat_cols else []) + (date_cols[:1] if date_cols else []),
                    priority=1,
                    expected_value=0.98,
                    computational_cost="medium",
                    confidence=0.90,
                    reason="Explicit root cause inquiry requested by user command.",
                )
            )

        # ----------------------------------------------------------------------
        # Sorting & Depth Limits
        # ----------------------------------------------------------------------
        candidates.sort(key=lambda c: (c.priority, -c.expected_value))

        depth_limits = {
            AnalysisDepth.QUICK: 4,
            AnalysisDepth.STANDARD: 7,
            AnalysisDepth.DEEP: max_steps,
        }
        limit = min(depth_limits.get(depth, 7), max_steps)
        return candidates[:limit]

    def create_plan_from_candidates(
        self,
        candidates: List[AnalysisCandidate],
        df: pd.DataFrame,
    ) -> Any:
        """Construct DAG ExecutionPlan from discovered analysis candidates."""
        from agent.dynamic_planner import ExecutionPlan, ExecutionStep
        steps: List[ExecutionStep] = []
        step_idx = 1

        for cand in candidates:
            step_id = f"step_{step_idx}"
            upstream = [f"step_{step_idx - 1}"] if step_idx > 1 and cand.analysis_type != "data_quality" else []
            steps.append(
                ExecutionStep(
                    step_id=step_id,
                    tool_name="autonomous_analyst",
                    agent_name="AutonomousAnalystAgent",
                    purpose=cand.objective,
                    inputs={
                        "analysis_type": cand.analysis_type,
                        "required_inputs": cand.required_inputs,
                    },
                    required_capabilities=[cand.analysis_type],
                    dependencies=upstream,
                )
            )
            step_idx += 1

        return ExecutionPlan(
            plan_id=f"plan_discovery_{step_idx}",
            user_intent="Autonomous Data Exploration and Insight Discovery",
            steps=steps,
            is_valid=True,
            estimated_runtime=float(len(steps) * 0.25),
        )

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """Execute discovery run returning prioritized candidates and ExecutionPlan."""
        self._start()
        df = task.get("data", self.data)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return self._error("No data provided for analysis discovery.", category="INPUT_VALIDATION")

        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception as exc:
                return self._error(f"Cannot convert input data to DataFrame: {str(exc)}", category="INPUT_VALIDATION")

        knowledge = task.get("knowledge")
        user_intent = task.get("user_intent")
        depth = AnalysisDepth(task.get("analysis_depth", "standard"))
        max_steps = int(task.get("max_analysis_steps", 10))

        candidates = self.discover_analyses(
            df=df,
            knowledge=knowledge,
            user_intent=user_intent,
            depth=depth,
            max_steps=max_steps,
        )
        plan = self.create_plan_from_candidates(candidates, df)

        evidence = [
            Evidence(
                source="AnalysisDiscoveryAgent",
                method="rule_based_capability_discovery",
                data_ref={
                    "total_candidates": len(candidates),
                    "selected_analyses": [c.analysis_type for c in candidates],
                    "analysis_depth": depth.value,
                },
                confidence=0.95,
                claim_type=ClaimType.FACT,
            )
        ]

        return self._finish(
            result={
                "candidates": [c.to_dict() for c in candidates],
                "plan": plan.model_dump(),
            },
            evidence=evidence,
            confidence=0.95,
            metadata={"candidate_count": len(candidates), "depth": depth.value},
        )
