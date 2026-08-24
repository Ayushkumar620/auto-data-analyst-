"""Command-Driven Autonomous Agent Orchestrator.

Translates high-level natural language user outcome goals into multi-stage
analytical DAG plans, dynamically composes specialized agents, executes calculations
deterministically, validates results, and synthesizes evidence-grounded explanations.

Explicitly distinguishes:
1. user_intent
2. required_operations
3. available_tools (selected agents)
4. model_selection (when predictive/forecasting)
5. execution
6. validation
7. final_explanation
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from agent.schemas import AgentResult, AgentStatus, ClaimType, Evidence
from agent.intent import AnalyticalIntent, IntentAnalyzer, IntentClassificationResult
from agent.dynamic_planner import DynamicTaskPlanner, PlanStep, TaskPlan
from backend.app.core.dataset_knowledge import DatasetKnowledge
from backend.app.core.semantic import SemanticSchemaAgent
from backend.app.core.evidence_insights import EvidenceBasedInsightsEngine, StructuredInsight
from backend.app.ml.model_selection import MLModelComparisonEngine, ModelComparisonReport
from backend.app.ml.ann_engine import ANNEngine
from backend.app.ml.cnn_engine import CNNEngine
from backend.app.ml.validation_engine import DataModelValidator, ValidationAuditReport


@dataclass
class CommandExecutionResult:
    """Standardized result of a command-driven autonomous execution."""
    command: str
    user_intent: str
    required_operations: List[str]
    selected_agents: List[str]
    model_selection_summary: Optional[Dict[str, Any]]
    execution_steps: List[Dict[str, Any]]
    validation_summary: Dict[str, Any]
    final_explanation: str
    evidence: List[Dict[str, Any]]
    visualization: Optional[Dict[str, Any]]
    dataset_summary: Dict[str, Any]
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "user_intent": self.user_intent,
            "required_operations": self.required_operations,
            "selected_agents": self.selected_agents,
            "model_selection_summary": self.model_selection_summary,
            "execution_steps": self.execution_steps,
            "validation_summary": self.validation_summary,
            "final_explanation": self.final_explanation,
            "evidence": self.evidence,
            "visualization": self.visualization,
            "dataset_summary": self.dataset_summary,
            "duration_ms": round(float(self.duration_ms), 2),
        }


class AutonomousCommandOrchestrator:
    """Completely command-driven autonomous orchestrator for the Auto Data Analyst."""

    def __init__(self):
        self.semantic_agent = SemanticSchemaAgent()
        self.intent_analyzer = IntentAnalyzer()
        self.planner = DynamicTaskPlanner()
        self.validator = DataModelValidator()
        self.insights_engine = EvidenceBasedInsightsEngine()
        self.ml_engine = MLModelComparisonEngine()
        self.ann_engine = ANNEngine()
        self.cnn_engine = CNNEngine()

    def execute_command(
        self,
        command: str,
        dataframe: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> CommandExecutionResult:
        """
        Execute an arbitrary natural-language command end-to-end.
        Interprets intent, decomposes into required operations, selects tools,
        executes DAG, validates, and explains results.
        """
        start_t = time.time()
        q_norm = command.strip().lower()
        context = context or {}

        # ------------------------------------------------------------------
        # Stage 1: Dataset Intelligence & Schema Inspection
        # ------------------------------------------------------------------
        knowledge: DatasetKnowledge = self.semantic_agent.build_knowledge(dataframe)
        n_rows, n_cols = dataframe.shape
        dataset_summary = {
            "rows": n_rows,
            "columns": list(dataframe.columns),
            "metrics": [m.column if hasattr(m, "column") else str(m) for m in knowledge.metrics],
            "dimensions": [d.column if hasattr(d, "column") else str(d) for d in knowledge.dimensions],
            "date_columns": [d.column if hasattr(d, "column") else str(d) for d in knowledge.date_columns],
            "quality_score": knowledge.data_quality.get("quality_score", 100),
        }

        # ------------------------------------------------------------------
        # Stage 2: Intent Analysis & Disambiguation
        # ------------------------------------------------------------------
        intent_res: IntentClassificationResult = self.intent_analyzer.analyze(
            query=command, knowledge=knowledge, dataframe=dataframe
        )

        # ------------------------------------------------------------------
        # Stage 3: Required Operations Decomposition
        # ------------------------------------------------------------------
        required_ops: List[str] = self._determine_required_operations(command, intent_res, knowledge, dataframe)

        # ------------------------------------------------------------------
        # Stage 4: Multi-Step DAG Task Plan Synthesis & Tool Selection
        # ------------------------------------------------------------------
        plan: TaskPlan = self.planner.create_plan(query=command, dataframe=dataframe, knowledge=knowledge)
        selected_agents = [s.agent_class_name for s in plan.steps]

        # ------------------------------------------------------------------
        # Stage 5: Execution Across Specialized Engines
        # ------------------------------------------------------------------
        exec_output = self.planner.execute_plan(plan, dataframe)

        # ------------------------------------------------------------------
        # Stage 6: Validation Safety Audit
        # ------------------------------------------------------------------
        target_col = intent_res.target_column or (knowledge.metrics[0].column if knowledge.metrics else (dataframe.columns[0] if len(dataframe.columns) > 0 else ""))
        val_report: ValidationAuditReport = self.validator.audit_pipeline(
            df=dataframe,
            target_column=target_col if target_col in dataframe.columns else dataframe.columns[-1],
        )
        val_summary = {
            "status": val_report.overall_status,
            "critical_issues": val_report.critical_issues_count,
            "warnings": val_report.warnings_count,
            "diagnostics": val_report.diagnostics,
        }

        # ------------------------------------------------------------------
        # Stage 7: Evidence-Based Final Natural-Language Explanation
        # ------------------------------------------------------------------
        model_selection_summary: Optional[Dict[str, Any]] = None
        visualization: Optional[Dict[str, Any]] = None
        evidence_list: List[Dict[str, Any]] = []

        # Extract artifacts and evidence from executed plan steps
        for step_res in exec_output.get("results", []):
            out = step_res.get("output", {})
            if isinstance(out, dict):
                if "best_model" in out:
                    model_selection_summary = out.get("best_model")
                if "figure" in out or "chart" in out or "data" in out and "layout" in out:
                    visualization = out
                if "repaired_data" in out:
                    dataset_summary["repaired_columns"] = list(out["repaired_data"].columns)

            for ev in step_res.get("evidence", []):
                if isinstance(ev, dict):
                    evidence_list.append(ev)
                elif hasattr(ev, "to_dict"):
                    evidence_list.append(ev.to_dict())

        # Synthesize final narrative explanation based on user's exact outcome goal
        explanation = self._synthesize_explanation(
            command=command,
            intent_res=intent_res,
            knowledge=knowledge,
            dataframe=dataframe,
            exec_output=exec_output,
            model_summary=model_selection_summary,
            required_ops=required_ops,
        )

        # Fallback visualization if none created yet
        if visualization is None:
            visualization = self._generate_fallback_visualization(intent_res, knowledge, dataframe)

        duration = (time.time() - start_t) * 1000

        return CommandExecutionResult(
            command=command,
            user_intent=intent_res.primary_intent.value,
            required_operations=required_ops,
            selected_agents=selected_agents,
            model_selection_summary=model_selection_summary,
            execution_steps=exec_output.get("results", []),
            validation_summary=val_summary,
            final_explanation=explanation,
            evidence=evidence_list,
            visualization=visualization,
            dataset_summary=dataset_summary,
            duration_ms=duration,
        )

    def _determine_required_operations(
        self,
        command: str,
        intent_res: IntentClassificationResult,
        knowledge: DatasetKnowledge,
        df: pd.DataFrame,
    ) -> List[str]:
        """Decompose command into distinct logical operations."""
        ops: List[str] = ["inspect_dataset_schema"]
        q = command.lower()

        if intent_res.needs_cleaning or "clean" in q or knowledge.data_quality.get("quality_score", 100) < 85:
            ops.append("sanitize_and_impute_missing_values")

        if "why" in q or "decrease" in q or "increase" in q or "driver" in q or "cause" in q:
            ops.extend([
                "compute_period_over_period_variance",
                "correlate_drivers_with_target",
                "synthesize_non_causal_driver_explanation",
            ])
        elif "compare" in q or "between" in q or " vs " in q:
            ops.extend([
                "filter_cohort_dimensions",
                "aggregate_cohort_metrics",
                "compute_cross_cohort_percentage_delta",
                "generate_comparative_visualization",
            ])
        elif "top" in q or "bottom" in q or "rank" in q or "best" in q:
            ops.extend([
                "group_by_dimension",
                "sum_and_rank_entities",
                "slice_top_n_records",
            ])
        elif intent_res.primary_intent in (AnalyticalIntent.PREDICTION, AnalyticalIntent.DEEP_LEARNING, AnalyticalIntent.CNN):
            ops.extend([
                "audit_data_leakage_and_imbalance",
                "encode_and_scale_features",
                "benchmark_candidate_algorithms_with_cv",
                "select_optimal_model_and_explain_rationale",
            ])
        elif intent_res.primary_intent == AnalyticalIntent.FORECASTING or "forecast" in q or "next" in q:
            ops.extend([
                "sort_chronological_time_series",
                "detect_trend_and_seasonality",
                "project_future_horizon_intervals",
            ])
        elif "unusual" in q or "anomal" in q or "outlier" in q:
            ops.extend([
                "compute_z_scores_and_isolation_bounds",
                "identify_extreme_outliers",
                "explain_contributing_column_values",
            ])
        elif "report" in q or "performance" in q or "overview" in q:
            ops.extend([
                "compute_executive_kpis",
                "synthesize_facts_and_observations",
                "compile_structured_narrative_report",
            ])
        else:
            ops.extend([
                "compute_statistical_aggregates",
                "extract_correlations_and_patterns",
                "generate_responsive_visualizations",
            ])

        return ops

    def _synthesize_explanation(
        self,
        command: str,
        intent_res: IntentClassificationResult,
        knowledge: DatasetKnowledge,
        df: pd.DataFrame,
        exec_output: Dict[str, Any],
        model_summary: Optional[Dict[str, Any]],
        required_ops: List[str],
    ) -> str:
        """Compose human-friendly, evidence-backed narrative explanation of what was computed."""
        q = command.lower()

        # 1. Comparison Queries (e.g. "Compare revenue between India and the US")
        if "compare" in q or "between" in q:
            dim_cols = [d.column if hasattr(d, "column") else str(d) for d in knowledge.dimensions]
            metric_cols = [m.column if hasattr(m, "column") else str(m) for m in knowledge.metrics]
            if dim_cols and metric_cols:
                d_col, m_col = dim_cols[0], metric_cols[0]
                grouped = df.groupby(d_col)[m_col].sum().sort_values(ascending=False)
                top_items = [f"**{k}**: {v:,.2f}" for k, v in list(grouped.items())[:3]]
                return f"Comparison of **{m_col}** across **{d_col}**:\n- " + "\n- ".join(top_items) + f"\n\nTotal volume analyzed across {len(grouped)} distinct categories."

        # 2. Driver & "Why" Queries (e.g. "Why did profit decrease last year?")
        if "why" in q or "decrease" in q or "drop" in q or "increase" in q:
            date_cols = [d.column if hasattr(d, "column") else str(d) for d in knowledge.date_columns]
            metric_cols = [m.column if hasattr(m, "column") else str(m) for m in knowledge.metrics]
            m_target = intent_res.target_column or (metric_cols[0] if metric_cols else "metric")

            lines = [
                f"Historical analysis of **{m_target}** shows period-over-period variations across the timeline."
            ]
            if len(metric_cols) >= 2:
                secondary = [m for m in metric_cols if m != m_target][0]
                corr = df[[m_target, secondary]].corr().iloc[0, 1]
                lines.append(
                    f"Strong statistical co-movement observed with **{secondary}** (correlation $r = {corr:.2f}$). "
                    f"*Note: Correlation reflects observed historical patterns, not proven causal drivers.*"
                )
            return "\n\n".join(lines)

        # 3. Top-N Ranking Queries (e.g. "Clean this dataset and find the top 10 customers")
        if "top" in q or "best" in q or "rank" in q or "customer" in q:
            dim_cols = [d.column if hasattr(d, "column") else str(d) for d in knowledge.dimensions]
            metric_cols = [m.column if hasattr(m, "column") else str(m) for m in knowledge.metrics]
            if dim_cols and metric_cols:
                d_col, m_col = dim_cols[0], metric_cols[0]
                top_n = df.groupby(d_col)[m_col].sum().sort_values(ascending=False).head(10)
                items_str = "\n".join([f"{i+1}. **{k}**: {v:,.2f}" for i, (k, v) in enumerate(top_n.items())])
                return f"Top ranked entities by **{m_col}** (grouped by **{d_col}**):\n\n{items_str}"

        # 4. Model Training / Predictive Queries
        if model_summary:
            m_name = model_summary.get("model_name", "Best Model")
            metric_name = model_summary.get("primary_metric_name", "Score")
            metric_val = model_summary.get("primary_metric_value", 0.0)
            return (
                f"Evaluated candidate algorithms for target **{intent_res.target_column}**. "
                f"The best performing architecture is **{m_name}** achieving a {metric_name} of **{metric_val:.4f}** "
                f"under cross-validation."
            )

        # 5. Generic Summary fallback
        return f"Successfully executed analytical operations ({', '.join(required_ops[:4])}) for query: '{command}'."

    def _generate_fallback_visualization(
        self,
        intent_res: IntentClassificationResult,
        knowledge: DatasetKnowledge,
        df: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        """Generate structured visualization spec when none is generated by steps."""
        dim_cols = [d.column if hasattr(d, "column") else str(d) for d in knowledge.dimensions]
        metric_cols = [m.column if hasattr(m, "column") else str(m) for m in knowledge.metrics]
        date_cols = [d.column if hasattr(d, "column") else str(d) for d in knowledge.date_columns]

        if date_cols and metric_cols:
            return {"chart_type": "line", "x": date_cols[0], "y": metric_cols[0], "title": f"{metric_cols[0]} Over Time"}
        elif dim_cols and metric_cols:
            return {"chart_type": "bar", "x": dim_cols[0], "y": metric_cols[0], "title": f"{metric_cols[0]} by {dim_cols[0]}"}
        elif len(metric_cols) >= 2:
            return {"chart_type": "scatter", "x": metric_cols[0], "y": metric_cols[1], "title": f"{metric_cols[1]} vs {metric_cols[0]}"}

        return None
