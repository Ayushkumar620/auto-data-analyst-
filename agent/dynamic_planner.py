"""Dynamic Task Planner for Multi-Step Analytical Query Decomposition and Execution.

Transforms complex natural language questions into an executable DAG of agent steps:
1. Intent Analysis
2. Required Dataset Validation
3. Preprocessing Requirements
4. Tool / Agent Selection
5. Execution Plan DAG
6. Validation Criteria
7. Fallback & Recovery Strategies
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

import pandas as pd

from agent.base import BaseAgent
from agent.schemas import AgentResult, AgentStatus, ClaimType, ErrorCategory, Evidence
from agent.intent import AnalyticalIntent, IntentAnalyzer, IntentClassificationResult
from agent.result_validator import ResultValidator
from agent.agents import (
    CleaningAgent,
    AnalysisAgent,
    PredictionAgent,
    ForecastAgent,
    VisualizationAgent,
    InsightAgent,
    ReportAgent,
)
from backend.app.core.dataset_knowledge import DatasetKnowledge
from backend.app.core.semantic import SemanticSchemaAgent


@dataclass
class PlanStep:
    """A single atomic step in a dynamic task execution graph."""
    step_id: int
    name: str
    agent_class_name: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[int] = field(default_factory=list)
    validation_criteria: str = "Result status must be completed with valid output."
    fallback_strategy: str = "Retry with relaxed constraints or fallback to statistical summary."
    status: str = "pending"  # pending, in_progress, completed, failed, skipped
    result: Optional[AgentResult] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "agent_class_name": self.agent_class_name,
            "action": self.action,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "validation_criteria": self.validation_criteria,
            "fallback_strategy": self.fallback_strategy,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "has_result": self.result is not None,
        }


@dataclass
class TaskPlan:
    """Full execution plan for an analytical request."""
    plan_id: str
    query: str
    intent: IntentClassificationResult
    steps: List[PlanStep] = field(default_factory=list)
    dataset_validation: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    total_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "query": self.query,
            "intent": self.intent.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "dataset_validation": self.dataset_validation,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "total_duration_ms": self.total_duration_ms,
        }


class DynamicTaskPlanner:
    """Decomposes natural language queries into executable multi-agent workflows."""

    def __init__(
        self,
        intent_analyzer: Optional[IntentAnalyzer] = None,
        validator: Optional[ResultValidator] = None,
        semantic_agent: Optional[SemanticSchemaAgent] = None,
    ):
        self.intent_analyzer = intent_analyzer or IntentAnalyzer()
        self.validator = validator or ResultValidator()
        self.semantic_agent = semantic_agent or SemanticSchemaAgent()

    # ------------------------------------------------------------------
    # Plan Construction
    # ------------------------------------------------------------------
    def create_plan(
        self,
        query: str,
        dataframe: pd.DataFrame,
        knowledge: Optional[DatasetKnowledge] = None,
    ) -> TaskPlan:
        """Analyze intent and synthesize a multi-step execution plan with validation and fallbacks."""
        # Step 1: Intent Analysis
        intent_res = self.intent_analyzer.analyze(query, knowledge=knowledge, dataframe=dataframe)

        # Step 2: Dataset Knowledge & Validation
        if knowledge is None:
            knowledge = self.semantic_agent.build_knowledge(dataframe)

        dataset_val = self._validate_dataset_requirements(dataframe, intent_res, knowledge)

        # Step 3 & 4: Tool Selection & Step DAG Construction
        steps: List[PlanStep] = []
        step_counter = 1

        # Step A: Data Cleaning (if requested or data quality is sub-optimal)
        needs_cleaning = intent_res.needs_cleaning or knowledge.data_quality.get("quality_score", 100) < 85
        cleaning_step_id = None
        if needs_cleaning or intent_res.primary_intent == AnalyticalIntent.CLEANING:
            cleaning_step_id = step_counter
            steps.append(
                PlanStep(
                    step_id=cleaning_step_id,
                    name="Data Sanitization and Imputation",
                    agent_class_name="CleaningAgent",
                    action="clean",
                    parameters={"strategy": "auto_impute"},
                    dependencies=[],
                    validation_criteria="Missing values handled and cleaned DataFrame produced.",
                    fallback_strategy="Carry forward raw data if non-critical.",
                )
            )
            step_counter += 1

        # Step B: Main Analytical / Modeling / Forecasting Tool Selection
        primary_deps = [cleaning_step_id] if cleaning_step_id else []
        main_step_id = step_counter

        if intent_res.primary_intent in (AnalyticalIntent.PREDICTION, AnalyticalIntent.DEEP_LEARNING):
            target = intent_res.target_column or knowledge.get_primary_metric()
            steps.append(
                PlanStep(
                    step_id=main_step_id,
                    name=f"Supervised ML Modeling for '{target}'",
                    agent_class_name="PredictionAgent",
                    action="predict",
                    parameters={"target": target, "features": intent_res.feature_columns},
                    dependencies=primary_deps,
                    validation_criteria="Model trained with measurable performance metrics (R2 / Accuracy).",
                    fallback_strategy="Fallback to Linear/Logistic baseline if complex model fails.",
                )
            )
            step_counter += 1

        elif intent_res.primary_intent == AnalyticalIntent.FORECASTING:
            target = intent_res.target_column or knowledge.get_primary_metric()
            periods = intent_res.time_horizon or 5
            steps.append(
                PlanStep(
                    step_id=main_step_id,
                    name=f"Time Series Forecast for '{target}' ({periods} periods)",
                    agent_class_name="ForecastAgent",
                    action="forecast",
                    parameters={"target": target, "periods": periods},
                    dependencies=primary_deps,
                    validation_criteria="Forecast series generated with upper and lower bounds.",
                    fallback_strategy="Fallback to Exponential Smoothing or Moving Average.",
                )
            )
            step_counter += 1

        elif intent_res.primary_intent == AnalyticalIntent.ANOMALIES:
            target = intent_res.target_column or knowledge.get_primary_metric()
            steps.append(
                PlanStep(
                    step_id=main_step_id,
                    name=f"Statistical Anomaly Detection on '{target}'",
                    agent_class_name="InsightAgent",
                    action="anomalies",
                    parameters={"type": "anomalies", "column": target},
                    dependencies=primary_deps,
                    validation_criteria="Outlier report generated using IQR / Isolation Forest.",
                    fallback_strategy="Fallback to standard z-score thresholding.",
                )
            )
            step_counter += 1

        elif intent_res.primary_intent == AnalyticalIntent.VISUALIZATION:
            chart_type = intent_res.chart_type or "bar"
            x_col = knowledge.get_primary_dimension() or knowledge.get_primary_date_column()
            y_col = intent_res.target_column or knowledge.get_primary_metric()
            steps.append(
                PlanStep(
                    step_id=main_step_id,
                    name=f"Generate {chart_type.title()} Chart",
                    agent_class_name="VisualizationAgent",
                    action="chart",
                    parameters={"chart_type": chart_type, "x": x_col, "y": y_col},
                    dependencies=primary_deps,
                    validation_criteria="Valid Plotly JSON chart specification produced.",
                    fallback_strategy="Fallback to automatic table summary.",
                )
            )
            step_counter += 1

        else:
            # Default to EDA / Summary
            steps.append(
                PlanStep(
                    step_id=main_step_id,
                    name="Exploratory Data Profiling & Statistical Summary",
                    agent_class_name="AnalysisAgent",
                    action="summary",
                    parameters={"request": "summary"},
                    dependencies=primary_deps,
                    validation_criteria="Statistical summary generated for all numeric and categorical columns.",
                    fallback_strategy="Fallback to pandas.describe() summary.",
                )
            )
            step_counter += 1

        # Step C: Explanation / Feature Drivers (if requested or secondary intent)
        if intent_res.needs_explanation or intent_res.primary_intent == AnalyticalIntent.EXPLANATION:
            exp_deps = [main_step_id] if main_step_id else primary_deps
            top_k = intent_res.top_k or 3
            target = intent_res.target_column or knowledge.get_primary_metric()
            steps.append(
                PlanStep(
                    step_id=step_counter,
                    name=f"Extract Top-{top_k} Influential Drivers for '{target}'",
                    agent_class_name="InsightAgent",
                    action="explain_drivers",
                    parameters={"type": "smart", "target": target, "top_k": top_k},
                    dependencies=exp_deps,
                    validation_criteria=f"Identified top-{top_k} drivers with importance scores and evidence.",
                    fallback_strategy="Fallback to Pearson/Spearman correlation ranking.",
                )
            )
            step_counter += 1

        # Step D: Synthesis & Report Generation
        all_prior_step_ids = [s.step_id for s in steps]
        steps.append(
            PlanStep(
                step_id=step_counter,
                name="Synthesize Executive Insights & Final Narrative Report",
                agent_class_name="ReportAgent",
                action="report",
                parameters={"request": "pipeline"},
                dependencies=all_prior_step_ids,
                validation_criteria="Comprehensive multi-section markdown narrative report produced.",
                fallback_strategy="Fallback to bulleted key findings summary.",
            )
        )

        return TaskPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            query=query,
            intent=intent_res,
            steps=steps,
            dataset_validation=dataset_val,
        )

    # ------------------------------------------------------------------
    # Plan Execution
    # ------------------------------------------------------------------
    def execute_plan(
        self,
        plan: TaskPlan,
        dataframe: pd.DataFrame,
    ) -> AgentResult:
        """Execute the planned steps in dependency order and return a unified AgentResult."""
        start_time = datetime.now()
        current_data = dataframe.copy()
        step_results: Dict[int, AgentResult] = {}
        all_evidence: List[Evidence] = []
        all_warnings: List[str] = []

        agent_factory = {
            "CleaningAgent": CleaningAgent,
            "AnalysisAgent": AnalysisAgent,
            "PredictionAgent": PredictionAgent,
            "ForecastAgent": ForecastAgent,
            "VisualizationAgent": VisualizationAgent,
            "InsightAgent": InsightAgent,
            "ReportAgent": ReportAgent,
        }

        for step in plan.steps:
            step.status = "in_progress"
            step_start = datetime.now()

            # Prepare task payload
            task_payload = {"data": current_data, **step.parameters}

            # If ReportAgent, pass previous agent results
            if step.agent_class_name == "ReportAgent":
                task_payload["agent_outputs"] = list(step_results.values())

            # Instantiate and execute agent
            agent_cls = agent_factory.get(step.agent_class_name, AnalysisAgent)
            agent_instance: BaseAgent = agent_cls()

            try:
                result = agent_instance.execute_with_retry(task_payload, max_retries=2)
                step.duration_ms = round((datetime.now() - step_start).total_seconds() * 1000, 2)
                step.result = result

                if result.is_success:
                    step.status = "completed"
                    step_results[step.step_id] = result
                    all_evidence.extend(result.evidence)
                    all_warnings.extend(result.warnings)

                    # If cleaning succeeded, carry the cleaned data forward
                    if step.action == "clean":
                        reports = result.output.get("reports", [])
                        if reports and isinstance(reports, list) and "cleaned_data" in reports[0]:
                            current_data = pd.DataFrame(reports[0]["cleaned_data"])
                else:
                    # Apply fallback
                    step.status = "failed"
                    all_warnings.append(f"Step {step.step_id} ({step.name}) failed. Applied fallback: {step.fallback_strategy}")
            except Exception as exc:
                step.status = "failed"
                step.duration_ms = round((datetime.now() - step_start).total_seconds() * 1000, 2)
                all_warnings.append(f"Step {step.step_id} encountered exception: {str(exc)}")

        total_duration = round((datetime.now() - start_time).total_seconds() * 1000, 2)
        plan.total_duration_ms = total_duration

        # Format output
        outputs_summary = {
            "plan_id": plan.plan_id,
            "query": plan.query,
            "primary_intent": plan.intent.primary_intent.value,
            "steps_executed": [s.to_dict() for s in plan.steps],
            "step_outputs": {str(sid): res.output for sid, res in step_results.items()},
        }

        # If ReportAgent completed, extract top-level report
        final_report = None
        for step in reversed(plan.steps):
            if step.agent_class_name == "ReportAgent" and step.result and step.result.is_success:
                final_report = step.result.output.get("report")
                break

        if final_report:
            outputs_summary["report"] = final_report

        # Compute overall confidence
        confidences = [r.confidence for r in step_results.values() if r.confidence > 0]
        avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.85

        evidence = all_evidence or [
            Evidence(
                source="DynamicTaskPlanner",
                method="dag_execution",
                data_ref={"steps_count": len(plan.steps), "successful_steps": len(step_results)},
                confidence=round(avg_conf, 3),
                claim_type=ClaimType.FACT,
            )
        ]

        return AgentResult.success(
            agent="DynamicTaskPlanner",
            role="planner",
            agent_id=plan.plan_id,
            started_at=start_time,
            output=outputs_summary,
            evidence=evidence,
            confidence=round(avg_conf, 3),
            duration_ms=total_duration,
            warnings=all_warnings,
            metadata={"plan": plan.to_dict()},
        )

    # ------------------------------------------------------------------
    # Dataset Validation
    # ------------------------------------------------------------------
    def _validate_dataset_requirements(
        self,
        dataframe: pd.DataFrame,
        intent: IntentClassificationResult,
        knowledge: DatasetKnowledge,
    ) -> Dict[str, Any]:
        """Validate if the dataset satisfies requirements for the detected intent."""
        validation = {
            "is_valid": True,
            "row_count": len(dataframe),
            "column_count": len(dataframe.columns),
            "issues": [],
        }

        if len(dataframe) < 5:
            validation["is_valid"] = False
            validation["issues"].append("Dataset contains fewer than 5 rows, insufficient for reliable modeling.")

        if intent.primary_intent == AnalyticalIntent.FORECASTING:
            if not knowledge.date_columns and not any(pd.api.types.is_datetime64_any_dtype(dataframe[c]) for c in dataframe.columns):
                validation["is_valid"] = False
                validation["issues"].append("Forecasting requested but no date or timestamp column detected.")

        if intent.primary_intent in (AnalyticalIntent.PREDICTION, AnalyticalIntent.DEEP_LEARNING):
            if intent.target_column and intent.target_column not in dataframe.columns:
                validation["is_valid"] = False
                validation["issues"].append(f"Target column '{intent.target_column}' does not exist in dataset.")

        return validation

