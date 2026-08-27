"""
Universal Agent Orchestrator & End-to-End Autonomous Analytical Execution Engine.

Single source of truth for:
1. Natural-Language Command & Intent Interpretation
2. Dataset-Aware Semantic Planning (via CanonicalDataLayer & SemanticProfile)
3. Structured AnalyticalPlan & Task DAG Generation
4. Topological Dependency Resolution & Concurrent Level Execution
5. Per-Task PreExecutionValidator Auditing & Error Isolation
6. Registry-Driven Tool/Agent Invocation (via DEFAULT_TOOL_REGISTRY)
7. Failure Classification, Bounded Retries & Graceful Partial Success
8. Multi-Agent Result, Metrics, Warnings & Evidence Aggregation
9. Principled Composite Confidence Calculation [0.0, 1.0]
10. Canonical AgentResult Contract with Traceback Containment
"""
from __future__ import annotations

import concurrent.futures
from datetime import datetime
from enum import Enum
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

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
from agent.intent import AnalyticalIntent, CommandIntelligenceAgent, IntentAnalyzer, IntentClassificationResult, IntentType
from agent.pre_execution_validator import PreExecutionValidator
from agent.result_validator import ResultValidator


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class PlanTask(BaseModel):
    """Atomic analytical task inside an AnalyticalPlan DAG."""
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    task_type: str  # eda, regression, classification, forecasting, anomaly_detection, clustering, statistical_analysis, hypothesis_testing, transformation, aggregation
    tool_name: str
    agent_name: str = ""
    purpose: str = ""
    required_columns: List[str] = Field(default_factory=list)
    optional_columns: List[str] = Field(default_factory=list)
    target_column: Optional[str] = None
    time_column: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    priority: int = 1
    status: str = TaskStatus.PENDING.value
    retry_policy: Dict[str, Any] = Field(default_factory=lambda: {"max_retries": 1, "backoff": 0.05})
    result: Optional[Dict[str, Any]] = None
    agent_result: Optional[AgentResult] = None
    duration_ms: float = 0.0
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "tool_name": self.tool_name,
            "agent_name": self.agent_name,
            "purpose": self.purpose,
            "required_columns": self.required_columns,
            "optional_columns": self.optional_columns,
            "target_column": self.target_column,
            "time_column": self.time_column,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "status": self.status,
            "retry_policy": self.retry_policy,
            "duration_ms": self.duration_ms,
            "has_result": self.result is not None or self.agent_result is not None,
            "error": self.error,
        }


class AnalyticalPlan(BaseModel):
    """Structured analytical plan generated from user command and dataset profile."""
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    user_request: str
    dataset_id: Optional[str] = None
    detected_intent: str = "eda"
    primary_intent: Optional[str] = None
    secondary_intents: List[str] = Field(default_factory=list)
    tasks: List[PlanTask] = Field(default_factory=list)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    execution_order: List[str] = Field(default_factory=list)
    requested_columns: List[str] = Field(default_factory=list)
    inferred_columns: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    ambiguity_information: Optional[Dict[str, Any]] = None
    is_ambiguous: bool = False
    is_unsupported: bool = False
    unsupported_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "user_request": self.user_request,
            "dataset_id": self.dataset_id,
            "detected_intent": self.detected_intent,
            "primary_intent": self.primary_intent,
            "secondary_intents": self.secondary_intents,
            "tasks": [t.to_dict() for t in self.tasks],
            "dependencies": self.dependencies,
            "execution_order": self.execution_order,
            "requested_columns": self.requested_columns,
            "inferred_columns": self.inferred_columns,
            "assumptions": self.assumptions,
            "ambiguity_information": self.ambiguity_information,
            "is_ambiguous": self.is_ambiguous,
            "is_unsupported": self.is_unsupported,
            "unsupported_reason": self.unsupported_reason,
        }


class UniversalOrchestrator:
    """
    Authoritative single source of truth for universal agent orchestration,
    task planning, DAG dependency execution, and multi-agent result synthesis.
    """

    def __init__(
        self,
        tool_registry: Optional[Any] = None,
        max_workers: int = 4,
        random_state: int = 42,
    ):
        if tool_registry is not None:
            self.tool_registry = tool_registry
        else:
            from agent.tool_registry import DEFAULT_TOOL_REGISTRY
            self.tool_registry = DEFAULT_TOOL_REGISTRY
        self.max_workers = max_workers
        self.random_state = random_state
        self.intent_analyzer = IntentAnalyzer()
        self.command_agent = CommandIntelligenceAgent()
        self.result_validator = ResultValidator()
        from agent.insight_synthesis_engine import InsightSynthesisEngine
        self.synthesis_engine = InsightSynthesisEngine()
        from agent.explanation_engine import ExplanationEngine
        self.explanation_engine = ExplanationEngine()

    # --------------------------------------------------------------------------
    # Public Entrypoints
    # --------------------------------------------------------------------------

    def orchestrate(
        self,
        command: str,
        data: Optional[Union[pd.DataFrame, Dict[str, Any], Any]] = None,
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
        time_column: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        dataset_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AgentResult:
        """
        Execute end-to-end orchestration pipeline from natural language command to AgentResult.
        Supports session context, conversational reference resolution, and follow-up turns.
        """
        start_time = datetime.now()
        orchestration_id = f"orch_{uuid.uuid4().hex[:8]}"
        from agent.analytical_context import DEFAULT_SESSION_CONTEXT_MANAGER

        effective_command = command
        resolution = None

        # 0. Conversational Context & Reference Resolution
        if session_id:
            ctx = DEFAULT_SESSION_CONTEXT_MANAGER.get_context(session_id)
            if ctx:
                resolution = DEFAULT_SESSION_CONTEXT_MANAGER.resolver.resolve(command, ctx, df=None)
                if resolution.needs_clarification:
                    return self._build_contextual_clarification_result(
                        command=command,
                        orchestration_id=orchestration_id,
                        reason=resolution.clarification_reason or "Ambiguous reference.",
                        suggested_options=resolution.suggested_options,
                        session_id=session_id,
                    )
                effective_command = resolution.resolved_command
                target = target or resolution.target
                features = features or (resolution.features if resolution.features else None)
                time_column = time_column or resolution.time_column
                dataset_id = dataset_id or resolution.dataset_id

        # 1. Extract & validate raw DataFrame
        df = self._extract_dataframe(data)
        if df is None or len(df) == 0 or len(df.columns) == 0:
            if session_id:
                cached_df = DEFAULT_SESSION_CONTEXT_MANAGER.get_dataset(session_id, dataset_id=dataset_id)
                if cached_df is not None and not cached_df.empty:
                    df = cached_df
                else:
                    return self._build_missing_session_dataset_error(command, orchestration_id, session_id)
            else:
                return self._build_empty_dataset_error(command, orchestration_id)

        # Register dataset in session if new
        if session_id and data is not None and (isinstance(data, pd.DataFrame) or isinstance(data, dict)):
            DEFAULT_SESSION_CONTEXT_MANAGER.register_dataset(
                session_id=session_id,
                df=df,
                dataset_id=dataset_id,
            )

        # 2. Ingest into CanonicalDataLayer & build SemanticProfile
        canonical_ds: CanonicalDataset = CanonicalDataLayer.ingest(df)
        profile: SemanticProfile = canonical_ds.profile

        # 3. Generate Analytical Plan
        plan: AnalyticalPlan = self.plan(
            command=effective_command,
            df=df,
            profile=profile,
            target=target,
            features=features,
            time_column=time_column,
            dataset_id=dataset_id,
            config=config,
        )

        # 4. Handle Ambiguous or Unsupported requests
        if plan.is_ambiguous:
            return self._build_clarification_result(plan, orchestration_id)
        if plan.is_unsupported:
            return self._build_unsupported_result(plan, orchestration_id)

        # 5. Execute Analytical Plan
        res = self.execute_plan(
            plan=plan,
            df=df,
            profile=profile,
            orchestration_id=orchestration_id,
            start_time=start_time,
        )

        # 6. Update Session Context
        if session_id:
            DEFAULT_SESSION_CONTEXT_MANAGER.record_execution(
                session_id=session_id,
                result=res,
                user_command=command,
                resolved_command=effective_command,
                target=target,
                features=features,
                time_column=time_column,
                resolved_references=resolution.resolved_references if resolution else {},
            )
            is_fu = bool(resolution.is_follow_up) if resolution else False
            refs = resolution.resolved_references if resolution else {}
            if isinstance(res.provenance, dict):
                res.provenance["session_id"] = session_id
                res.provenance["is_follow_up"] = is_fu
                res.provenance["resolved_references"] = refs
            if isinstance(res.result, dict):
                res.result["session_id"] = session_id
                res.result["is_follow_up"] = is_fu
                res.result["resolved_references"] = refs
            if isinstance(res.data, dict):
                res.data["session_id"] = session_id
                res.data["is_follow_up"] = is_fu
                res.data["resolved_references"] = refs

        return res

    # --------------------------------------------------------------------------
    # Planning Engine
    # --------------------------------------------------------------------------

    def plan(
        self,
        command: str,
        df: pd.DataFrame,
        profile: Optional[SemanticProfile] = None,
        target: Optional[str] = None,
        features: Optional[List[str]] = None,
        time_column: Optional[str] = None,
        dataset_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> AnalyticalPlan:
        """
        Dynamically synthesize an AnalyticalPlan DAG from user command and dataset profile.
        """
        if profile is None:
            profile = CanonicalDataLayer.ingest(df).profile

        cmd_clean = (command or "").strip()
        cmd_lower = cmd_clean.lower()
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        # A. Detect Unsupported Requests (e.g. image generation, video, audio, external scraping)
        unsupported_keywords = ["render video", "generate audio", "speech synthesis", "scrape website", "crypto mining"]
        if any(uk in cmd_lower for uk in unsupported_keywords):
            return AnalyticalPlan(
                plan_id=plan_id,
                user_request=cmd_clean,
                dataset_id=dataset_id,
                is_unsupported=True,
                unsupported_reason=f"The requested operation is not supported by the tabular data analytical agent.",
            )

        # B. Detect completely ambiguous / empty queries
        if not cmd_clean or len(cmd_clean.strip()) < 2:
            return AnalyticalPlan(
                plan_id=plan_id,
                user_request=cmd_clean,
                dataset_id=dataset_id,
                is_ambiguous=True,
                ambiguity_information={"reason": "User command is empty or too short to extract analytical intent."},
            )

        # C. Intent Detection
        classification = self.intent_analyzer.analyze(cmd_clean, dataframe=df)
        primary_intent = classification.primary_intent.value if classification.primary_intent else "eda"
        secondary_intents = [i.value for i in classification.secondary_intents]

        tasks: List[PlanTask] = []
        dependencies: Dict[str, List[str]] = {}
        assumptions: List[str] = []

        # Target Column Resolution
        effective_target = target or classification.target_column
        if not effective_target and primary_intent in ("prediction", "classification", "regression", "forecasting"):
            # Check if command mentions a valid column name
            for col in df.columns:
                if str(col).lower() in cmd_lower:
                    effective_target = str(col)
                    break

        # Time Column Resolution
        effective_time_col = time_column or getattr(classification, "time_column", None)
        if not effective_time_col and profile.datetime_candidates:
            effective_time_col = profile.datetime_candidates[0]

        # Feature Resolution
        effective_features = features or classification.feature_columns
        if not effective_features:
            non_id_cols = [c for c in df.columns if c != effective_target and c != effective_time_col and c not in profile.identifier_columns and c not in profile.constant_columns]
            if non_id_cols:
                effective_features = non_id_cols
            else:
                effective_features = [c for c in df.columns if c != effective_target and c != effective_time_col]
        if not effective_features:
            effective_features = list(df.columns)

        # D. Dynamic Task Generation based on Intent & Query Components
        # Check for Compound Commands (e.g. "profile data, find anomalies, and forecast next 6 months")
        needs_eda = False
        needs_anomaly = False
        needs_clustering = False
        needs_stats = False
        needs_hyp = False
        needs_forecast = False
        needs_prediction = False
        needs_transform = False

        if any(w in cmd_lower for w in ("eda", "profile", "describe", "summary", "overview", "quality", "distributions", "missing")):
            needs_eda = True
        if any(w in cmd_lower for w in ("anomaly", "anomalies", "outlier", "outliers", "unusual", "spikes")):
            needs_anomaly = True
        if any(w in cmd_lower for w in ("cluster", "clustering", "segment", "segmentation", "groups", "natural groups")):
            needs_clustering = True
        if any(w in cmd_lower for w in ("correlation", "correlations", "relationship", "relationships", "dependency", "association", "associations", "correlate", "related", "pearson", "spearman", "kendall", "effect size", "fdr", "outlier sensitivity", "subgroup")):
            needs_stats = True
        if any(w in cmd_lower for w in ("hypothesis test", "hypothesis testing", "t-test", "t test", "welch", "mann-whitney", "groups differ", "significantly different", "test whether the means")):
            needs_hyp = True
        elif not needs_stats and any(w in cmd_lower for w in ("hypothesis", "statistically significant", "significance", "differ")):
            needs_hyp = True
        if any(w in cmd_lower for w in ("forecast", "future", "predict next", "horizon", "time series")):
            needs_forecast = True
        elif any(w in cmd_lower for w in ("predict", "train", "classification", "regression", "model", "churn", "estimate")):
            needs_prediction = True
        if any(w in cmd_lower for w in ("clean", "transform", "preprocess", "impute", "encode", "scale")):
            needs_transform = True

        # Default to primary intent if no specific sub-flags matched
        if not any([needs_eda, needs_anomaly, needs_clustering, needs_stats, needs_hyp, needs_forecast, needs_prediction, needs_transform]):
            if primary_intent == "forecasting":
                needs_forecast = True
            elif primary_intent in ("prediction", "classification", "regression"):
                needs_prediction = True
            elif primary_intent in ("clustering", "segmentation"):
                needs_clustering = True
            elif primary_intent in ("anomaly_detection", "anomalies"):
                needs_anomaly = True
            elif primary_intent in ("statistical_analysis", "correlation"):
                needs_stats = True
            elif primary_intent == "hypothesis_testing":
                needs_hyp = True
            elif primary_intent == "cleaning":
                needs_transform = True
            else:
                needs_eda = True

        # Check for target ambiguity in supervised tasks
        if needs_prediction and not effective_target:
            # If multiple potential numeric/categorical targets exist and user didn't specify one
            potential_targets = [c for c in df.columns if c not in profile.identifier_columns and c not in profile.constant_columns]
            if len(potential_targets) > 1 and "predict" in cmd_lower and not any(str(c).lower() in cmd_lower for c in potential_targets):
                return AnalyticalPlan(
                    plan_id=plan_id,
                    user_request=cmd_clean,
                    dataset_id=dataset_id,
                    detected_intent="prediction",
                    is_ambiguous=True,
                    ambiguity_information={
                        "reason": "Predictive modeling requested without specifying a target column.",
                        "available_candidate_targets": potential_targets[:5],
                        "suggested_action": "Specify target column name in command (e.g., 'predict target_column_name').",
                    },
                )

        # Build Task Objects & Dependencies
        eda_task_id = None
        if needs_eda:
            t_eda = PlanTask(
                task_id=f"task_eda_{uuid.uuid4().hex[:6]}",
                task_type="eda",
                tool_name="eda",
                agent_name="EDA Agent",
                purpose="Profile dataset distributions, missingness, schema types, and quality scoring.",
                parameters={"max_categories": 10},
                priority=1,
            )
            tasks.append(t_eda)
            eda_task_id = t_eda.task_id
            dependencies[eda_task_id] = []

        if needs_transform:
            t_trans = PlanTask(
                task_id=f"task_trans_{uuid.uuid4().hex[:6]}",
                task_type="transformation",
                tool_name="transformation",
                agent_name="Transformation Agent",
                purpose="Clean, impute, scale, and transform tabular dataset into model-ready representation.",
                target_column=effective_target,
                required_columns=effective_features,
                parameters={"target": effective_target, "features": effective_features},
                dependencies=[eda_task_id] if eda_task_id else [],
                priority=2,
            )
            tasks.append(t_trans)
            dependencies[t_trans.task_id] = [eda_task_id] if eda_task_id else []

        if needs_stats:
            if any(w in cmd_lower for w in ("all", "dataset", "comprehensively", "relationships", "correlations", "matrix", "identify the strongest", "strongest")) or len(effective_features) < 4:
                stats_cols = [c for c in df.columns if c not in profile.identifier_columns and c not in profile.constant_columns]
            else:
                stats_cols = effective_features if len(effective_features) >= 2 else list(df.columns)
            t_stats = PlanTask(
                task_id=f"task_stats_{uuid.uuid4().hex[:6]}",
                task_type="statistical_analysis",
                tool_name="statistical_analysis",
                agent_name="Statistical Analysis Agent",
                purpose="Analyze statistical relationships, correlations, and feature dependencies.",
                target_column=effective_target,
                required_columns=stats_cols,
                parameters={"target": effective_target, "features": stats_cols},
                dependencies=[eda_task_id] if eda_task_id else [],
                priority=2,
            )
            tasks.append(t_stats)
            dependencies[t_stats.task_id] = [eda_task_id] if eda_task_id else []

        if needs_hyp:
            t_hyp = PlanTask(
                task_id=f"task_hyp_{uuid.uuid4().hex[:6]}",
                task_type="hypothesis_testing",
                tool_name="hypothesis_testing",
                agent_name="Hypothesis Testing Agent",
                purpose="Perform statistical hypothesis tests, group comparisons, and effect size estimations.",
                target_column=effective_target,
                required_columns=effective_features,
                parameters={"target": effective_target, "features": effective_features},
                dependencies=[eda_task_id] if eda_task_id else [],
                priority=2,
            )
            tasks.append(t_hyp)
            dependencies[t_hyp.task_id] = [eda_task_id] if eda_task_id else []

        if needs_anomaly:
            t_anom = PlanTask(
                task_id=f"task_anom_{uuid.uuid4().hex[:6]}",
                task_type="anomaly_detection",
                tool_name="anomaly_detection",
                agent_name="Anomaly Detection Agent",
                purpose="Detect multivariate and univariate outliers and abnormal deviations.",
                required_columns=effective_features,
                parameters={"features": effective_features, "target": effective_target},
                dependencies=[eda_task_id] if eda_task_id else [],
                priority=2,
            )
            tasks.append(t_anom)
            dependencies[t_anom.task_id] = [eda_task_id] if eda_task_id else []

        if needs_clustering:
            t_clust = PlanTask(
                task_id=f"task_clust_{uuid.uuid4().hex[:6]}",
                task_type="clustering",
                tool_name="clustering",
                agent_name="Clustering Agent",
                purpose="Discover natural groupings and cluster segments across numeric features.",
                required_columns=effective_features,
                parameters={"features": effective_features},
                dependencies=[eda_task_id] if eda_task_id else [],
                priority=2,
            )
            tasks.append(t_clust)
            dependencies[t_clust.task_id] = [eda_task_id] if eda_task_id else []

        if needs_forecast:
            t_fc = PlanTask(
                task_id=f"task_fc_{uuid.uuid4().hex[:6]}",
                task_type="forecasting",
                tool_name="forecasting",
                agent_name="Forecast Agent",
                purpose="Generate autonomous time series forecast with confidence prediction intervals.",
                target_column=effective_target,
                time_column=effective_time_col,
                parameters={
                    "target_column": effective_target,
                    "time_column": effective_time_col,
                    "horizon": classification.time_horizon or 6,
                },
                dependencies=[eda_task_id] if eda_task_id else [],
                priority=3,
            )
            tasks.append(t_fc)
            dependencies[t_fc.task_id] = [eda_task_id] if eda_task_id else []

        if needs_prediction:
            t_pred = PlanTask(
                task_id=f"task_pred_{uuid.uuid4().hex[:6]}",
                task_type="prediction",
                tool_name="prediction",
                agent_name="Prediction Agent",
                purpose="Train, benchmark, and evaluate supervised machine learning models.",
                target_column=effective_target,
                required_columns=effective_features,
                parameters={"target": effective_target, "features": effective_features},
                dependencies=[eda_task_id] if eda_task_id else [],
                priority=3,
            )
            tasks.append(t_pred)
            dependencies[t_pred.task_id] = [eda_task_id] if eda_task_id else []

        # If no tasks created, fallback to EDA
        if not tasks:
            t_fallback = PlanTask(
                task_id=f"task_eda_{uuid.uuid4().hex[:6]}",
                task_type="eda",
                tool_name="eda",
                agent_name="EDA Agent",
                purpose="Explore dataset structure and statistical properties.",
                priority=1,
            )
            tasks.append(t_fallback)
            dependencies[t_fallback.task_id] = []

        execution_order = [t.task_id for t in tasks]

        return AnalyticalPlan(
            plan_id=plan_id,
            user_request=cmd_clean,
            dataset_id=dataset_id,
            detected_intent=primary_intent,
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            tasks=tasks,
            dependencies=dependencies,
            execution_order=execution_order,
            requested_columns=list(df.columns),
            inferred_columns=effective_features,
            assumptions=[
                "Execution respects topological dependency ordering.",
                "Non-fatal errors in independent tasks are isolated to preserve partial analytical output.",
            ],
        )

    # --------------------------------------------------------------------------
    # Execution Engine
    # --------------------------------------------------------------------------

    def execute_plan(
        self,
        plan: AnalyticalPlan,
        df: pd.DataFrame,
        profile: Optional[SemanticProfile] = None,
        orchestration_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
    ) -> AgentResult:
        """
        Execute an AnalyticalPlan with topological levels, pre-execution validation,
        error isolation, retries, and result aggregation.
        """
        if start_time is None:
            start_time = datetime.now()
        if orchestration_id is None:
            orchestration_id = f"orch_{uuid.uuid4().hex[:8]}"
        if profile is None:
            profile = CanonicalDataLayer.ingest(df).profile

        task_results: Dict[str, AgentResult] = {}
        task_execution_times: Dict[str, float] = {}
        levels = self._build_dependency_levels(plan.tasks, plan.dependencies)

        total_retries = 0

        # Execute level-by-level
        for level in levels:
            for task in level:
                t_id = task.task_id
                t_type = task.task_type
                tool_name = task.tool_name
                deps = task.dependencies

                # 1. Dependency Pre-check: Verify upstream prerequisites succeeded
                dep_failed = False
                for dep_id in deps:
                    dep_res = task_results.get(dep_id)
                    if not dep_res or not dep_res.is_success:
                        dep_failed = True
                        break

                if dep_failed:
                    task.status = TaskStatus.SKIPPED.value
                    task.error = {"message": f"Task skipped due to upstream dependency failure."}
                    continue

                # 2. Pre-Execution Validation
                pre_val = PreExecutionValidator.validate(
                    df,
                    task_type=t_type,
                    target=task.target_column,
                    feature_columns=task.required_columns or None,
                    agent_name=task.agent_name or tool_name,
                )

                if not pre_val.is_valid:
                    task.status = TaskStatus.BLOCKED.value
                    err = pre_val.error
                    err_dict = {
                        "code": err.code if err else "VALIDATION_FAILED",
                        "message": err.user_message if err else "Pre-execution validation failed.",
                        "category": str(err.category.value) if err else "data_invalid",
                        "details": err.technical_details if err else {},
                    }
                    task.error = err_dict
                    # Store synthetic failed AgentResult for isolation
                    failed_res = AgentResult.error(
                        error=err.user_message if err else "Pre-execution validation failed.",
                        code=err.code if err else "VALIDATION_FAILED",
                        agent_name=task.agent_name or tool_name,
                        category=err.category if err else ErrorCategory.DATA_INVALID,
                        task_type=t_type,
                    )
                    task_results[t_id] = failed_res
                    continue

                # 3. Execution with Retry Policy
                task.status = TaskStatus.RUNNING.value
                t_start = datetime.now()
                max_retries = task.retry_policy.get("max_retries", 1)
                attempt = 0
                res: Optional[AgentResult] = None

                while attempt <= max_retries:
                    try:
                        task_inputs = {"data": df, **task.parameters}
                        if task.target_column and "target" not in task_inputs:
                            task_inputs["target"] = task.target_column
                        if task.time_column and "time_column" not in task_inputs:
                            task_inputs["time_column"] = task.time_column

                        raw_output = self.tool_registry.execute(tool_name, **task_inputs)

                        if isinstance(raw_output, AgentResult):
                            res = raw_output
                        else:
                            res = AgentResult.success(
                                output=raw_output if isinstance(raw_output, dict) else {"result": raw_output},
                                agent_name=task.agent_name or tool_name,
                                task_type=t_type,
                            )

                        if res.is_success:
                            break
                        else:
                            attempt += 1
                            total_retries += 1
                            if attempt <= max_retries:
                                time.sleep(task.retry_policy.get("backoff", 0.05))
                    except Exception as exc:
                        attempt += 1
                        total_retries += 1
                        if attempt > max_retries:
                            res = AgentResult.error(
                                error=f"Task execution failed: {str(exc)}",
                                code="TASK_EXECUTION_ERROR",
                                agent_name=task.agent_name or tool_name,
                                category=ErrorCategory.COMPUTATION,
                                task_type=t_type,
                            )
                            break
                        time.sleep(task.retry_policy.get("backoff", 0.05))

                t_dur = round((datetime.now() - t_start).total_seconds() * 1000, 2)
                task.duration_ms = t_dur
                task_execution_times[t_id] = t_dur

                if res and res.is_success:
                    task.status = TaskStatus.SUCCESS.value
                    task.result = res.to_dict()
                    task.agent_result = res
                    task_results[t_id] = res
                else:
                    task.status = TaskStatus.FAILED.value
                    task.error = {
                        "message": res.error_message if res else "Task failed without result.",
                        "code": res.errors[0].code if (res and res.errors) else "EXECUTION_FAILURE",
                    }
                    if res:
                        task_results[t_id] = res

        # 6. Aggregate Results
        return self._aggregate_results(
            plan=plan,
            task_results=task_results,
            df=df,
            orchestration_id=orchestration_id,
            start_time=start_time,
            total_retries=total_retries,
            task_execution_times=task_execution_times,
        )

    # --------------------------------------------------------------------------
    # Aggregation & Synthesis
    # --------------------------------------------------------------------------

    def _aggregate_results(
        self,
        plan: AnalyticalPlan,
        task_results: Dict[str, AgentResult],
        df: pd.DataFrame,
        orchestration_id: str,
        start_time: datetime,
        total_retries: int,
        task_execution_times: Dict[str, float],
    ) -> AgentResult:
        """
        Synthesize individual task results, metrics, evidence, warnings, and confidence
        into a canonical AgentResult response.
        """
        duration_ms = round((datetime.now() - start_time).total_seconds() * 1000, 2)
        total_tasks = len(plan.tasks)
        success_tasks = [t for t in plan.tasks if t.status == TaskStatus.SUCCESS.value]
        failed_tasks = [t for t in plan.tasks if t.status in (TaskStatus.FAILED.value, TaskStatus.BLOCKED.value)]
        skipped_tasks = [t for t in plan.tasks if t.status == TaskStatus.SKIPPED.value]

        n_success = len(success_tasks)
        n_failed = len(failed_tasks)
        n_skipped = len(skipped_tasks)

        # Determine overall AgentStatus
        if n_success == total_tasks and total_tasks > 0:
            overall_status = AgentStatus.COMPLETED
        elif n_success > 0:
            overall_status = AgentStatus.PARTIAL
        else:
            overall_status = AgentStatus.ERROR

        # Aggregate Evidence
        all_evidence: List[Evidence] = []
        all_warnings: List[str] = []
        all_errors: List[AgentError] = []
        all_assumptions: List[str] = list(plan.assumptions)
        all_limitations: List[str] = []
        metrics_dict: Dict[str, Any] = {}
        task_outputs: Dict[str, Any] = {}

        # Preserve Evidence from each successful task
        for t in plan.tasks:
            t_res = task_results.get(t.task_id)
            if t_res:
                if t_res.is_success:
                    all_evidence.extend(t_res.evidence)
                    all_warnings.extend(t_res.warnings)
                    all_assumptions.extend(t_res.assumptions)
                    all_limitations.extend(t_res.limitations)
                    metrics_dict[t.task_type] = t_res.metrics
                    task_outputs[t.task_type] = t_res.data
                else:
                    all_errors.extend(t_res.errors)
                    all_warnings.extend(t_res.warnings)

        # Composite Confidence Calculation
        if total_tasks > 0:
            task_conf_sum = sum(
                (task_results[t.task_id].confidence if (t.task_id in task_results and task_results[t.task_id].is_success) else 0.20)
                for t in plan.tasks
            )
            raw_conf = task_conf_sum / float(total_tasks)
            # Apply success ratio penalty
            success_ratio = n_success / float(total_tasks)
            composite_confidence = round(max(0.0, min(1.0, raw_conf * (0.40 + 0.60 * success_ratio))), 4)
        else:
            composite_confidence = 0.50

        # Construct Unified Executive Summary / Narrative
        summary_narrative = self._synthesize_narrative(plan, success_tasks, failed_tasks, task_outputs)

        execution_graph = [
            {
                "task_id": t.task_id,
                "task_type": t.task_type,
                "tool": t.tool_name,
                "status": t.status,
                "duration_ms": t.duration_ms,
                "dependencies": t.dependencies,
            }
            for t in plan.tasks
        ]

        # 6. Cross-Agent Analytical Insight Synthesis
        synth_report = self.synthesis_engine.synthesize(
            orchestration_result={"task_outputs": task_outputs, "evidence": all_evidence, "confidence": composite_confidence},
            dataframe=df,
            command=plan.user_request,
        )

        # 7. Analytical Explanation & Evidence Traceability (Milestone 7, Task 4)
        explanation_report = self.explanation_engine.explain(
            result={
                "tasks": {t.task_type: (t.result or t.error) for t in plan.tasks},
                "task_outputs": task_outputs,
                "synthesis": synth_report.to_dict(),
                "evidence": all_evidence,
                "confidence": composite_confidence,
            },
            dataframe=df,
            command=plan.user_request,
        )

        aggregated_data = {
            "orchestration_id": orchestration_id,
            "plan_id": plan.plan_id,
            "user_request": plan.user_request,
            "detected_intent": plan.detected_intent,
            "status": overall_status.value,
            "task_summary": {
                "total_tasks": total_tasks,
                "completed_tasks": n_success,
                "failed_tasks": n_failed,
                "skipped_tasks": n_skipped,
                "retry_count": total_retries,
            },
            "summary": synth_report.executive_summary or summary_narrative,
            "executive_summary": synth_report.executive_summary,
            "key_insights": [i.to_dict() for i in synth_report.key_insights],
            "contradictions": [c.to_dict() for c in synth_report.contradictions],
            "recommended_next_questions": synth_report.recommended_next_questions,
            "synthesis": synth_report.to_dict(),
            "explanation": explanation_report.to_dict(),
            "tasks": {t.task_type: (t.result or t.error) for t in plan.tasks},
            "task_outputs": task_outputs,
            "execution_graph": execution_graph,
            "task_execution_times": task_execution_times,
            "dataset_rows": len(df),
            "dataset_columns": list(df.columns),
        }

        agent_result = AgentResult(
            status=overall_status,
            task_type="orchestration",
            agent_name="Universal Orchestrator",
            dataset_id=plan.dataset_id,
            target=plan.tasks[0].target_column if plan.tasks else None,
            execution_id=orchestration_id,
            result=aggregated_data,
            data=aggregated_data,
            output=aggregated_data,
            metrics=metrics_dict,
            evidence=all_evidence,
            confidence=composite_confidence,
            warnings=list(set(all_warnings)),
            errors=all_errors,
            assumptions=list(set(all_assumptions)),
            limitations=list(set(all_limitations)),
            duration_ms=duration_ms,
            execution_time_ms=duration_ms,
            provenance={
                "orchestrator": "UniversalOrchestrator",
                "plan_id": plan.plan_id,
                "plan_tasks": len(plan.tasks),
            },
        )

        return agent_result

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def _build_dependency_levels(
        self,
        tasks: List[PlanTask],
        dependencies: Dict[str, List[str]],
    ) -> List[List[PlanTask]]:
        """Group tasks into topological levels for dependency-aware concurrent execution."""
        if not tasks:
            return []

        task_map = {t.task_id: t for t in tasks}
        resolved_ids: Set[str] = set()
        remaining = list(tasks)
        levels: List[List[PlanTask]] = []

        while remaining:
            current_level: List[PlanTask] = []
            next_remaining: List[PlanTask] = []

            for t in remaining:
                deps = dependencies.get(t.task_id, t.dependencies)
                if all(d in resolved_ids for d in deps):
                    current_level.append(t)
                else:
                    next_remaining.append(t)

            if not current_level and next_remaining:
                # Cycle break safeguard
                current_level.append(next_remaining.pop(0))

            for t in current_level:
                resolved_ids.add(t.task_id)

            levels.append(current_level)
            remaining = next_remaining

        return levels

    def _synthesize_narrative(
        self,
        plan: AnalyticalPlan,
        success_tasks: List[PlanTask],
        failed_tasks: List[PlanTask],
        outputs: Dict[str, Any],
    ) -> str:
        """Create an executive narrative summary of all executed tasks."""
        lines: List[str] = [f"Analysis completed for command: '{plan.user_request}'."]

        if success_tasks:
            lines.append(f"Successfully executed {len(success_tasks)} analytical task(s): {', '.join(t.task_type for t in success_tasks)}.")
        if failed_tasks:
            lines.append(f"Encountered non-fatal issues in {len(failed_tasks)} task(s): {', '.join(t.task_type for t in failed_tasks)}.")

        for t_type, out in outputs.items():
            if isinstance(out, dict):
                if "summary" in out and isinstance(out["summary"], dict):
                    row_c = out["summary"].get("row_count") or out["summary"].get("original_rows")
                    if row_c:
                        lines.append(f"Dataset profile analyzed {row_c} observations.")
                elif "forecast" in out or "predictions" in out:
                    lines.append("Forecasting models generated predictive intervals.")
                elif "cluster_count" in out:
                    lines.append(f"Clustering discovered {out['cluster_count']} natural groups.")
                elif "anomalies" in out or "anomaly_count" in out:
                    lines.append(f"Anomaly detection flagged {out.get('anomaly_count', len(out.get('anomalies', [])))} outliers.")

        return " ".join(lines)

    def _build_empty_dataset_error(self, command: str, orchestration_id: str) -> AgentResult:
        err = AgentError(
            code="EMPTY_DATASET",
            category=ErrorCategory.INSUFFICIENT_DATA,
            user_message="Dataset is empty or contains 0 valid columns. Orchestration requires tabular data.",
            message="Dataset is empty or contains 0 valid columns.",
            agent_name="Universal Orchestrator",
        )
        return AgentResult.error(
            error=err.user_message,
            code=err.code,
            category=err.category,
            agent_name="Universal Orchestrator",
            task_type="orchestration",
            task_id=orchestration_id,
            errors=[err],
        )

    def _build_missing_session_dataset_error(self, command: str, orchestration_id: str, session_id: str) -> AgentResult:
        err = AgentError(
            code="SESSION_DATASET_MISSING",
            category=ErrorCategory.INSUFFICIENT_DATA,
            user_message=f"No dataset found for session '{session_id}'. Please provide a dataset with your analytical command.",
            message=f"No dataset found in session '{session_id}'.",
            agent_name="Universal Orchestrator",
        )
        return AgentResult(
            status=AgentStatus.NEEDS_CLARIFICATION,
            task_type="orchestration",
            agent_name="Universal Orchestrator",
            execution_id=orchestration_id,
            result={"error": err.user_message},
            data={"error": err.user_message},
            output={"error": err.user_message},
            confidence=0.20,
            errors=[err],
            warnings=[err.user_message],
        )

    def _build_contextual_clarification_result(
        self,
        command: str,
        orchestration_id: str,
        reason: str,
        suggested_options: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> AgentResult:
        err = AgentError(
            code="AMBIGUOUS_FOLLOW_UP",
            category=ErrorCategory.INPUT_INVALID,
            user_message=f"Ambiguous follow-up reference: {reason}",
            message=f"Ambiguous follow-up reference: {reason}",
            agent_name="Universal Orchestrator",
            technical_details={"suggested_options": suggested_options or [], "session_id": session_id},
        )
        return AgentResult(
            status=AgentStatus.NEEDS_CLARIFICATION,
            task_type="orchestration",
            agent_name="Universal Orchestrator",
            execution_id=orchestration_id,
            result={"error": err.user_message, "suggested_options": suggested_options or []},
            data={"error": err.user_message, "suggested_options": suggested_options or []},
            output={"error": err.user_message, "suggested_options": suggested_options or []},
            confidence=0.30,
            errors=[err],
            warnings=[f"Clarification needed: {reason}"],
        )

    def _build_clarification_result(self, plan: AnalyticalPlan, orchestration_id: str) -> AgentResult:
        reason = plan.ambiguity_information.get("reason", "Command is ambiguous.") if plan.ambiguity_information else "Command is ambiguous."
        err = AgentError(
            code="AMBIGUOUS_COMMAND",
            category=ErrorCategory.DATA_INVALID,
            user_message=f"Ambiguous request: {reason}",
            message=f"Ambiguous request: {reason}",
            agent_name="Universal Orchestrator",
            technical_details=plan.ambiguity_information or {},
        )
        res = AgentResult(
            status=AgentStatus.NEEDS_CLARIFICATION,
            task_type="orchestration",
            agent_name="Universal Orchestrator",
            execution_id=orchestration_id,
            result={"plan": plan.to_dict(), "error": err.user_message},
            data={"plan": plan.to_dict(), "error": err.user_message},
            output={"plan": plan.to_dict(), "error": err.user_message},
            confidence=0.30,
            errors=[err],
            warnings=["Command requires clarification before analytical tasks can proceed."],
        )
        return res

    def _build_unsupported_result(self, plan: AnalyticalPlan, orchestration_id: str) -> AgentResult:
        reason = plan.unsupported_reason or "The requested capability is not supported by the analytical platform."
        err = AgentError(
            code="UNSUPPORTED_COMMAND",
            category=ErrorCategory.UNSUPPORTED_TASK,
            user_message=f"Unsupported request: {reason}",
            message=f"Unsupported request: {reason}",
            agent_name="Universal Orchestrator",
        )
        res = AgentResult(
            status=AgentStatus.NOT_SUPPORTED,
            task_type="orchestration",
            agent_name="Universal Orchestrator",
            execution_id=orchestration_id,
            result={"plan": plan.to_dict(), "error": err.user_message},
            data={"plan": plan.to_dict(), "error": err.user_message},
            output={"plan": plan.to_dict(), "error": err.user_message},
            confidence=0.10,
            errors=[err],
            warnings=[reason],
        )
        return res

    def _extract_dataframe(self, data: Any) -> Optional[pd.DataFrame]:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, dict):
            if "records" in data and isinstance(data["records"], list):
                return pd.DataFrame(data["records"])
            for val in data.values():
                if isinstance(val, pd.DataFrame) and not val.empty:
                    return val
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                    return pd.DataFrame(val)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return pd.DataFrame(data)
        return None