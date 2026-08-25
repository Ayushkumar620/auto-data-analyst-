"""
Dynamic Task Planner & Execution Graph Engine.

Transforms natural language UserIntent and DatasetKnowledge into executable,
dependency-aware ExecutionPlan DAGs:
- Step Decomposition & Capability Mapping
- Topological Dependency Resolution
- Circular Dependency & Cycle Detection
- Tool & Agent Validation (rejects unregistered tools / invalid schemas)
- Deterministic Fallback & LLM Plan Validation
- Full backward compatibility for legacy TaskPlan & PlanStep callers
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from agent.base import BaseAgent
from agent.dataset_knowledge import DatasetKnowledge
from agent.execution_engine import ExecutionEngine, StepStatus
from agent.intent import AnalyticalIntent, CommandIntelligenceAgent, IntentAnalyzer, IntentClassificationResult, IntentType, UserIntent
from agent.result_validator import ResultValidator
from agent.schemas import AgentError, AgentResult, AgentStatus, ClaimType, Evidence
from agent.tool_registry import DEFAULT_TOOL_REGISTRY, ToolDefinition, ToolRegistry
from backend.app.core.llm_provider import BaseLLMProvider, LLMClientFactory, LLMMessage


# ---------------------------------------------------------------------------
# Execution Plan Models (Pydantic v2)
# ---------------------------------------------------------------------------

class ExecutionStep(BaseModel):
    """Atomic step in an analytical DAG execution graph."""
    step_id: str
    tool_name: str
    agent_name: str = ""
    purpose: str = ""
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    required_capabilities: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending, running, success, partial, failed, skipped, retrying
    retry_policy: Dict[str, Any] = Field(default_factory=lambda: {"max_retries": 1, "backoff": 1.0})
    timeout: Optional[float] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    result: Optional[AgentResult] = None
    duration_ms: float = 0.0

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return float(v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "agent_name": self.agent_name,
            "purpose": self.purpose,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "dependencies": self.dependencies,
            "required_capabilities": self.required_capabilities,
            "status": self.status,
            "retry_policy": self.retry_policy,
            "timeout": self.timeout,
            "confidence": round(float(self.confidence), 4),
            "duration_ms": self.duration_ms,
            "has_result": self.result is not None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionStep":
        return cls.model_validate(data)


class ExecutionPlan(BaseModel):
    """Complete dependency-aware task execution DAG."""
    plan_id: str
    task_id: str = ""
    user_intent: Union[UserIntent, Dict[str, Any]]
    objective: str = ""
    steps: List[ExecutionStep] = Field(default_factory=list)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    estimated_complexity: str = "medium"  # low, medium, high
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    required_tools: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    total_duration_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        intent_dict = self.user_intent.to_dict() if isinstance(self.user_intent, UserIntent) else self.user_intent
        return {
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "user_intent": intent_dict,
            "objective": self.objective,
            "steps": [s.to_dict() for s in self.steps],
            "dependencies": self.dependencies,
            "estimated_complexity": self.estimated_complexity,
            "confidence": round(float(self.confidence), 4),
            "required_tools": self.required_tools,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "total_duration_ms": self.total_duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPlan":
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Legacy TaskPlan & PlanStep Classes for Backwards Compatibility
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    step_id: int
    name: str
    agent_class_name: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[int] = field(default_factory=list)
    validation_criteria: str = "Result status must be completed with valid output."
    fallback_strategy: str = "Retry with relaxed constraints or fallback to statistical summary."
    status: str = "pending"
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
    plan_id: str
    query: str
    intent: Union[IntentClassificationResult, Dict[str, Any]]
    steps: List[Union[PlanStep, ExecutionStep]] = field(default_factory=list)
    dataset_validation: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    total_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        intent_dict = self.intent.to_dict() if hasattr(self.intent, "to_dict") else self.intent
        return {
            "plan_id": self.plan_id,
            "query": self.query,
            "intent": intent_dict,
            "steps": [s.to_dict() for s in self.steps],
            "dataset_validation": self.dataset_validation,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "total_duration_ms": self.total_duration_ms,
        }


# ---------------------------------------------------------------------------
# Execution Graph & Cycle Detection
# ---------------------------------------------------------------------------

class ExecutionGraph:
    """Manages DAG validation, cycle detection, and topological ordering."""

    def __init__(self, steps: List[ExecutionStep]):
        self.steps = steps
        self.adj: Dict[str, List[str]] = {}
        self.in_degree: Dict[str, int] = {}
        self._build_graph()

    def _build_graph(self) -> None:
        for step in self.steps:
            self.adj[step.step_id] = []
            self.in_degree[step.step_id] = 0

        for step in self.steps:
            for dep in step.dependencies:
                if dep in self.adj:
                    self.adj[dep].append(step.step_id)
                    self.in_degree[step.step_id] += 1

    def detect_cycles(self) -> bool:
        """Return True if a cycle exists in the dependency graph."""
        visited: Dict[str, int] = {k: 0 for k in self.adj}  # 0: unvisited, 1: visiting, 2: visited

        def dfs(node: str) -> bool:
            visited[node] = 1
            for neighbor in self.adj.get(node, []):
                if visited.get(neighbor) == 1:
                    return True
                if visited.get(neighbor) == 0:
                    if dfs(neighbor):
                        return True
            visited[node] = 2
            return False

        for node in self.adj:
            if visited[node] == 0:
                if dfs(node):
                    return True
        return False

    def get_execution_order(self) -> List[List[str]]:
        """Return topological levels for sequential/parallel step execution."""
        if self.detect_cycles():
            raise ValueError("Circular dependency detected in execution graph.")

        in_deg = dict(self.in_degree)
        levels: List[List[str]] = []

        current_level = [node for node, deg in in_deg.items() if deg == 0]
        while current_level:
            levels.append(current_level)
            next_level = []
            for node in current_level:
                for neighbor in self.adj.get(node, []):
                    in_deg[neighbor] -= 1
                    if in_deg[neighbor] == 0:
                        next_level.append(neighbor)
            current_level = next_level

        return levels

    @classmethod
    def validate_plan(cls, plan: ExecutionPlan, tool_registry: ToolRegistry) -> List[str]:
        """Validate an ExecutionPlan against tool registry and dependency rules."""
        errors: List[str] = []
        step_ids = {s.step_id for s in plan.steps}

        for step in plan.steps:
            # 1. Check Tool Registration
            if not tool_registry.has_tool(step.tool_name):
                errors.append(f"Step '{step.step_id}' references unregistered tool '{step.tool_name}'.")

            # 2. Check Dependency Existence
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(f"Step '{step.step_id}' has impossible dependency '{dep}' (step does not exist).")

        # 3. Check Cycles
        graph = cls(plan.steps)
        if graph.detect_cycles():
            errors.append("Circular dependency detected in execution plan steps.")

        return errors


# ---------------------------------------------------------------------------
# Dynamic Task Planner
# ---------------------------------------------------------------------------

class DynamicTaskPlanner(BaseAgent):
    """
    Synthesizes and validates dynamic execution plans from user intent and dataset knowledge.
    """
    name = "Dynamic Task Planner"
    description = "Synthesizes dependency-aware execution DAGs for multi-agent workflows."
    role = "planner"

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        intent_analyzer: Optional[IntentAnalyzer] = None,
        validator: Optional[ResultValidator] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
    ):
        super().__init__()
        self.tool_registry = tool_registry or DEFAULT_TOOL_REGISTRY
        self.intent_analyzer = intent_analyzer or IntentAnalyzer()
        self.validator = validator or ResultValidator()
        self.llm_provider = llm_provider
        self.command_agent = CommandIntelligenceAgent(llm_provider=llm_provider)
        self.execution_engine = ExecutionEngine(tool_registry=self.tool_registry)

    # ------------------------------------------------------------------
    # Modern Plan Creation Interface
    # ------------------------------------------------------------------
    def create_execution_plan(
        self,
        intent: Union[UserIntent, str],
        dataframe: Optional[pd.DataFrame] = None,
        knowledge: Optional[DatasetKnowledge] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> ExecutionPlan:
        """
        Synthesize a validated ExecutionPlan from UserIntent and DatasetKnowledge.
        """
        registry = tool_registry or self.tool_registry
        user_intent: UserIntent

        if isinstance(intent, str):
            user_intent = self.command_agent.analyze_intent(intent, dataset_knowledge=knowledge)
        else:
            user_intent = intent

        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        steps: List[ExecutionStep] = []
        deps_map: Dict[str, List[str]] = {}
        step_idx = 1

        # Check required capabilities & intent
        req_caps = set(user_intent.required_capabilities)
        primary_intent = user_intent.intent_type
        if isinstance(primary_intent, IntentType):
            primary_intent_val = primary_intent.value
        else:
            primary_intent_val = str(primary_intent)

        # 1. Dataset Profiling / Validation Step (if needed)
        profiling_step_id = f"step_{step_idx}"
        steps.append(
            ExecutionStep(
                step_id=profiling_step_id,
                tool_name="dataset_profiling",
                agent_name="DataValidationAgent",
                purpose="Validate dataset structure and baseline quality.",
                inputs={},
                required_capabilities=["dataset_profiling"],
                dependencies=[],
            )
        )
        step_idx += 1

        # 2. Data Cleaning Step (if requested or required)
        cleaning_step_id = None
        if "data_cleaning" in req_caps or "duplicate_handling" in req_caps or primary_intent_val == "data_cleaning":
            cleaning_step_id = f"step_{step_idx}"
            steps.append(
                ExecutionStep(
                    step_id=cleaning_step_id,
                    tool_name="data_cleaning",
                    agent_name="CleaningAgent",
                    purpose="Clean data, remove duplicates, and impute missing values.",
                    inputs={"strategy": "auto_impute"},
                    required_capabilities=["data_cleaning"],
                    dependencies=[profiling_step_id],
                )
            )
            step_idx += 1

        upstream_dep = [cleaning_step_id] if cleaning_step_id else [profiling_step_id]

        # 3. Main Analytical Execution Paths
        if primary_intent_val == "root_cause_analysis":
            # Multi-step Root Cause DAG:
            # A: Period Metric Aggregation -> B: Regional/Product Segmentation -> C: Anomaly Detection -> D: Explanation
            metric_target = user_intent.metrics[0] if user_intent.metrics else (knowledge.get_primary_metric() if knowledge else "revenue")
            
            # Step A: Aggregation & Trend
            step_a = f"step_{step_idx}"
            steps.append(
                ExecutionStep(
                    step_id=step_a,
                    tool_name="aggregation",
                    agent_name="AnalysisAgent",
                    purpose=f"Calculate historical and current period aggregates for '{metric_target}'.",
                    inputs={"metric": metric_target, "request": "summary"},
                    required_capabilities=["aggregation"],
                    dependencies=upstream_dep,
                )
            )
            step_idx += 1

            # Step B: Anomaly Detection (runs in parallel with aggregation or after cleaning)
            step_b = f"step_{step_idx}"
            steps.append(
                ExecutionStep(
                    step_id=step_b,
                    tool_name="anomaly_detection",
                    agent_name="InsightAgent",
                    purpose=f"Detect statistical outliers in '{metric_target}'.",
                    inputs={"column": metric_target},
                    required_capabilities=["anomaly_detection"],
                    dependencies=upstream_dep,
                )
            )
            step_idx += 1

            # Step C: Explanation & Driver Synthesis
            step_c = f"step_{step_idx}"
            steps.append(
                ExecutionStep(
                    step_id=step_c,
                    tool_name="explanation",
                    agent_name="InsightAgent",
                    purpose=f"Extract top drivers explaining changes in '{metric_target}'.",
                    inputs={"target": metric_target, "top_k": 3},
                    required_capabilities=["explanation"],
                    dependencies=[step_a, step_b],
                )
            )
            step_idx += 1

        elif primary_intent_val == "forecasting" or "forecasting" in req_caps:
            metric_target = user_intent.metrics[0] if user_intent.metrics else (knowledge.get_primary_metric() if knowledge else "revenue")
            periods = 5
            if user_intent.time_range and "month" in user_intent.time_range:
                periods = 6

            # If regional analysis or aggregation was requested before forecasting:
            if "regional_analysis" in req_caps or "aggregation" in req_caps:
                agg_step = f"step_{step_idx}"
                steps.append(
                    ExecutionStep(
                        step_id=agg_step,
                        tool_name="aggregation",
                        agent_name="AnalysisAgent",
                        purpose=f"Aggregate '{metric_target}' by dimensions.",
                        inputs={"metric": metric_target, "request": "summary"},
                        required_capabilities=["aggregation"],
                        dependencies=upstream_dep,
                    )
                )
                step_idx += 1
                upstream_dep = [agg_step]

            fc_step = f"step_{step_idx}"
            steps.append(
                ExecutionStep(
                    step_id=fc_step,
                    tool_name="forecasting",
                    agent_name="ForecastAgent",
                    purpose=f"Forecast '{metric_target}' for future periods.",
                    inputs={"target": metric_target, "periods": periods},
                    required_capabilities=["forecasting"],
                    dependencies=upstream_dep,
                )
            )
            step_idx += 1

        elif primary_intent_val in ("prediction", "classification", "regression") or "prediction" in req_caps:
            metric_target = user_intent.metrics[0] if user_intent.metrics else (knowledge.get_primary_metric() if knowledge else "target")
            pred_step = f"step_{step_idx}"
            steps.append(
                ExecutionStep(
                    step_id=pred_step,
                    tool_name="prediction",
                    agent_name="PredictionAgent",
                    purpose=f"Train and validate predictive model for '{metric_target}'.",
                    inputs={"target": metric_target, "features": user_intent.dimensions},
                    required_capabilities=["prediction"],
                    dependencies=upstream_dep,
                )
            )
            step_idx += 1

        elif primary_intent_val == "anomaly_detection" or "anomaly_detection" in req_caps:
            col_target = user_intent.metrics[0] if user_intent.metrics else (knowledge.get_primary_metric() if knowledge else None)
            anom_step = f"step_{step_idx}"
            steps.append(
                ExecutionStep(
                    step_id=anom_step,
                    tool_name="anomaly_detection",
                    agent_name="InsightAgent",
                    purpose=f"Identify anomalous spikes and records in '{col_target}'.",
                    inputs={"column": col_target},
                    required_capabilities=["anomaly_detection"],
                    dependencies=upstream_dep,
                )
            )
            step_idx += 1

        elif primary_intent_val == "comparison" or "comparison" in req_caps:
            metric_target = user_intent.metrics[0] if user_intent.metrics else (knowledge.get_primary_metric() if knowledge else "revenue")
            dim_target = user_intent.dimensions[0] if user_intent.dimensions else "country"
            comp_step = f"step_{step_idx}"
            steps.append(
                ExecutionStep(
                    step_id=comp_step,
                    tool_name="aggregation",
                    agent_name="AnalysisAgent",
                    purpose=f"Compare '{metric_target}' across '{dim_target}' entities.",
                    inputs={"metric": metric_target, "dimension": dim_target, "comparison": user_intent.comparison},
                    required_capabilities=["comparison"],
                    dependencies=upstream_dep,
                )
            )
            step_idx += 1

        elif primary_intent_val == "aggregation" or "aggregation" in req_caps or "regional_analysis" in req_caps:
            metric_target = user_intent.metrics[0] if user_intent.metrics else (knowledge.get_primary_metric() if knowledge else "revenue")
            dim_target = user_intent.dimensions[0] if user_intent.dimensions else (knowledge.get_primary_dimension() if knowledge else "country")
            agg_step = f"step_{step_idx}"
            steps.append(
                ExecutionStep(
                    step_id=agg_step,
                    tool_name="aggregation",
                    agent_name="AnalysisAgent",
                    purpose=f"Aggregate '{metric_target}' by '{dim_target}'.",
                    inputs={"metric": metric_target, "dimension": dim_target, "request": "summary"},
                    required_capabilities=["aggregation"],
                    dependencies=upstream_dep,
                )
            )
            step_idx += 1

        else:
            # Default EDA / General Summary
            metric_target = user_intent.metrics[0] if user_intent.metrics else (knowledge.get_primary_metric() if knowledge else None)
            eda_step = f"step_{step_idx}"
            steps.append(
                ExecutionStep(
                    step_id=eda_step,
                    tool_name="eda",
                    agent_name="AnalysisAgent",
                    purpose="Compute summary statistics and distributions.",
                    inputs={"request": "summary"},
                    required_capabilities=["eda"],
                    dependencies=upstream_dep,
                )
            )
            step_idx += 1

        # 4. Final Executive Narrative Synthesis Step
        all_prior_steps = [s.step_id for s in steps]
        report_step_id = f"step_{step_idx}"
        steps.append(
            ExecutionStep(
                step_id=report_step_id,
                tool_name="reporting",
                agent_name="ReportAgent",
                purpose="Synthesize multi-step outputs into structured report.",
                inputs={"request": "pipeline"},
                required_capabilities=["reporting"],
                dependencies=all_prior_steps,
            )
        )

        # Build dependencies map and required tools list
        for s in steps:
            deps_map[s.step_id] = list(s.dependencies)
        required_tools = list(dict.fromkeys(s.tool_name for s in steps))

        # Build and validate ExecutionPlan
        plan = ExecutionPlan(
            plan_id=plan_id,
            task_id=uuid.uuid4().hex[:8],
            user_intent=user_intent,
            objective=user_intent.objective or "Execute analytical pipeline.",
            steps=steps,
            dependencies=deps_map,
            estimated_complexity="high" if len(steps) >= 4 else "medium",
            confidence=user_intent.confidence,
            required_tools=required_tools,
            metadata={"step_count": len(steps)},
        )

        validation_errors = ExecutionGraph.validate_plan(plan, registry)
        if validation_errors:
            raise ValueError(f"Plan validation failed: {'; '.join(validation_errors)}")

        return plan

    # ------------------------------------------------------------------
    # Plan Execution Interface
    # ------------------------------------------------------------------
    def execute_plan(
        self,
        plan: Union[ExecutionPlan, TaskPlan],
        dataframe: pd.DataFrame,
    ) -> AgentResult:
        """Execute plan via ExecutionEngine with upfront validation."""
        if isinstance(plan, ExecutionPlan):
            val_errors = ExecutionGraph.validate_plan(plan, self.tool_registry)
            if val_errors:
                raise ValueError(f"Plan validation failed: {'; '.join(val_errors)}")
        return self.execution_engine.execute_plan(plan, dataframe)

    # ------------------------------------------------------------------
    # Legacy create_plan for Backwards Compatibility
    # ------------------------------------------------------------------
    def create_plan(
        self,
        query: str,
        dataframe: pd.DataFrame,
        knowledge: Optional[Any] = None,
    ) -> TaskPlan:
        """Legacy helper returning TaskPlan for older callers."""
        intent_res = self.intent_analyzer.analyze(query, knowledge=knowledge, dataframe=dataframe)
        
        # Build modern plan
        exec_plan = self.create_execution_plan(query, dataframe=dataframe, knowledge=knowledge)

        # Convert to legacy TaskPlan
        legacy_steps: List[PlanStep] = []
        for idx, s in enumerate(exec_plan.steps):
            legacy_steps.append(
                PlanStep(
                    step_id=idx + 1,
                    name=s.purpose or s.tool_name,
                    agent_class_name=s.agent_name or "AnalysisAgent",
                    action=s.tool_name,
                    parameters=s.inputs,
                    dependencies=[int(d.replace("step_", "")) for d in s.dependencies if d.startswith("step_") and d.replace("step_", "").isdigit()],
                    status=s.status,
                )
            )

        return TaskPlan(
            plan_id=exec_plan.plan_id,
            query=query,
            intent=intent_res,
            steps=legacy_steps,
            dataset_validation={},
        )
