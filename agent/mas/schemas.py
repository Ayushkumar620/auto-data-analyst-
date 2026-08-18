"""
Core schemas for the Multi-Agent System.

These dataclasses define the structured contracts that every agent and the
Master Orchestrator use to exchange information:

- ``Artifact`` -- a typed, named container for data that flows between agents
- ``AgentInput`` -- what an agent receives when ``execute()`` is called
- ``AgentOutput`` -- what an agent returns after execution (status, artifacts, logs)
- ``Task`` / ``TaskGraph`` -- the decomposed, dependency-ordered plan
- ``OrchestrationResult`` -- the final assembled result from the orchestrator
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExecutionStatus(str, Enum):
    """Lifecycle states an agent or task can be in."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


class ArtifactType(str, Enum):
    """Every artifact produced or consumed by an agent is tagged with one of these types."""
    # --- Data ---
    DATAFRAME = "dataframe"
    CLEANED_DATAFRAME = "cleaned_dataframe"
    DATA_PROFILE = "data_profile"

    # --- Cleaning ---
    CLEANING_REPORT = "cleaning_report"

    # --- EDA ---
    EDA_SUMMARY = "eda_summary"
    EDA_STATISTICS = "eda_statistics"
    EDA_CORRELATIONS = "eda_correlations"
    EDA_ANOMALIES = "eda_anomalies"
    EDA_DISTRIBUTIONS = "eda_distributions"
    EDA_CATEGORICAL = "eda_categorical"
    EDA_TIME_SERIES = "eda_time_series"
    EDA_RECOMMENDATIONS = "eda_recommendations"

    # --- Visualization ---
    CHART_RECOMMENDATIONS = "chart_recommendations"
    CHARTS = "charts"

    # --- Insights ---
    FACTS = "facts"
    INSIGHTS = "insights"
    RECOMMENDATIONS = "recommendations"

    # --- Forecasting ---
    FORECAST_RESULT = "forecast_result"

    # --- Reports ---
    REPORT = "report"
    REPORT_PDF = "report_pdf"

    # --- Chat ---
                CHAT_RESPONSE = "chat_response"
    CHAT_HISTORY = "chat_history"


@dataclass
class Artifact:
    """A typed data container passed between agents.

    Parameters
    ----------
    type : str
        The semantic type of the artifact (a value from ``ArtifactType``).
    name : str
        Human-readable name, e.g. ``'raw_dataframe'`` or ``'cleaned_df'``.
    payload : Any
        The actual data -- typically a ``pd.DataFrame``, a ``dict`` of
        analysis results, or a result object.
    metadata : dict
        Extra context: column names, dtypes, row counts, quality scores, etc.
    producer : str
        Name of the agent that produced this artifact.
    """
    type: str
    name: str
    payload: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    producer: str = ""

    def to_dict_summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary (no heavy payloads)."""
        return {
            "type": self.type,
            "name": self.name,
            "metadata": self.metadata,
            "producer": self.producer,
        }


@dataclass
class AgentInput:
    """Structured input delivered to an agent's ``execute()`` method.

    Parameters
    ----------
    task : str
        The action the agent should perform (e.g. ``'clean'``, ``'summarize'``).
    artifacts : Dict[str, Artifact]
        Artifacts produced by upstream agents that this agent may consume.
        Keys are artifact *names* (matching ``Artifact.name``).
    params : Dict[str, Any]
        Task-specific parameters (e.g. ``{'target': 'revenue', 'periods': 5}``).
    data_summary : Dict[str, Any]
        Lightweight description of the dataset (columns, dtypes, shape) so
        the agent can make decisions without holding the full DataFrame.
    """
    task: str
    artifacts: Dict[str, Artifact] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    data_summary: Dict[str, Any] = field(default_factory=dict)

    def get_artifact(self, name: str):
        """Retrieve an artifact by name, or ``None`` if not present."""
        return self.artifacts.get(name)

    def get_dataframe(self):
        """Convenience: find the first DataFrame payload among input artifacts."""
        import pandas as pd
        for art in self.artifacts.values():
            if isinstance(art.payload, pd.DataFrame):
                return art.payload
        return None


@dataclass
class AgentOutput:
    """Structured output returned by every agent."""
    agent_name: str
    role: str
    status: str
    artifacts: List[Artifact] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "role": self.role,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "messages": self.messages,
            "error": self.error,
            "artifacts": [a.to_dict_summary() for a in self.artifacts],
        }


@dataclass
class Task:
    """A single unit of work in the decomposed plan."""
    id: str
    agent: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    artifact_inputs: List[str] = field(default_factory=list)
    artifact_outputs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent,
            "action": self.action,
            "params": self.params,
            "depends_on": self.depends_on,
            "artifact_inputs": self.artifact_inputs,
            "artifact_outputs": self.artifact_outputs,
        }


@dataclass
class TaskGraph:
    """Dependency-ordered graph of tasks."""
    tasks: Dict[str, Task] = field(default_factory=dict)
    root_ids: List[str] = field(default_factory=list)
    levels: List[List[str]] = field(default_factory=list)

    def all_task_ids(self) -> List[str]:
        return list(self.tasks.keys())

    def get(self, task_id: str):
        return self.tasks.get(task_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
            "root_ids": self.root_ids,
            "levels": self.levels,
        }


@dataclass
class TaskResult:
    """The result of executing a single task."""
    task_id: str
    task: Dict[str, Any]
    output: AgentOutput

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "result": self.output.to_dict(),
        }


@dataclass
class OrchestrationResult:
    """The final assembled result returned by the Master Orchestrator."""
    status: str
    request: str
    task_results: List[TaskResult] = field(default_factory=list)
    artifacts: Dict[str, Artifact] = field(default_factory=dict)
    summary: str = ""
    parallel_groups: List[int] = field(default_factory=list)
    duration_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "request": self.request,
            "task_results": [r.to_dict() for r in self.task_results],
            "artifacts": {name: art.to_dict_summary() for name, art in self.artifacts.items()},
            "summary": self.summary,
            "parallel_groups": self.parallel_groups,
            "duration_ms": self.duration_ms,
            "errors": self.errors,
        }
