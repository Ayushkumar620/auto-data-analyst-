"""Core schemas for the Multi-Agent System (MAS): Artifact, AgentInput,
AgentOutput, Task, TaskGraph, TaskResult, OrchestrationResult."""
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
    DATAFRAME = "dataframe"
    CLEANED_DATAFRAME = "cleaned_dataframe"
    DATA_PROFILE = "data_profile"
    CLEANING_REPORT = "cleaning_report"
    EDA_SUMMARY = "eda_summary"
    EDA_STATISTICS = "eda_statistics"
    EDA_CORRELATIONS = "eda_correlations"
    EDA_ANOMALIES = "eda_anomalies"
    EDA_DISTRIBUTIONS = "eda_distributions"
    EDA_CATEGORICAL = "eda_categorical"
    EDA_TIME_SERIES = "eda_time_series"
    EDA_RECOMMENDATIONS = "eda_recommendations"
    CHART_RECOMMENDATIONS = "chart_recommendations"
    CHARTS = "charts"
    FACTS = "facts"
    INSIGHTS = "insights"
    RECOMMENDATIONS = "recommendations"
    FORECAST_RESULT = "forecast_result"
    REPORT = "report"
    REPORT_PDF = "report_pdf"
    CHAT_RESPONSE = "chat_response"
    CHAT_HISTORY = "chat_history"


@dataclass
class Artifact:
    """A typed data container passed between agents."""
    type: str
    name: str
    payload: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    producer: str = ""

    def to_dict_summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary (no heavy payloads)."""
        return {
            "type": self.type, "name": self.name,
            "metadata": self.metadata, "producer": self.producer,
        }


@dataclass
class AgentInput:
    """Structured input delivered to an agent's execute() method."""
    task: str
    artifacts: Dict[str, Artifact] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    data_summary: Dict[str, Any] = field(default_factory=dict)

    def get_artifact(self, name: str):
        """Retrieve an artifact by name, or None if not present."""
        return self.artifacts.get(name)

    def get_dataframe(self):
        """Find the first DataFrame payload among input artifacts."""
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
            "agent": self.agent_name, "role": self.role,
            "status": self.status, "duration_ms": self.duration_ms,
            "messages": self.messages, "error": self.error,
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
            "id": self.id, "agent": self.agent, "action": self.action,
            "params": self.params, "depends_on": self.depends_on,
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
            "root_ids": self.root_ids, "levels": self.levels,
        }


@dataclass
class TaskResult:
    """The result of executing a single task."""
    task_id: str
    task: Dict[str, Any]
    output: AgentOutput

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id, "task": self.task,
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
            "status": self.status, "request": self.request,
            "task_results": [r.to_dict() for r in self.task_results],
            "artifacts": {name: a.to_dict_summary()
                          for name, a in self.artifacts.items()},
            "summary": self.summary, "parallel_groups": self.parallel_groups,
            "duration_ms": self.duration_ms, "errors": self.errors,
        }
