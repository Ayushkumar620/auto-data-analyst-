"""
ADAA Multi-Agent System (MAS)
=============================

A production-grade multi-agent architecture for automated data analysis.

Every major capability is a **specialized agent** with a clearly defined
responsibility, a *restricted* toolset (each agent receives only the tools
required for its job), explicit input/output schemas, a full execution
lifecycle, a validation step, and structured error handling.

A **Master Orchestrator** decomposes user requests into tasks, selects the
appropriate specialist agents, executes independent tasks in parallel when
possible, passes validated artifacts between dependent agents, recovers from
failures, and assembles the final response.

Public entry points
-------------------
- ``MasterOrchestrator`` — the top-level orchestrator (``agent.mas.orchestrator``)
- ``BaseAgent`` — base class for all agents (``agent.mas.base``)
- ``MasterToolRegistry`` — central tool registry (``agent.mas.tools.registry``)
- ``RequestPlanner`` — request decomposer (``agent.mas.planner``)
"""

from .orchestrator import MasterOrchestrator
from .base import BaseAgent
from .schemas import (
    Artifact,
    ArtifactType,
    AgentInput,
    AgentOutput,
    Task,
    TaskGraph,
    OrchestrationResult,
    ExecutionStatus,
)
from .errors import (
    AgentError,
    ValidationError,
    ToolAccessError,
    TaskExecutionError,
    OrchestrationError,
)

__all__ = [
    "MasterOrchestrator",
    "BaseAgent",
    "Artifact",
    "ArtifactType",
    "AgentInput",
    "AgentOutput",
    "Task",
    "TaskGraph",
    "OrchestrationResult",
    "ExecutionStatus",
    "AgentError",
    "ValidationError",
    "ToolAccessError",
    "TaskExecutionError",
    "OrchestrationError",
]
