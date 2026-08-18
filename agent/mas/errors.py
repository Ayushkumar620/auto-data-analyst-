"""Exception hierarchy for the Multi-Agent System.

All custom exceptions derive from :class:`AgentError` so that the Master
Orchestrator can catch any agent-related failure with a single ``except``
block while still distinguishing specific failure modes when needed.
"""
from __future__ import annotations


class AgentError(Exception):
    """Base exception for all multi-agent-system errors."""


class ValidationError(AgentError):
    """Raised when agent input or output fails schema validation."""


class ToolAccessError(AgentError):
    """Raised when an agent attempts to call a tool it does not have access to."""


class ToolExecutionError(AgentError):
    """Raised when a tool itself fails during execution."""


class TaskExecutionError(AgentError):
    """Raised when a task cannot be completed by its assigned agent."""


class OrchestrationError(AgentError):
    """Raised for errors in the Master Orchestrator (planning, graph, assembly)."""
