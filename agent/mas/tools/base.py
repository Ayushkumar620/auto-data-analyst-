"""Base classes for the tool system.

A **Tool** wraps a callable with metadata so agents can describe what they
intend to do before executing.  The :class:`MasterToolRegistry`` centralises all
available tools and can hand each agent a *restricted* :class:`AgentToolSet`
that contains only the tools relevant to that agent's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class Tool:
    """Descriptor for a single callable tool.

    Parameters
    ----------
    name : str
        Unique identifier, e.g. ``'clean_data'``.
    description : str
        Human-readable description of what the tool does.
    func : Callable
        The underlying function.  Called as ``func(**params)``.
    input_schema : dict
        Describes required/optional parameters and their types.
        Example:: ``{"target": {"type": "str", "required": True}}``
    output_type : str
        A short label for the return value (e.g. ``'dataframe'``).
    """
    name: str
    description: str
    func: Callable[..., Any]
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_type: str = "any"


@dataclass
class ToolResult:
    """The outcome of executing a tool."""
    success: bool
    data: Any = None
    error: str = ""

    @classmethod
    def ok(cls, data: Any) -> "ToolResult":
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        return cls(success=False, error=error)
