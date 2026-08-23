"""
Base Agent - Defines the base class for all specialized agents.
"""
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .schemas import (
    AgentResult,
    AgentError,
    AgentStatus,
    Evidence,
    ErrorCategory,
    ValidationResult,
)


class BaseAgent:
    """Base class for all agents in the multi-agent system."""

    name = "Base Agent"
    description = "Base agent class"
    role = "generalist"

    def __init__(self, data=None):
        self.data = data
        self.agent_id = str(uuid.uuid4())[:8]
        self.status = AgentStatus.IDLE
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self.messages: List[str] = []

    def _start(self):
        """Mark the agent as started."""
        self.status = AgentStatus.WORKING
        self.started_at = datetime.now()
        self.messages.append(f"{self.name} started working.")

    def _finish(
        self,
        result: Dict[str, Any],
        evidence: List[Evidence] = None,
        confidence: float = 1.0,
        warnings: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> AgentResult:
        """Mark the agent as finished and return standardized AgentResult."""
        self.finished_at = datetime.now()
        self.status = AgentStatus.COMPLETED
        duration = round((self.finished_at - self.started_at).total_seconds() * 1000, 2) if self.started_at else 0
        self.messages.append(f"{self.name} completed in {duration}ms.")

        return AgentResult.success(
            agent=self.name,
            role=self.role,
            agent_id=self.agent_id,
            started_at=self.started_at,
            output=result,
            evidence=evidence or [],
            confidence=confidence,
            duration_ms=duration,
            warnings=warnings or [],
            metadata=metadata or {},
        )

    def _error(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        details: Dict[str, Any] = None,
        recoverable: bool = True,
        suggested_fix: str = None,
        fallback_agent: str = None,
        output: Dict[str, Any] = None,
    ) -> AgentResult:
        """Mark the agent as failed and return standardized AgentResult."""
        self.finished_at = datetime.now()
        self.status = AgentStatus.ERROR

        error = AgentError(
            category=category,
            message=message,
            details=details or {},
            recoverable=recoverable,
            suggested_fix=suggested_fix,
            fallback_agent=fallback_agent,
        )

        return AgentResult.failure(
            agent=self.name,
            role=self.role,
            agent_id=self.agent_id,
            started_at=self.started_at or datetime.now(),
            errors=[error],
            duration_ms=round((self.finished_at - self.started_at).total_seconds() * 1000, 2) if self.started_at else 0,
            output=output or {"error": message},
            warnings=self.messages + [f"{self.name} failed: {message}"],
        )

    def run(self, task) -> AgentResult:
        """Execute the task. Subclasses must override and return AgentResult."""
        raise NotImplementedError("Subclasses must implement run() and return AgentResult")

    # Backward compatibility: allow dict-like access for existing code
    def to_legacy_dict(self, result: AgentResult) -> Dict[str, Any]:
        """Convert AgentResult to legacy dict format for backward compatibility."""
        return result.to_dict()
