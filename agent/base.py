"""
Base Agent - Defines the base class for all specialized agents.
"""
import time
import uuid


class BaseAgent:
    """Base class for all agents in the multi-agent system."""

    name = "Base Agent"
    description = "Base agent class"
    role = "generalist"

    def __init__(self, data=None):
        self.data = data
        self.agent_id = str(uuid.uuid4())[:8]
        self.status = "idle"
        self.started_at = None
        self.finished_at = None
        self.messages = []

    def _start(self):
        """Mark the agent as started."""
        self.status = "working"
        self.started_at = time.time()
        self.messages.append(f"{self.name} started working.")

    def _finish(self, result):
        """Mark the agent as finished and attach timing."""
        self.finished_at = time.time()
        self.status = "completed"
        duration = round((self.finished_at - self.started_at) * 1000, 2) if self.started_at else 0
        self.messages.append(f"{self.name} completed in {duration}ms.")
        return {
            "agent": self.name,
            "role": self.role,
            "agent_id": self.agent_id,
            "status": "completed",
            "duration_ms": duration,
            "messages": self.messages,
            "output": result,
        }

    def _error(self, message):
        """Mark the agent as failed."""
        self.status = "error"
        self.finished_at = time.time()
        return {
            "agent": self.name,
            "role": self.role,
            "agent_id": self.agent_id,
            "status": "error",
            "messages": self.messages + [f"{self.name} failed: {message}"],
            "output": {"error": message},
        }

    def run(self, task):
        """Execute the task. Subclasses must override."""
        raise NotImplementedError("Subclasses must implement run()")
