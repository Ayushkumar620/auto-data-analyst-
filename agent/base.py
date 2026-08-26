"""
Base Agent - Standardized execution contract for all multi-agent components.

Guarantees:
1. Every agent returns a standardized Pydantic AgentResult
2. Uniform lifecycle management (_start, _finish, _partial, _error)
3. Safe execution isolation that does not expose raw stack traces to users
4. Automatic execution time measurement and diagnostic metadata
"""
from __future__ import annotations

import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .schemas import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
    ValidationResult,
)


class BaseAgent:
    """Standardized base class for all specialized agents in the system."""

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

    def _start(self) -> None:
        """Mark the agent as started and record timestamp."""
        self.status = AgentStatus.WORKING
        self.started_at = datetime.now()
        self.messages.append(f"{self.name} started working.")

    def _calculate_duration(self) -> float:
        """Calculate elapsed execution time in milliseconds."""
        if self.started_at:
            end_t = self.finished_at or datetime.now()
            return round((end_t - self.started_at).total_seconds() * 1000, 2)
        return 0.0

    def _finish(
        self,
        result: Dict[str, Any],
        message: str = "",
        evidence: Optional[List[Evidence]] = None,
        confidence: float = 1.0,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_used: Optional[str] = None,
    ) -> AgentResult:
        """Mark the agent as successfully finished and return standardized AgentResult."""
        self.finished_at = datetime.now()
        self.status = AgentStatus.COMPLETED
        duration = self._calculate_duration()
        msg = message or f"{self.name} completed successfully in {duration}ms."
        self.messages.append(msg)

        return AgentResult(
            status=AgentStatus.COMPLETED,
            agent=self.name,
            agent_name=self.name,
            role=self.role,
            agent_id=self.agent_id,
            task_id=self.agent_id,
            execution_id=self.agent_id,
            started_at=self.started_at,
            finished_at=self.finished_at or datetime.now(),
            timestamp=self.started_at or datetime.now(),
            output=result,
            data=result,
            result=result,
            message=msg,
            evidence=evidence or [],
            confidence=confidence,
            duration_ms=duration,
            execution_time=duration,
            execution_time_ms=duration,
            warnings=(warnings or []) + [m for m in self.messages if "warning" in m.lower()],
            metadata=metadata or {},
            model_used=model_used,
        )

    def _partial(
        self,
        result: Dict[str, Any],
        message: str = "",
        evidence: Optional[List[Evidence]] = None,
        confidence: float = 0.5,
        errors: Optional[List[AgentError]] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_used: Optional[str] = None,
    ) -> AgentResult:
        """Mark the agent as partially finished and return standardized AgentResult."""
        self.finished_at = datetime.now()
        self.status = AgentStatus.PARTIAL
        duration = self._calculate_duration()
        msg = message or f"{self.name} partially completed in {duration}ms."
        self.messages.append(msg)

        return AgentResult(
            success=True,
            status=AgentStatus.PARTIAL,
            agent=self.name,
            agent_name=self.name,
            role=self.role,
            agent_id=self.agent_id,
            task_id=self.agent_id,
            execution_id=self.agent_id,
            started_at=self.started_at,
            timestamp=self.started_at or datetime.now(),
            output=result,
            data=result,
            result=result,
            message=msg,
            evidence=evidence or [],
            confidence=confidence,
            duration_ms=duration,
            execution_time=duration,
            execution_time_ms=duration,
            errors=errors or [],
            warnings=(warnings or []) + self.messages,
            metadata=metadata or {},
            model_used=model_used,
        )

    def _needs_clarification(
        self,
        clarification_message: str,
        options: List[Dict[str, Any]],
        task_type: Optional[str] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Return a structured clarification result when request is ambiguous."""
        self.finished_at = datetime.now()
        self.status = AgentStatus.NEEDS_CLARIFICATION
        return AgentResult.create_needs_clarification(
            agent_name=self.name,
            clarification_message=clarification_message,
            options=options,
            task_type=task_type,
            diagnostics=diagnostics,
        )

    def _not_supported(
        self,
        reason: str,
        task_type: Optional[str] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Return a structured not-supported result when dataset modality is incompatible."""
        self.finished_at = datetime.now()
        self.status = AgentStatus.NOT_SUPPORTED
        return AgentResult.create_not_supported(
            agent_name=self.name,
            reason=reason,
            task_type=task_type,
            diagnostics=diagnostics,
        )

    def _error(
        self,
        message: str,
        code: str = "COMPUTATION_ERROR",
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        suggested_fix: Optional[str] = None,
        fallback_agent: Optional[str] = None,
        output: Optional[Dict[str, Any]] = None,
        model_used: Optional[str] = None,
    ) -> AgentResult:
        """Mark the agent as failed with sanitized error and return standardized AgentResult."""
        self.finished_at = datetime.now()
        self.status = AgentStatus.ERROR
        duration = self._calculate_duration()

        # Sanitize message: never expose raw multi-line Python tracebacks in user message
        clean_message = message.split("\n")[-1].strip() if "\n" in message else message.strip()
        if not clean_message:
            clean_message = "An internal computational error occurred during agent execution."

        error = AgentError(
            code=code,
            category=category,
            message=clean_message,
            user_message=clean_message,
            technical_details=details or {},
            details=details or {},
            recoverable=recoverable,
            agent_name=self.name,
            suggested_action=suggested_fix,
            suggested_fix=suggested_fix,
            fallback_agent=fallback_agent,
        )

        return AgentResult(
            success=False,
            status=AgentStatus.ERROR,
            agent=self.name,
            agent_name=self.name,
            role=self.role,
            agent_id=self.agent_id,
            task_id=self.agent_id,
            execution_id=self.agent_id,
            started_at=self.started_at or datetime.now(),
            timestamp=self.started_at or datetime.now(),
            errors=[error],
            duration_ms=duration,
            execution_time=duration,
            execution_time_ms=duration,
            output=output or {"error": clean_message},
            data=output or {"error": clean_message},
            result=output or {"error": clean_message},
            message=f"{self.name} failed: {clean_message}",
            warnings=self.messages + [f"{self.name} error encountered."],
            model_used=model_used,
        )

    def run(self, task) -> AgentResult:
        """Execute the task. Subclasses must override and return AgentResult."""
        raise NotImplementedError("Subclasses must implement run() and return AgentResult")

    def safe_run(self, task: Any) -> AgentResult:
        """
        Safely execute the agent's run() method, catching unhandled exceptions
        without exposing raw internal tracebacks to the user.
        """
        self._start()
        try:
            res = self.run(task)
            if isinstance(res, AgentResult):
                return res
            elif isinstance(res, dict):
                return self._finish(res)
            else:
                return self._finish({"result": res})
        except Exception as exc:
            tb = traceback.format_exc()
            return self._error(
                message=f"Agent execution encountered an error: {str(exc)}",
                code="UNHANDLED_EXCEPTION",
                category=ErrorCategory.COMPUTATION,
                details={"exception_type": type(exc).__name__, "traceback": tb},
                recoverable=False,
            )

    def execute_with_retry(
        self,
        task: Dict[str, Any],
        max_retries: int = 2,
        retry_delay_ms: int = 50,
    ) -> AgentResult:
        """Execute the agent task with automated retry for recoverable errors."""
        attempts = 0
        last_result: Optional[AgentResult] = None

        while attempts <= max_retries:
            try:
                result = self.run(task)
                result.retry_count = attempts
                if result.is_success:
                    return result
                # If error is not recoverable, stop retrying
                if result.is_error:
                    has_recoverable = any(e.recoverable for e in result.errors)
                    if not has_recoverable or attempts >= max_retries:
                        return result
            except Exception as exc:
                tb = traceback.format_exc()
                err_result = self._error(
                    message=f"Execution error on attempt {attempts + 1}: {str(exc)}",
                    code="RETRY_EXCEPTION",
                    category=ErrorCategory.COMPUTATION,
                    details={"exception_type": type(exc).__name__, "traceback": tb},
                    recoverable=True,
                )
                err_result.retry_count = attempts
                if attempts >= max_retries:
                    return err_result

            attempts += 1
            if attempts <= max_retries:
                self.status = AgentStatus.RETRYING
                self.messages.append(f"{self.name} retrying attempt {attempts}/{max_retries}.")
                if retry_delay_ms > 0:
                    time.sleep(retry_delay_ms / 1000.0)

        return last_result or self._error("Max retries exceeded", category=ErrorCategory.COMPUTATION)

    # ------------------------------------------------------------------
    # Evidence helper
    # ------------------------------------------------------------------
    def make_evidence(
        self,
        method: str,
        data_ref: Dict[str, Any],
        confidence: float = 1.0,
        claim_type: ClaimType = ClaimType.OBSERVATION,
        raw_value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        dataset_name: Optional[str] = None,
        columns: Optional[List[str]] = None,
        operation: Optional[str] = None,
        calculation: Optional[str] = None,
    ) -> Evidence:
        """Create a standardized Evidence instance attributed to this agent."""
        cols = []
        if columns is not None:
            cols = list(columns) if isinstance(columns, (list, tuple, set)) else [str(columns)]
        elif isinstance(data_ref, dict):
            c_val = data_ref.get("column_names")
            if isinstance(c_val, (list, tuple, set)):
                cols = list(c_val)
            elif isinstance(c_val, str):
                cols = [c_val]

        op = operation or method
        return Evidence(
            source=self.name,
            source_reference=self.name,
            method=method,
            operation=op,
            calculation=calculation,
            dataset_name=dataset_name,
            dataset_id=dataset_name,
            columns=cols,
            data_ref=data_ref,
            confidence=confidence,
            claim_type=claim_type,
            raw_value=raw_value,
            result=raw_value,
            metadata=metadata or {},
        )

    def to_legacy_dict(self, result: AgentResult) -> Dict[str, Any]:
        """Convert AgentResult to legacy dict format for backward compatibility."""
        return result.to_dict()
