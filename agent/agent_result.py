"""
Universal Canonical Agent Result Contract & Error Architecture.

Defines standardized Pydantic data contracts for all analytical agents,
ensuring consistent, verifiable outputs with:
- Standardized lifecycle statuses (SUCCESS, PARTIAL, FAILED, NEEDS_CLARIFICATION, NOT_SUPPORTED, VALIDATION_FAILED)
- Granular error categories with safe user-facing messages
- Dynamic confidence and prediction uncertainty separation
- Traceable evidence and provenance
- Backward and forward compatibility with legacy agent outputs
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentStatus(str, Enum):
    """Standardized agent execution and lifecycle status."""
    IDLE = "idle"
    WORKING = "working"
    SUCCESS = "success"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    ERROR = "error"
    NEEDS_CLARIFICATION = "needs_clarification"
    NOT_SUPPORTED = "not_supported"
    VALIDATION_FAILED = "validation_failed"
    RETRYING = "retrying"


class ErrorCategory(str, Enum):
    """Controlled categorization of agent errors for routing and diagnostics."""
    INPUT_INVALID = "input_invalid"
    DATA_INVALID = "data_invalid"
    TARGET_NOT_FOUND = "target_not_found"
    TIME_COLUMN_NOT_FOUND = "time_column_not_found"
    INSUFFICIENT_DATA = "insufficient_data"
    UNSUPPORTED_TASK = "unsupported_task"
    MODEL_FAILURE = "model_failure"
    VALIDATION_FAILURE = "validation_failure"
    EXECUTION_FAILURE = "execution_failure"
    DEPENDENCY_FAILURE = "dependency_failure"
    AMBIGUOUS_REQUEST = "ambiguous_request"
    INTERNAL_ERROR = "internal_error"

    # Legacy category aliases
    INPUT_VALIDATION = "input_validation"
    DATA_QUALITY = "data_quality"
    COMPUTATION = "computation"
    RESOURCE = "resource"
    SEMANTIC = "semantic"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


class ValidationSeverity(str, Enum):
    """Severity of validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ClaimType(str, Enum):
    """Epistemic nature of evidence claims."""
    FACT = "fact"
    OBSERVATION = "observation"
    CORRELATION = "correlation"
    INFERENCE = "inference"
    RECOMMENDATION = "recommendation"


# ---------------------------------------------------------------------------
# Evidence & Error Models
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    """Traceable proof supporting a claim or analytical computation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    operation: str = ""
    calculation: Optional[str] = None
    source_reference: Optional[str] = None
    result: Any = None
    confidence: float = Field(default=1.0)

    # Backward compatibility fields
    source: Optional[str] = None
    method: Optional[str] = None
    data_ref: Dict[str, Any] = Field(default_factory=dict)
    claim_type: Union[ClaimType, str] = ClaimType.OBSERVATION
    raw_value: Any = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        val = float(v)
        if val < 0.0 or val > 1.0:
            raise ValueError(f"confidence must be within [0.0, 1.0], got {val}")
        return val

    @model_validator(mode="before")
    @classmethod
    def sync_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "source" in data and not data.get("source_reference"):
                data["source_reference"] = data["source"]
            elif "source_reference" in data and not data.get("source"):
                data["source"] = data["source_reference"]

            if "method" in data and not data.get("operation"):
                data["operation"] = data["method"]
            elif "operation" in data and not data.get("method"):
                data["method"] = data["operation"]

            if "calculation" in data and not data.get("operation"):
                data["operation"] = str(data["calculation"])

            if "raw_value" in data and data.get("result") is None:
                data["result"] = data["raw_value"]
            elif "result" in data and data.get("raw_value") is None:
                data["raw_value"] = data["result"]

            if "dataset_name" in data and not data.get("dataset_id"):
                data["dataset_id"] = data["dataset_name"]
            elif "dataset_id" in data and not data.get("dataset_name"):
                data["dataset_name"] = data["dataset_id"]
        return data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id or self.dataset_name,
            "dataset_name": self.dataset_name or self.dataset_id,
            "columns": self.columns,
            "operation": self.operation or self.method,
            "calculation": self.calculation,
            "source_reference": self.source_reference or self.source,
            "result": self.result if self.result is not None else self.raw_value,
            "confidence": round(float(self.confidence), 4),
            "source": self.source or self.source_reference or "",
            "method": self.method or self.operation or "",
            "data_ref": self.data_ref,
            "claim_type": self.claim_type.value if isinstance(self.claim_type, ClaimType) else str(self.claim_type),
            "raw_value": self.raw_value if self.raw_value is not None else self.result,
            "metadata": self.metadata,
        }


class AgentError(BaseModel):
    """Standardized error contract with user safety and debugging isolation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    code: str = "INTERNAL_ERROR"
    category: Union[ErrorCategory, str] = ErrorCategory.INTERNAL_ERROR
    message: str = ""
    user_message: str = ""
    technical_details: Dict[str, Any] = Field(default_factory=dict)
    recoverable: bool = False
    retryable: bool = False
    field: Optional[str] = None
    suggested_action: Optional[str] = None
    agent_name: Optional[str] = None

    # Backward compatibility aliases
    details: Dict[str, Any] = Field(default_factory=dict)
    suggested_fix: Optional[str] = None
    fallback_agent: Optional[str] = None
    retry_after_ms: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def sync_error_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "message" in data and not data.get("user_message"):
                msg = str(data["message"])
                clean_msg = msg.split("\n")[0] if "Traceback" in msg else msg
                data["user_message"] = clean_msg
            elif "user_message" in data and not data.get("message"):
                data["message"] = str(data["user_message"])

            if "details" in data and not data.get("technical_details"):
                data["technical_details"] = data["details"] if isinstance(data["details"], dict) else {"details": data["details"]}
            elif "technical_details" in data and not data.get("details"):
                data["details"] = data["technical_details"]

            if "suggested_fix" in data and not data.get("suggested_action"):
                data["suggested_action"] = data["suggested_fix"]
            elif "suggested_action" in data and not data.get("suggested_fix"):
                data["suggested_fix"] = data["suggested_action"]

            if "category" in data and ("code" not in data or data["code"] == "INTERNAL_ERROR"):
                cat = data["category"]
                data["code"] = cat.value if isinstance(cat, ErrorCategory) else str(cat).upper()
        return data

    def to_dict(self) -> Dict[str, Any]:
        cat_str = self.category.value if isinstance(self.category, ErrorCategory) else str(self.category)
        return {
            "code": self.code,
            "category": cat_str,
            "message": self.message,
            "user_message": self.user_message or self.message,
            "technical_details": self.technical_details or self.details,
            "recoverable": self.recoverable,
            "retryable": self.retryable,
            "field": self.field,
            "suggested_action": self.suggested_action or self.suggested_fix,
            "agent_name": self.agent_name,
            "details": self.details or self.technical_details,
            "suggested_fix": self.suggested_fix or self.suggested_action,
            "fallback_agent": self.fallback_agent,
            "retry_after_ms": self.retry_after_ms,
        }

    @classmethod
    def create(
        cls,
        category: Union[ErrorCategory, str],
        user_message: str,
        technical_details: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
        retryable: bool = False,
        suggested_action: Optional[str] = None,
        field: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> "AgentError":
        cat_enum = category if isinstance(category, ErrorCategory) else ErrorCategory(str(category).lower())
        return cls(
            code=cat_enum.value.upper(),
            category=cat_enum,
            message=user_message,
            user_message=user_message,
            technical_details=technical_details or {},
            recoverable=recoverable,
            retryable=retryable,
            suggested_action=suggested_action,
            field=field,
            agent_name=agent_name,
        )


class ValidationIssue(BaseModel):
    """A detected anomaly or contract violation in an AgentResult."""
    severity: ValidationSeverity = ValidationSeverity.ERROR
    code: str
    message: str
    field: Optional[str] = None
    expected: Any = None
    actual: Any = None
    repair_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value if isinstance(self.severity, ValidationSeverity) else str(self.severity),
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "repair_hint": self.repair_hint,
        }


class ValidationResult(BaseModel):
    """Outcome of pre-execution or post-execution validation."""
    passed: bool = True
    issues: List[ValidationIssue] = Field(default_factory=list)
    repaired: bool = False
    repair_actions: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() if isinstance(i, ValidationIssue) else i for i in self.issues],
            "repaired": self.repaired,
            "repair_actions": self.repair_actions,
        }

    def add_issue(
        self,
        severity: ValidationSeverity,
        code: str,
        message: str,
        field: Optional[str] = None,
        expected: Any = None,
        actual: Any = None,
        repair_hint: Optional[str] = None,
    ) -> None:
        self.issues.append(ValidationIssue(
            severity=severity,
            code=code,
            message=message,
            field=field,
            expected=expected,
            actual=actual,
            repair_hint=repair_hint,
        ))
        if severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.passed = False


# ---------------------------------------------------------------------------
# Canonical Universal Agent Result
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    """
    Authoritative, standardized Pydantic data contract for all analytical agents.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    status: Union[AgentStatus, str] = Field(default=AgentStatus.SUCCESS)
    task_type: Optional[str] = None
    agent_name: str = Field(default="Unknown Agent")
    dataset_id: Optional[str] = None
    target: Optional[str] = None
    execution_id: str = Field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:8]}")
    result: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=1.0)
    warnings: List[str] = Field(default_factory=list)
    errors: List[AgentError] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    model_info: Optional[Dict[str, Any]] = None
    provenance: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)

    # Backward compatibility mirror fields
    data: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None
    task_id: str = Field(default="")
    agent_id: Optional[str] = None
    agent: Optional[str] = None
    role: str = "generalist"
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    execution_time: float = Field(default=0.0)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    validation: Optional[ValidationResult] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_used: Optional[str] = None
    retry_count: int = 0

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        val = float(v)
        if val < 0.0 or val > 1.0:
            raise ValueError(f"confidence must be within [0.0, 1.0], got {val}")
        return val

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Union[AgentStatus, str]) -> Union[AgentStatus, str]:
        valid_statuses = {
            "success", "completed", "partial", "failed", "error", "working", "idle",
            "retrying", "validation_failed", "needs_clarification", "not_supported"
        }
        val_str = v.value if isinstance(v, AgentStatus) else str(v).lower()
        if val_str not in valid_statuses:
            raise ValueError(f"Invalid status '{v}'. Must be one of {valid_statuses}")
        return v

    @model_validator(mode="before")
    @classmethod
    def sync_legacy_and_modern_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Sync result / data / output
            res = data.get("result") or data.get("data") or data.get("output") or {}
            if not isinstance(res, dict):
                res = {"value": res}
            data["result"] = res
            data["data"] = res
            data["output"] = res

            # Sync execution_id / task_id / agent_id
            ex_id = data.get("execution_id") or data.get("task_id") or data.get("agent_id") or f"exec_{uuid.uuid4().hex[:8]}"
            data["execution_id"] = str(ex_id)
            data["task_id"] = str(ex_id)
            data["agent_id"] = str(ex_id)

            # Sync agent_name / agent
            ag_name = data.get("agent_name") or data.get("agent") or "Unknown Agent"
            data["agent_name"] = str(ag_name)
            data["agent"] = str(ag_name)

            # Sync execution_time_ms / duration_ms / execution_time
            dur = data.get("execution_time_ms") or data.get("duration_ms") or data.get("execution_time") or 0.0
            data["execution_time_ms"] = float(dur)
            data["duration_ms"] = float(dur)
            data["execution_time"] = float(dur)

            # Sync model_info / model_used
            if "model_used" in data and not data.get("model_info"):
                data["model_info"] = {"name": data["model_used"]}
            elif "model_info" in data and isinstance(data["model_info"], dict) and not data.get("model_used"):
                data["model_used"] = data["model_info"].get("name") or data["model_info"].get("model_name")
        return data

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.agent:
            self.agent = self.agent_name
        if not self.agent_id:
            self.agent_id = self.task_id
        if self.output is None:
            self.output = self.data

    @property
    def is_success(self) -> bool:
        st = self.status.value if isinstance(self.status, AgentStatus) else str(self.status).lower()
        return st in ("success", "completed")

    @property
    def is_partial(self) -> bool:
        st = self.status.value if isinstance(self.status, AgentStatus) else str(self.status).lower()
        return st == "partial"

    @property
    def is_needs_clarification(self) -> bool:
        st = self.status.value if isinstance(self.status, AgentStatus) else str(self.status).lower()
        return st == "needs_clarification"

    @property
    def is_not_supported(self) -> bool:
        st = self.status.value if isinstance(self.status, AgentStatus) else str(self.status).lower()
        return st == "not_supported"

    @property
    def is_error(self) -> bool:
        st = self.status.value if isinstance(self.status, AgentStatus) else str(self.status).lower()
        return st in ("failed", "error", "validation_failed")

    @property
    def has_evidence(self) -> bool:
        return len(self.evidence) > 0

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def error_message(self) -> Optional[str]:
        if self.errors:
            return self.errors[0].user_message or self.errors[0].message
        return None

    def add_error(self, error: AgentError) -> None:
        self.errors.append(error)
        self.status = AgentStatus.ERROR

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)

    def add_evidence(self, evidence: Evidence) -> None:
        self.evidence.append(evidence)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "success":
            return self.is_success
        if key == "status":
            return self.status.value if isinstance(self.status, AgentStatus) else str(self.status)
        if key in ("output", "data", "result"):
            return self.result if self.result is not None else (self.data or {})
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is not None or hasattr(self, key):
            return val
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in ("success", "status", "output", "data", "result")

    def to_dict(self) -> Dict[str, Any]:
        st_val = self.status.value if isinstance(self.status, AgentStatus) else str(self.status)
        return {
            "success": self.is_success,
            "status": st_val,
            "task_type": self.task_type,
            "agent_name": self.agent_name,
            "dataset_id": self.dataset_id,
            "target": self.target,
            "execution_id": self.execution_id,
            "result": self.result,
            "metrics": self.metrics,
            "evidence": [e.to_dict() if isinstance(e, Evidence) else e for e in self.evidence],
            "confidence": round(float(self.confidence), 4),
            "warnings": self.warnings,
            "errors": [e.to_dict() if isinstance(e, AgentError) else e for e in self.errors],
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "diagnostics": self.diagnostics,
            "model_info": self.model_info,
            "provenance": self.provenance,
            "execution_time_ms": round(float(self.execution_time_ms), 2),
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else str(self.timestamp),
            # Legacy compatibility mappings
            "data": self.result,
            "output": self.result,
            "task_id": self.execution_id,
            "agent_id": self.execution_id,
            "agent": self.agent_name,
            "role": self.role,
            "message": self.message,
            "duration_ms": round(float(self.execution_time_ms), 2),
            "execution_time": round(float(self.execution_time_ms), 2),
            "model_used": self.model_used,
            "metadata": self.metadata,
            "validation": self.validation.to_dict() if self.validation else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentResult":
        """Deserialize AgentResult from dictionary representation."""
        return cls.model_validate(data)

    # ------------------------------------------------------------------
    # Standardized Factory Methods
    # ------------------------------------------------------------------

    @classmethod
    def create_success(
        cls,
        agent_name: str,
        result: Dict[str, Any],
        task_type: Optional[str] = None,
        target: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        evidence: Optional[List[Evidence]] = None,
        confidence: float = 1.0,
        warnings: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        model_info: Optional[Dict[str, Any]] = None,
        execution_time_ms: float = 0.0,
        message: str = "",
    ) -> "AgentResult":
        return cls(
            status=AgentStatus.SUCCESS,
            task_type=task_type,
            agent_name=agent_name,
            target=target,
            result=result,
            metrics=metrics or {},
            evidence=evidence or [],
            confidence=confidence,
            warnings=warnings or [],
            assumptions=assumptions or [],
            limitations=limitations or [],
            diagnostics=diagnostics or {},
            model_info=model_info,
            execution_time_ms=execution_time_ms,
            message=message or f"{agent_name} completed successfully.",
        )

    @classmethod
    def create_error(
        cls,
        agent_name: str,
        error: AgentError,
        task_type: Optional[str] = None,
        target: Optional[str] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        execution_time_ms: float = 0.0,
    ) -> "AgentResult":
        return cls(
            status=AgentStatus.ERROR,
            task_type=task_type,
            agent_name=agent_name,
            target=target,
            result={"error": error.user_message or error.message},
            errors=[error],
            confidence=0.0,
            diagnostics=diagnostics or error.technical_details,
            execution_time_ms=execution_time_ms,
            message=error.user_message or error.message,
        )

    @classmethod
    def success(
        cls,
        agent: str = "Agent",
        agent_name: Optional[str] = None,
        role: str = "generalist",
        agent_id: str = "",
        task_id: Optional[str] = None,
        started_at: Optional[datetime] = None,
        timestamp: Optional[datetime] = None,
        output: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        message: str = "Operation completed successfully.",
        evidence: Optional[List[Evidence]] = None,
        confidence: float = 1.0,
        duration_ms: float = 0.0,
        execution_time: Optional[float] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_used: Optional[str] = None,
        **kwargs: Any,
    ) -> "AgentResult":
        name = agent_name or agent
        t_id = task_id or agent_id
        dt = data if data is not None else (output or {})
        exec_t = execution_time if execution_time is not None else duration_ms
        ts = timestamp or started_at or datetime.now()

        return cls(
            status=AgentStatus.COMPLETED,
            agent_name=name,
            agent=name,
            role=role,
            task_id=t_id,
            agent_id=t_id,
            execution_id=t_id,
            data=dt,
            output=dt,
            result=dt,
            message=message,
            started_at=ts,
            timestamp=ts,
            finished_at=datetime.now(),
            duration_ms=exec_t,
            execution_time=exec_t,
            execution_time_ms=exec_t,
            evidence=evidence or [],
            confidence=confidence,
            warnings=warnings or [],
            metadata=metadata or {},
            model_used=model_used,
            **kwargs,
        )

    @classmethod
    def failure(
        cls,
        agent: str = "Agent",
        agent_name: Optional[str] = None,
        role: str = "generalist",
        agent_id: str = "",
        task_id: Optional[str] = None,
        started_at: Optional[datetime] = None,
        timestamp: Optional[datetime] = None,
        errors: Optional[List[AgentError]] = None,
        duration_ms: float = 0.0,
        execution_time: Optional[float] = None,
        output: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        message: str = "Operation failed.",
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_used: Optional[str] = None,
        **kwargs: Any,
    ) -> "AgentResult":
        name = agent_name or agent
        t_id = task_id or agent_id
        dt = data if data is not None else (output or {})
        exec_t = execution_time if execution_time is not None else duration_ms
        ts = timestamp or started_at or datetime.now()

        return cls(
            status=AgentStatus.ERROR,
            agent_name=name,
            agent=name,
            role=role,
            task_id=t_id,
            agent_id=t_id,
            execution_id=t_id,
            data=dt,
            output=dt,
            result=dt,
            message=message,
            started_at=ts,
            timestamp=ts,
            finished_at=datetime.now(),
            duration_ms=exec_t,
            execution_time=exec_t,
            execution_time_ms=exec_t,
            evidence=[],
            confidence=0.0,
            errors=errors or [],
            warnings=warnings or [],
            metadata=metadata or {},
            model_used=model_used,
            **kwargs,
        )

    @classmethod
    def error(
        cls,
        error: Union[str, AgentError] = "",
        agent: str = "Agent",
        agent_name: Optional[str] = None,
        role: str = "generalist",
        code: str = "ERROR",
        details: Optional[Dict[str, Any]] = None,
        task_id: str = "",
        message: Optional[str] = None,
        errors: Optional[List[AgentError]] = None,
        **kwargs: Any,
    ) -> "AgentResult":
        name = agent_name or agent
        err_msg = str(error) if error else (message or "Error")
        err_list = errors or []
        if not err_list:
            err_obj = error if isinstance(error, AgentError) else AgentError(
                code=code,
                message=err_msg,
                user_message=err_msg,
                details=details or {},
                technical_details=details or {},
                agent_name=name,
            )
            err_list = [err_obj]
        return cls(
            status=AgentStatus.ERROR,
            agent_name=name,
            agent=name,
            role=role,
            task_id=task_id,
            agent_id=task_id,
            execution_id=task_id or f"exec_{uuid.uuid4().hex[:8]}",
            data={"error": err_msg},
            output={"error": err_msg},
            result={"error": err_msg},
            message=message or f"{name} failed: {err_msg}",
            errors=err_list,
            confidence=0.0,
            **kwargs,
        )

    @classmethod
    def partial(
        cls,
        agent: str = "Agent",
        agent_name: Optional[str] = None,
        role: str = "generalist",
        agent_id: str = "",
        task_id: Optional[str] = None,
        started_at: Optional[datetime] = None,
        timestamp: Optional[datetime] = None,
        output: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        message: str = "Operation partially completed.",
        evidence: Optional[List[Evidence]] = None,
        confidence: float = 0.5,
        duration_ms: float = 0.0,
        execution_time: Optional[float] = None,
        errors: Optional[List[AgentError]] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_used: Optional[str] = None,
        **kwargs: Any,
    ) -> "AgentResult":
        name = agent_name or agent
        t_id = task_id or agent_id
        dt = data if data is not None else (output or {})
        exec_t = execution_time if execution_time is not None else duration_ms
        ts = timestamp or started_at or datetime.now()

        return cls(
            status=AgentStatus.PARTIAL,
            agent_name=name,
            agent=name,
            role=role,
            task_id=t_id,
            agent_id=t_id,
            execution_id=t_id,
            data=dt,
            output=dt,
            result=dt,
            message=message,
            started_at=ts,
            timestamp=ts,
            finished_at=datetime.now(),
            duration_ms=exec_t,
            execution_time=exec_t,
            execution_time_ms=exec_t,
            evidence=evidence or [],
            confidence=confidence,
            errors=errors or [],
            warnings=warnings or [],
            metadata=metadata or {},
            model_used=model_used,
            **kwargs,
        )

    @classmethod
    def create_needs_clarification(
        cls,
        agent_name: str,
        clarification_message: str,
        options: List[Dict[str, Any]],
        task_type: Optional[str] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> "AgentResult":
        err = AgentError.create(
            category=ErrorCategory.AMBIGUOUS_REQUEST,
            user_message=clarification_message,
            technical_details={"options": options},
            suggested_action="Please select one of the suggested targets or specify your goal explicitly.",
            agent_name=agent_name,
        )
        return cls(
            status=AgentStatus.NEEDS_CLARIFICATION,
            task_type=task_type,
            agent_name=agent_name,
            result={"message": clarification_message, "options": options},
            errors=[err],
            confidence=0.5,
            diagnostics=diagnostics or {"options": options},
            message=clarification_message,
        )

    @classmethod
    def create_not_supported(
        cls,
        agent_name: str,
        reason: str,
        task_type: Optional[str] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> "AgentResult":
        err = AgentError.create(
            category=ErrorCategory.UNSUPPORTED_TASK,
            user_message=reason,
            technical_details=diagnostics or {},
            agent_name=agent_name,
        )
        return cls(
            status=AgentStatus.NOT_SUPPORTED,
            task_type=task_type,
            agent_name=agent_name,
            result={"error": reason},
            errors=[err],
            confidence=0.0,
            diagnostics=diagnostics or {},
            message=reason,
        )
