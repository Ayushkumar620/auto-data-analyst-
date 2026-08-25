"""
Core Schemas - Standardized Pydantic data contracts for the reliability architecture.

Defines:
- AgentResult: Standardized Pydantic output from every agent
- AgentError: Standardized Pydantic error with recovery hints
- Evidence: Traceable Pydantic proof for claims and calculations
- DatasetKnowledge: Semantic understanding of a dataset
- SemanticMapping: Column-to-concept mapping with confidence
- ValidationResult: Validation outcomes with repair suggestions
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class AgentStatus(str, Enum):
    """Standardized agent execution status."""
    IDLE = "idle"
    WORKING = "working"
    SUCCESS = "success"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    ERROR = "error"
    RETRYING = "retrying"
    VALIDATION_FAILED = "validation_failed"


class ClaimType(str, Enum):
    """Explicit distinction between types of claims - never conflate."""
    FACT = "fact"
    OBSERVATION = "observation"
    CORRELATION = "correlation"
    INFERENCE = "inference"
    RECOMMENDATION = "recommendation"


class ValidationSeverity(str, Enum):
    """Severity of validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Category of agent error for recovery routing."""
    INPUT_VALIDATION = "input_validation"
    DATA_QUALITY = "data_quality"
    COMPUTATION = "computation"
    RESOURCE = "resource"
    SEMANTIC = "semantic"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


class Evidence(BaseModel):
    """
    Traceable proof supporting a claim or result.
    Grounds agent outputs with deterministic provenance.
    """
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    operation: str = ""
    calculation: Optional[str] = None
    source_reference: Optional[str] = None
    result: Any = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    
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
        if v < 0.0 or v > 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return float(v)

    @model_validator(mode="before")
    @classmethod
    def sync_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Sync source / source_reference
            if "source" in data and not data.get("source_reference"):
                data["source_reference"] = data["source"]
            elif "source_reference" in data and not data.get("source"):
                data["source"] = data["source_reference"]

            # Sync method / operation / calculation
            if "method" in data and not data.get("operation"):
                data["operation"] = data["method"]
            elif "operation" in data and not data.get("method"):
                data["method"] = data["operation"]
            if "calculation" in data and not data.get("operation"):
                data["operation"] = str(data["calculation"])

            # Sync raw_value / result
            if "raw_value" in data and data.get("result") is None:
                data["result"] = data["raw_value"]
            elif "result" in data and data.get("raw_value") is None:
                data["raw_value"] = data["result"]

            # Sync dataset_name / dataset_id
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        return cls.model_validate(data)


class SemanticMapping(BaseModel):
    """Maps a physical column to a semantic concept with confidence."""
    column_name: str
    semantic_concept: str
    concept_category: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: List[Evidence] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    description: str = ""

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return float(v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_name": self.column_name,
            "semantic_concept": self.semantic_concept,
            "concept_category": self.concept_category,
            "confidence": round(float(self.confidence), 4),
            "evidence": [e.to_dict() if isinstance(e, Evidence) else e for e in self.evidence],
            "aliases": self.aliases,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticMapping":
        return cls.model_validate(data)


class AgentError(BaseModel):
    """Standardized error structure with recovery hints and diagnostic details."""
    code: str = "ERROR"
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    recoverable: bool = True
    agent_name: Optional[str] = None
    category: Optional[Union[ErrorCategory, str]] = None
    retry_after_ms: Optional[int] = None
    suggested_fix: Optional[str] = None
    fallback_agent: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def sync_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Sync category / code
            if "category" in data and ("code" not in data or data["code"] == "ERROR"):
                cat = data["category"]
                data["code"] = cat.value if isinstance(cat, ErrorCategory) else str(cat).upper()
            elif "code" in data and not data.get("category"):
                data["category"] = data["code"].lower()
        return data

    def to_dict(self) -> Dict[str, Any]:
        cat_str = self.category.value if isinstance(self.category, ErrorCategory) else str(self.category or self.code)
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
            "agent_name": self.agent_name,
            "category": cat_str,
            "retry_after_ms": self.retry_after_ms,
            "suggested_fix": self.suggested_fix,
            "fallback_agent": self.fallback_agent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentError":
        return cls.model_validate(data)


class ValidationIssue(BaseModel):
    """A single validation issue found during result validation."""
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationIssue":
        return cls.model_validate(data)


class ValidationResult(BaseModel):
    """Validation outcome with repair suggestions."""
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        return cls.model_validate(data)

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


class AgentResult(BaseModel):
    """
    Standardized Pydantic execution output from every agent.
    
    Guarantees deterministic data contracts:
    - status: 'success' | 'partial' | 'failed' (plus status enums)
    - agent_name / task_id / data / message
    - errors / warnings / confidence / evidence / metadata
    - execution_time / model_used / timestamp
    """
    status: Union[AgentStatus, str] = Field(default="success")
    agent_name: str = Field(default="Unknown Agent")
    task_id: str = Field(default="")
    data: Dict[str, Any] = Field(default_factory=dict)
    message: str = Field(default="")
    errors: List[AgentError] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: List[Evidence] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float = Field(default=0.0)
    model_used: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    # Legacy & supplementary fields for seamless backward compatibility
    agent: Optional[str] = None
    role: str = "generalist"
    agent_id: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
    duration_ms: float = Field(default=0.0)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    validation: Optional[ValidationResult] = None
    retry_count: int = 0

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return float(v)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Union[AgentStatus, str]) -> Union[AgentStatus, str]:
        valid_statuses = {
            "success", "partial", "failed", "completed", "error", "working", "idle",
            "retrying", "validation_failed"
        }
        val_str = v.value if isinstance(v, AgentStatus) else str(v).lower()
        if val_str not in valid_statuses:
            raise ValueError(f"Invalid status '{v}'. Must be one of {valid_statuses}")
        return v

    @model_validator(mode="before")
    @classmethod
    def sync_legacy_and_modern_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 1. Sync agent / agent_name
            if "agent" in data and not data.get("agent_name"):
                data["agent_name"] = str(data["agent"])
            elif "agent_name" in data and not data.get("agent"):
                data["agent"] = str(data["agent_name"])

            # 2. Sync agent_id / task_id
            if "agent_id" in data and not data.get("task_id"):
                data["task_id"] = str(data["agent_id"])
            elif "task_id" in data and not data.get("agent_id"):
                data["agent_id"] = str(data["task_id"])

            # 3. Sync output / data
            if "output" in data and not data.get("data"):
                data["data"] = data["output"] if isinstance(data["output"], dict) else {}
            elif "data" in data and not data.get("output"):
                data["output"] = data["data"] if isinstance(data["data"], dict) else {}

            # 4. Sync duration_ms / execution_time
            if "duration_ms" in data and not data.get("execution_time"):
                data["execution_time"] = float(data["duration_ms"])
            elif "execution_time" in data and not data.get("duration_ms"):
                data["duration_ms"] = float(data["execution_time"])

            # 5. Sync started_at / timestamp
            if "started_at" in data and not data.get("timestamp"):
                data["timestamp"] = data["started_at"]
            elif "timestamp" in data and not data.get("started_at"):
                data["started_at"] = data["timestamp"]

            # 6. Normalize status
            if "status" in data:
                st = data["status"]
                st_str = st.value if isinstance(st, AgentStatus) else str(st).lower()
                if st_str == "completed":
                    data["status"] = AgentStatus.COMPLETED
                elif st_str == "error":
                    data["status"] = AgentStatus.ERROR
        return data

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Ensure mirror synchronization on instance
        if not self.agent:
            self.agent = self.agent_name
        if not self.agent_id:
            self.agent_id = self.task_id
        if self.output is None:
            self.output = self.data
        if self.duration_ms is None:
            self.duration_ms = self.execution_time
        if self.started_at is None:
            self.started_at = self.timestamp

    @property
    def is_success(self) -> bool:
        st = self.status.value if isinstance(self.status, AgentStatus) else str(self.status).lower()
        return st in ("success", "completed")

    @property
    def is_partial(self) -> bool:
        st = self.status.value if isinstance(self.status, AgentStatus) else str(self.status).lower()
        return st == "partial"

    @property
    def is_error(self) -> bool:
        st = self.status.value if isinstance(self.status, AgentStatus) else str(self.status).lower()
        return st in ("failed", "error", "validation_failed")

    @property
    def is_retrying(self) -> bool:
        st = self.status.value if isinstance(self.status, AgentStatus) else str(self.status).lower()
        return st == "retrying"

    @property
    def has_evidence(self) -> bool:
        return len(self.evidence) > 0

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)

    def add_error(self, error: AgentError) -> None:
        self.errors.append(error)
        self.status = AgentStatus.ERROR

    def add_evidence(self, evidence: Evidence) -> None:
        self.evidence.append(evidence)

    def to_dict(self) -> Dict[str, Any]:
        st_val = self.status.value if isinstance(self.status, AgentStatus) else str(self.status)
        return {
            "status": st_val,
            "agent_name": self.agent_name or self.agent,
            "task_id": self.task_id or self.agent_id,
            "data": self.data or self.output or {},
            "message": self.message,
            "errors": [e.to_dict() if isinstance(e, AgentError) else e for e in self.errors],
            "warnings": self.warnings,
            "confidence": round(float(self.confidence), 4),
            "evidence": [e.to_dict() if isinstance(e, Evidence) else e for e in self.evidence],
            "metadata": self.metadata,
            "execution_time": round(float(self.execution_time or self.duration_ms or 0.0), 2),
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            # Legacy fields
            "agent": self.agent or self.agent_name,
            "role": self.role,
            "agent_id": self.agent_id or self.task_id,
            "output": self.output or self.data or {},
            "duration_ms": round(float(self.duration_ms or self.execution_time or 0.0), 2),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentResult":
        return cls.model_validate(data)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style get() for seamless backward compatibility."""
        if hasattr(self, key):
            val = getattr(self, key)
            if key == "status":
                return val.value if isinstance(val, AgentStatus) else str(val)
            if key in ("output", "data"):
                return val if val is not None else {}
            return val
        return default

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is not None or hasattr(self, key):
            return val
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

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
    ) -> "AgentResult":
        """Factory for successful result."""
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
            data=dt,
            output=dt,
            message=message,
            started_at=ts,
            timestamp=ts,
            finished_at=datetime.now(),
            duration_ms=exec_t,
            execution_time=exec_t,
            evidence=evidence or [],
            confidence=confidence,
            warnings=warnings or [],
            metadata=metadata or {},
            model_used=model_used,
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
    ) -> "AgentResult":
        """Factory for partial success result."""
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
            data=dt,
            output=dt,
            message=message,
            started_at=ts,
            timestamp=ts,
            finished_at=datetime.now(),
            duration_ms=exec_t,
            execution_time=exec_t,
            evidence=evidence or [],
            confidence=confidence,
            errors=errors or [],
            warnings=warnings or [],
            metadata=metadata or {},
            model_used=model_used,
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
    ) -> "AgentResult":
        """Factory for failed result."""
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
            data=dt,
            output=dt,
            message=message,
            started_at=ts,
            timestamp=ts,
            finished_at=datetime.now(),
            duration_ms=exec_t,
            execution_time=exec_t,
            evidence=[],
            confidence=0.0,
            errors=errors or [],
            warnings=warnings or [],
            metadata=metadata or {},
            model_used=model_used,
        )

    @classmethod
    def retrying(
        cls,
        agent: str,
        role: str,
        agent_id: str,
        started_at: datetime,
        retry_count: int,
        output: Optional[Dict[str, Any]] = None,
        errors: Optional[List[AgentError]] = None,
    ) -> "AgentResult":
        """Factory for retry state."""
        return cls(
            agent=agent,
            agent_name=agent,
            role=role,
            agent_id=agent_id,
            task_id=agent_id,
            status=AgentStatus.RETRYING,
            started_at=started_at,
            timestamp=started_at,
            duration_ms=0.0,
            execution_time=0.0,
            output=output or {},
            data=output or {},
            evidence=[],
            confidence=0.5,
            errors=errors or [],
            retry_count=retry_count,
        )


class DatasetKnowledge(BaseModel):
    """
    Complete semantic understanding of a dataset.
    This object is created once and shared with all downstream agents.
    """
    dataset_id: str
    dataset_type: str
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: List[SemanticMapping] = Field(default_factory=list)
    dimensions: List[SemanticMapping] = Field(default_factory=list)
    temporal_columns: List[SemanticMapping] = Field(default_factory=list)
    identifiers: List[SemanticMapping] = Field(default_factory=list)
    semantic_mappings: List[SemanticMapping] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_type": self.dataset_type,
            "entities": self.entities,
            "metrics": [m.to_dict() if isinstance(m, SemanticMapping) else m for m in self.metrics],
            "dimensions": [d.to_dict() if isinstance(d, SemanticMapping) else d for d in self.dimensions],
            "temporal_columns": [t.to_dict() if isinstance(t, SemanticMapping) else t for t in self.temporal_columns],
            "identifiers": [i.to_dict() if isinstance(i, SemanticMapping) else i for i in self.identifiers],
            "semantic_mappings": [s.to_dict() if isinstance(s, SemanticMapping) else s for s in self.semantic_mappings],
            "relationships": self.relationships,
            "data_quality": self.data_quality,
            "overall_confidence": round(float(self.overall_confidence), 4),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
        }