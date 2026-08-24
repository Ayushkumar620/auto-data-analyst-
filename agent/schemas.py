"""
Core Schemas - Standardized data contracts for the reliability architecture.

Defines:
- AgentResult: Standardized output from every agent
- AgentError: Standardized error with recovery hints
- DatasetKnowledge: Semantic understanding of a dataset
- SemanticMapping: Column-to-concept mapping with confidence
- Evidence: Traceable proof for claims
- ValidationResult: Validation outcomes with repair suggestions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentStatus(str, Enum):
    """Standardized agent execution status."""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
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


@dataclass
class Evidence:
    """Traceable evidence supporting a claim or result."""
    source: str
    method: str
    data_ref: Dict[str, Any]
    confidence: float
    claim_type: ClaimType = ClaimType.OBSERVATION
    raw_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "method": self.method,
            "data_ref": self.data_ref,
            "confidence": round(float(self.confidence), 4) if isinstance(self.confidence, (int, float)) else self.confidence,
            "claim_type": self.claim_type.value if isinstance(self.claim_type, ClaimType) else str(self.claim_type),
            "raw_value": self.raw_value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        claim_type = data.get("claim_type", ClaimType.OBSERVATION.value)
        if isinstance(claim_type, str):
            try:
                claim_type = ClaimType(claim_type)
            except ValueError:
                claim_type = ClaimType.OBSERVATION

        return cls(
            source=str(data.get("source", "")),
            method=str(data.get("method", "")),
            data_ref=data.get("data_ref", {}) if isinstance(data.get("data_ref"), dict) else {},
            confidence=float(data.get("confidence", 1.0)),
            claim_type=claim_type,
            raw_value=data.get("raw_value"),
            metadata=data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {},
        )


@dataclass
class SemanticMapping:
    """Maps a physical column to a semantic concept with confidence."""
    column_name: str
    semantic_concept: str
    concept_category: str
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_name": self.column_name,
            "semantic_concept": self.semantic_concept,
            "concept_category": self.concept_category,
            "confidence": round(float(self.confidence), 4) if isinstance(self.confidence, (int, float)) else self.confidence,
            "evidence": [e.to_dict() if isinstance(e, Evidence) else e for e in self.evidence],
            "aliases": self.aliases,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticMapping":
        raw_evidence = data.get("evidence", [])
        evidence_list = [
            Evidence.from_dict(e) if isinstance(e, dict) else e
            for e in raw_evidence
        ]
        return cls(
            column_name=str(data.get("column_name", "")),
            semantic_concept=str(data.get("semantic_concept", "")),
            concept_category=str(data.get("concept_category", "")),
            confidence=float(data.get("confidence", 1.0)),
            evidence=evidence_list,
            aliases=list(data.get("aliases", [])),
            description=str(data.get("description", "")),
        )


@dataclass
class AgentError:
    """Standardized error with recovery hints."""
    category: ErrorCategory
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True
    retry_after_ms: Optional[int] = None
    suggested_fix: Optional[str] = None
    fallback_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value if isinstance(self.category, ErrorCategory) else str(self.category),
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
            "retry_after_ms": self.retry_after_ms,
            "suggested_fix": self.suggested_fix,
            "fallback_agent": self.fallback_agent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentError":
        cat = data.get("category", ErrorCategory.UNKNOWN.value)
        if isinstance(cat, str):
            try:
                cat = ErrorCategory(cat)
            except ValueError:
                cat = ErrorCategory.UNKNOWN
        return cls(
            category=cat,
            message=str(data.get("message", "")),
            details=data.get("details", {}) if isinstance(data.get("details"), dict) else {},
            recoverable=bool(data.get("recoverable", True)),
            retry_after_ms=data.get("retry_after_ms"),
            suggested_fix=data.get("suggested_fix"),
            fallback_agent=data.get("fallback_agent"),
        )


@dataclass
class ValidationIssue:
    """A single validation issue found during result validation."""
    severity: ValidationSeverity
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
        sev = data.get("severity", ValidationSeverity.ERROR.value)
        if isinstance(sev, str):
            try:
                sev = ValidationSeverity(sev)
            except ValueError:
                sev = ValidationSeverity.ERROR
        return cls(
            severity=sev,
            code=str(data.get("code", "")),
            message=str(data.get("message", "")),
            field=data.get("field"),
            expected=data.get("expected"),
            actual=data.get("actual"),
            repair_hint=data.get("repair_hint"),
        )


@dataclass
class ValidationResult:
    """Validation outcome with repair suggestions."""
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    repaired: bool = False
    repair_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() if isinstance(i, ValidationIssue) else i for i in self.issues],
            "repaired": self.repaired,
            "repair_actions": self.repair_actions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        raw_issues = data.get("issues", [])
        issues_list = [
            ValidationIssue.from_dict(i) if isinstance(i, dict) else i
            for i in raw_issues
        ]
        return cls(
            passed=bool(data.get("passed", True)),
            issues=issues_list,
            repaired=bool(data.get("repaired", False)),
            repair_actions=list(data.get("repair_actions", [])),
        )

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
        self.issues.append(ValidationIssue(severity, code, message, field, expected, actual, repair_hint))
        if severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.passed = False


@dataclass
class AgentResult:
    """
    Standardized output from every agent.

    All agents must return this structure from their run() method.
    The BaseAgent._finish() and _error() methods construct this automatically.
    """
    agent: str
    role: str
    agent_id: str
    status: AgentStatus
    started_at: datetime
    finished_at: Optional[datetime] = None
    duration_ms: float = 0.0
    output: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Evidence] = field(default_factory=list)
    confidence: float = 1.0
    validation: Optional[ValidationResult] = None
    errors: List[AgentError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0

    @property
    def is_success(self) -> bool:
        """Check if execution completed successfully."""
        return self.status == AgentStatus.COMPLETED

    @property
    def is_error(self) -> bool:
        """Check if execution failed."""
        return self.status in (AgentStatus.ERROR, AgentStatus.VALIDATION_FAILED)

    @property
    def is_retrying(self) -> bool:
        """Check if agent is currently retrying."""
        return self.status == AgentStatus.RETRYING

    @property
    def has_evidence(self) -> bool:
        """Check if any evidence items are attached."""
        return len(self.evidence) > 0

    @property
    def has_errors(self) -> bool:
        """Check if any errors are recorded."""
        return len(self.errors) > 0

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def add_error(self, error: AgentError) -> None:
        """Add an error item."""
        self.errors.append(error)
        self.status = AgentStatus.ERROR

    def add_evidence(self, evidence: Evidence) -> None:
        """Add an evidence item."""
        self.evidence.append(evidence)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for API responses / storage."""
        return {
            "agent": self.agent,
            "role": self.role,
            "agent_id": self.agent_id,
            "status": self.status.value if isinstance(self.status, AgentStatus) else str(self.status),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_ms": round(float(self.duration_ms), 2) if isinstance(self.duration_ms, (int, float)) else self.duration_ms,
            "output": self.output,
            "evidence": [e.to_dict() if isinstance(e, Evidence) else e for e in self.evidence],
            "confidence": round(float(self.confidence), 4) if isinstance(self.confidence, (int, float)) else self.confidence,
            "validation": self.validation.to_dict() if self.validation else None,
            "errors": [e.to_dict() if isinstance(e, AgentError) else e for e in self.errors],
            "warnings": self.warnings,
            "metadata": self.metadata,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentResult":
        """Deserialize from dictionary."""
        status_val = data.get("status", AgentStatus.COMPLETED.value)
        if isinstance(status_val, str):
            try:
                status_val = AgentStatus(status_val)
            except ValueError:
                status_val = AgentStatus.COMPLETED

        started_at = None
        if data.get("started_at"):
            try:
                started_at = datetime.fromisoformat(data["started_at"])
            except (ValueError, TypeError):
                started_at = datetime.now()
        else:
            started_at = datetime.now()

        finished_at = None
        if data.get("finished_at"):
            try:
                finished_at = datetime.fromisoformat(data["finished_at"])
            except (ValueError, TypeError):
                finished_at = None

        evidence_list = [
            Evidence.from_dict(e) if isinstance(e, dict) else e
            for e in data.get("evidence", [])
        ]
        errors_list = [
            AgentError.from_dict(e) if isinstance(e, dict) else e
            for e in data.get("errors", [])
        ]
        val_result = None
        if data.get("validation"):
            val_result = ValidationResult.from_dict(data["validation"]) if isinstance(data["validation"], dict) else data["validation"]

        return cls(
            agent=str(data.get("agent", "Unknown Agent")),
            role=str(data.get("role", "generalist")),
            agent_id=str(data.get("agent_id", "")),
            status=status_val,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=float(data.get("duration_ms", 0.0)),
            output=data.get("output", {}) if isinstance(data.get("output"), dict) else {},
            evidence=evidence_list,
            confidence=float(data.get("confidence", 1.0)),
            validation=val_result,
            errors=errors_list,
            warnings=list(data.get("warnings", [])),
            metadata=data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {},
            retry_count=int(data.get("retry_count", 0)),
        )

    # ------------------------------------------------------------------
    # Backward-compatibility helpers: allow dict-style access so existing
    # callers (planner.py, app.py, ReportAgent) work with AgentResult
    # without modification. New code should prefer attribute access.
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style get(). Supports 'status' -> string, 'output' -> dict."""
        if hasattr(self, key):
            value = getattr(self, key)
            if key == "status":
                return value.value if isinstance(value, AgentStatus) else str(value)
            if key == "output":
                return value if value is not None else {}
            return value
        return default

    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is not None or hasattr(self, key):
            return value
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    @classmethod
    def success(
        cls,
        agent: str,
        role: str,
        agent_id: str,
        started_at: datetime,
        output: Dict[str, Any],
        evidence: Optional[List[Evidence]] = None,
        confidence: float = 1.0,
        duration_ms: float = 0.0,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentResult":
        """Factory for successful result."""
        return cls(
            agent=agent,
            role=role,
            agent_id=agent_id,
            status=AgentStatus.COMPLETED,
            started_at=started_at,
            finished_at=datetime.now(),
            duration_ms=duration_ms,
            output=output,
            evidence=evidence or [],
            confidence=confidence,
            warnings=warnings or [],
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls,
        agent: str,
        role: str,
        agent_id: str,
        started_at: datetime,
        errors: List[AgentError],
        duration_ms: float = 0.0,
        output: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AgentResult":
        """Factory for failed result."""
        return cls(
            agent=agent,
            role=role,
            agent_id=agent_id,
            status=AgentStatus.ERROR,
            started_at=started_at,
            finished_at=datetime.now(),
            duration_ms=duration_ms,
            output=output or {},
            evidence=[],
            confidence=0.0,
            errors=errors,
            warnings=warnings or [],
            metadata=metadata or {},
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
            role=role,
            agent_id=agent_id,
            status=AgentStatus.RETRYING,
            started_at=started_at,
            duration_ms=0.0,
            output=output or {},
            evidence=[],
            confidence=0.5,
            errors=errors or [],
            retry_count=retry_count,
        )


@dataclass
class DatasetKnowledge:
    """
    Complete semantic understanding of a dataset.
    This object is created once and shared with all downstream agents.
    """
    dataset_id: str
    dataset_type: str
    entities: List[Dict[str, Any]] = field(default_factory=list)
    metrics: List[SemanticMapping] = field(default_factory=list)
    dimensions: List[SemanticMapping] = field(default_factory=list)
    temporal_columns: List[SemanticMapping] = field(default_factory=list)
    identifiers: List[SemanticMapping] = field(default_factory=list)
    semantic_mappings: List[SemanticMapping] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    overall_confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

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