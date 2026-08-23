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
import uuid


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
            "confidence": self.confidence,
            "claim_type": self.claim_type.value,
            "raw_value": self.raw_value,
            "metadata": self.metadata,
        }


@dataclass
class SemanticMapping:
    """Maps a physical column to a semantic concept with confidence."""
    column_name: str
    semantic_concept: str
    concept_category: str
    confidence: float
    evidence: List[Evidence]
    aliases: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_name": self.column_name,
            "semantic_concept": self.semantic_concept,
            "concept_category": self.concept_category,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "aliases": self.aliases,
            "description": self.description,
        }
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
            "category": self.category.value,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
            "retry_after_ms": self.retry_after_ms,
            "suggested_fix": self.suggested_fix,
            "fallback_agent": self.fallback_agent,
        }


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
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "repair_hint": self.repair_hint,
        }


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
            "issues": [i.to_dict() for i in self.issues],
            "repaired": self.repaired,
            "repair_actions": self.repair_actions,
        }

    def add_issue(self, severity: ValidationSeverity, code: str, message: str,
                  field: Optional[str] = None, expected: Any = None, actual: Any = None,
                  repair_hint: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue(severity, code, message, field, expected, actual, repair_hint))
        if severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.passed = False
@dataclass
class AgentResult:
    """
    Standardized output from every agent.

    All agents must return this structure (or a subclass) from their run() method.
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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for API responses / storage."""
        return {
            "agent": self.agent,
            "role": self.role,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_ms": self.duration_ms,
            "output": self.output,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": self.confidence,
            "validation": self.validation.to_dict() if self.validation else None,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": self.warnings,
            "metadata": self.metadata,
            "retry_count": self.retry_count,
        }

    @classmethod
    def success(cls, agent: str, role: str, agent_id: str, started_at: datetime,
                output: Dict[str, Any], evidence: List[Evidence] = None,
                confidence: float = 1.0, duration_ms: float = 0.0,
                warnings: List[str] = None, metadata: Dict[str, Any] = None) -> "AgentResult":
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
    def failure(cls, agent: str, role: str, agent_id: str, started_at: datetime,
                errors: List[AgentError], duration_ms: float = 0.0,
                output: Dict[str, Any] = None, warnings: List[str] = None,
                metadata: Dict[str, Any] = None) -> "AgentResult":
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
    def retrying(cls, agent: str, role: str, agent_id: str, started_at: datetime,
                 retry_count: int, output: Dict[str, Any] = None,
                 errors: List[AgentError] = None) -> "AgentResult":
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
    entities: List[Dict[str, Any]]
    metrics: List[SemanticMapping]
    dimensions: List[SemanticMapping]
    temporal_columns: List[SemanticMapping]
    identifiers: List[SemanticMapping]
    semantic_mappings: List[SemanticMapping]
    relationships: List[Dict[str, Any]]
    data_quality: Dict[str, Any]
    overall_confidence: float
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)