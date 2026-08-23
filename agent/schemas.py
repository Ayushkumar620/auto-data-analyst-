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