"""
Core Schemas - Standardized Pydantic data contracts for the reliability architecture.

Defines:
- AgentResult: Standardized Pydantic output from every agent (canonical re-export)
- AgentError: Standardized Pydantic error with recovery hints (canonical re-export)
- Evidence: Traceable Pydantic proof for claims and calculations (canonical re-export)
- SemanticMapping: Column-to-concept mapping with confidence
- DatasetKnowledge: Semantic understanding of a dataset
- ValidationResult: Validation outcomes with repair suggestions (canonical re-export)
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator

# Re-export canonical reliability contracts
from agent.agent_result import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)


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
            return max(0.0, min(1.0, float(v)))
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
