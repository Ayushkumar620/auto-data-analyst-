"""
Core Schemas - Standardized data contracts for the reliability architecture.

Defines:
- AgentResult: Standardized output from every agent
- DatasetKnowledge: Semantic understanding of a dataset
- SemanticMapping: Column-to-concept mapping with confidence
- Evidence: Traceable proof for claims
- ValidationResult: Validation outcomes with repair suggestions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid


class AgentStatus(str, Enum):
    """Standardized agent execution status."""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    ERROR = "error"
    RETRYING = "retrying"
@dataclass
class Evidence:
    """Traceable evidence supporting a claim or result."""
    source: str                      # Agent/tool that produced this
    method: str                      # How it was computed
    data_ref: Dict[str, Any]         # Reference to source data (column, rows, query)
    confidence: float                # 0.0 - 1.0
    raw_value: Any = None            # The actual computed value
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "method": self.method,
            "data_ref": self.data_ref,
            "confidence": self.confidence,
            "raw_value": self.raw_value,
            "metadata": self.metadata,
        }


@dataclass
class SemanticMapping:
    """Maps a physical column to a semantic concept with confidence."""
    column_name: str                 # Actual column name in dataset
    semantic_concept: str            # Business concept (e.g., "revenue", "customer_id", "transaction_date")
    concept_category: str            # "metric", "dimension", "identifier", "temporal", "entity"
    confidence: float                # 0.0 - 1.0 based on evidence
    evidence: List[Evidence]         # Why this mapping was made
    aliases: List[str] = field(default_factory=list)  # Other names this could be
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
    VALIDATION_FAILED = "validation_failed"


class ClaimType(str, Enum):
    """Explicit distinction between types of claims - never conflate."""
    FACT = "fact"                    # Directly computable from data (e.g., "sum of sales = 100")
    OBSERVATION = "observation"      # Pattern seen in data (e.g., "sales peak in December")
    CORRELATION = "correlation"      # Statistical relationship (e.g., "sales correlates with marketing spend")
    INFERENCE = "inference"          # Reasoned conclusion (e.g., "marketing drives sales")
    RECOMMENDATION = "recommendation"  # Suggested action (e.g., "increase marketing in Q4")


class ValidationSeverity(str, Enum):
    """Severity of validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
@dataclass
class DatasetKnowledge:
    """
    Complete semantic understanding of a dataset.
    This object is created once and shared with all downstream agents.
    """
    dataset_id: str
    dataset_type: str                # "transactional", "time_series", "cross_sectional", "panel", "unknown"
    entities: List[Dict[str, Any]]   # Business entities identified (customers, products, etc.)
    metrics: List[SemanticMapping]   # Columns that are measurable metrics (revenue, quantity, etc.)
    dimensions: List[SemanticMapping] # Columns for grouping/filtering (region, category, etc.)
    temporal_columns: List[SemanticMapping]  # Date/time columns
    identifiers: List[SemanticMapping]       # Primary/foreign keys
    semantic_mappings: List[SemanticMapping] # All column mappings
    relationships: List[Dict[str, Any]]      # Detected relationships (FK, hierarchy, etc.)
    data_quality: Dict[str, Any]    # Quality metrics per column
    overall_confidence: float        # Aggregate confidence in understanding
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)