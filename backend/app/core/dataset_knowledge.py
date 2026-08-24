"""Dataset Knowledge Engine.

Centralized semantic understanding of a dataset. Aggregates:
- dataset metadata (shape, dtypes, memory, file info)
- column classifications (metrics, dimensions, dates, identifiers, categories)
- semantic mappings (business concepts, aliases, confidence, evidence)
- relationships (mathematical formulas, correlations, functional dependencies)
- data quality assessment (quality score, issue list, recommendations)
- temporal properties (frequency, date span, ordering)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.schemas import ClaimType, Evidence, SemanticMapping


@dataclass
class DatasetKnowledge:
    """
    Complete semantic and analytical profile of a dataset.
    Constructed once by SemanticSchemaAgent and shared with all downstream agents.
    """
    dataset_id: str
    dataset_type: str = "tabular"  # tabular, time_series, transactional, financial, customer, general
    columns: List[str] = field(default_factory=list)
    data_types: Dict[str, str] = field(default_factory=dict)
    semantic_meanings: Dict[str, str] = field(default_factory=dict)
    metrics: List[SemanticMapping] = field(default_factory=list)
    dimensions: List[SemanticMapping] = field(default_factory=list)
    date_columns: List[SemanticMapping] = field(default_factory=list)
    identifiers: List[SemanticMapping] = field(default_factory=list)
    semantic_mappings: List[SemanticMapping] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numeric_columns: List[str] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    missing_values: Dict[str, Any] = field(default_factory=dict)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    overall_confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Query / Lookup Helpers
    # ------------------------------------------------------------------
    def get_column_mapping(self, column_name: str) -> Optional[SemanticMapping]:
        """Find the SemanticMapping for a specific column name."""
        col_norm = column_name.strip().lower()
        for mapping in self.semantic_mappings:
            if mapping.column_name.strip().lower() == col_norm:
                return mapping
        return None

    def find_columns_by_concept(self, concept: str, min_confidence: float = 0.5) -> List[str]:
        """Find all column names associated with a semantic concept (or its aliases)."""
        concept_norm = concept.strip().lower()
        matches: List[str] = []
        for mapping in self.semantic_mappings:
            if mapping.confidence < min_confidence:
                continue
            if mapping.semantic_concept.strip().lower() == concept_norm:
                matches.append(mapping.column_name)
            elif any(alias.strip().lower() == concept_norm for alias in mapping.aliases):
                matches.append(mapping.column_name)
        return matches

    def get_primary_date_column(self) -> Optional[str]:
        """Return the primary date or timestamp column with highest confidence."""
        if not self.date_columns:
            return None
        sorted_dates = sorted(self.date_columns, key=lambda m: m.confidence, reverse=True)
        return sorted_dates[0].column_name

    def get_primary_metric(self) -> Optional[str]:
        """Return the primary numeric target metric with highest confidence."""
        if not self.metrics:
            return None
        sorted_metrics = sorted(self.metrics, key=lambda m: m.confidence, reverse=True)
        return sorted_metrics[0].column_name

    def get_primary_dimension(self) -> Optional[str]:
        """Return the primary grouping dimension with highest confidence."""
        if not self.dimensions:
            return None
        sorted_dims = sorted(self.dimensions, key=lambda m: m.confidence, reverse=True)
        return sorted_dims[0].column_name

    def is_time_series(self) -> bool:
        """Check if dataset has strong temporal structure."""
        return self.dataset_type == "time_series" or len(self.date_columns) > 0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize DatasetKnowledge to dictionary for API responses & storage."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_type": self.dataset_type,
            "columns": self.columns,
            "data_types": self.data_types,
            "semantic_meanings": self.semantic_meanings,
            "metrics": [m.to_dict() if isinstance(m, SemanticMapping) else m for m in self.metrics],
            "dimensions": [d.to_dict() if isinstance(d, SemanticMapping) else d for d in self.dimensions],
            "date_columns": [t.to_dict() if isinstance(t, SemanticMapping) else t for t in self.date_columns],
            "identifiers": [i.to_dict() if isinstance(i, SemanticMapping) else i for i in self.identifiers],
            "semantic_mappings": [s.to_dict() if isinstance(s, SemanticMapping) else s for s in self.semantic_mappings],
            "categorical_columns": self.categorical_columns,
            "numeric_columns": self.numeric_columns,
            "relationships": self.relationships,
            "missing_values": self.missing_values,
            "data_quality": self.data_quality,
            "confidence_scores": self.confidence_scores,
            "entities": self.entities,
            "overall_confidence": round(float(self.overall_confidence), 4),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize DatasetKnowledge to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetKnowledge":
        """Deserialize DatasetKnowledge from dictionary."""
        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"])
            except (ValueError, TypeError):
                created_at = datetime.now()
        else:
            created_at = datetime.now()

        def parse_mappings(raw_list: List[Any]) -> List[SemanticMapping]:
            return [
                SemanticMapping.from_dict(m) if isinstance(m, dict) else m
                for m in (raw_list or [])
            ]

        return cls(
            dataset_id=str(data.get("dataset_id", "")),
            dataset_type=str(data.get("dataset_type", "tabular")),
            columns=list(data.get("columns", [])),
            data_types=dict(data.get("data_types", {})),
            semantic_meanings=dict(data.get("semantic_meanings", {})),
            metrics=parse_mappings(data.get("metrics", [])),
            dimensions=parse_mappings(data.get("dimensions", [])),
            date_columns=parse_mappings(data.get("date_columns", [])),
            identifiers=parse_mappings(data.get("identifiers", [])),
            semantic_mappings=parse_mappings(data.get("semantic_mappings", [])),
            categorical_columns=list(data.get("categorical_columns", [])),
            numeric_columns=list(data.get("numeric_columns", [])),
            relationships=list(data.get("relationships", [])),
            missing_values=dict(data.get("missing_values", {})),
            data_quality=dict(data.get("data_quality", {})),
            confidence_scores=dict(data.get("confidence_scores", {})),
            entities=list(data.get("entities", [])),
            overall_confidence=float(data.get("overall_confidence", 1.0)),
            created_at=created_at,
            metadata=dict(data.get("metadata", {})),
        )

