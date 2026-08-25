"""
Dataset Knowledge Engine - Central semantic data model for multi-agent analytical workflows.

Defines:
- SemanticType: Controlled semantic categories (METRIC, DIMENSION, IDENTIFIER, DATE, etc.)
- ColumnKnowledge: Deep analytical profile of an individual column
- DataQuality: Comprehensive data quality metrics (missing, duplicates, outliers, quality score)
- DatasetKnowledge: Unified dataset understanding shared across all downstream agents
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator

from agent.schemas import ClaimType, Evidence, SemanticMapping


class SemanticType(str, Enum):
    """Controlled semantic categories for columns."""
    METRIC = "METRIC"
    DIMENSION = "DIMENSION"
    IDENTIFIER = "IDENTIFIER"
    DATE = "DATE"
    DATETIME = "DATETIME"
    CATEGORY = "CATEGORY"
    TEXT = "TEXT"
    BOOLEAN = "BOOLEAN"
    TARGET = "TARGET"
    UNKNOWN = "UNKNOWN"


class DataQuality(BaseModel):
    """Data quality indicators and health score."""
    missing_values: Union[Dict[str, int], int] = Field(default_factory=dict)
    duplicates: int = 0
    duplicate_rows: Optional[int] = None
    outliers: Union[Dict[str, int], int] = Field(default_factory=dict)
    invalid_values: Union[Dict[str, Any], int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    quality_score: float = Field(default=100.0, ge=0.0, le=100.0)

    @model_validator(mode="before")
    @classmethod
    def sync_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "duplicate_rows" in data and "duplicates" not in data:
                data["duplicates"] = int(data["duplicate_rows"])
            elif "duplicates" in data and "duplicate_rows" not in data:
                data["duplicate_rows"] = int(data["duplicates"])
        return data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "missing_values": self.missing_values,
            "duplicates": self.duplicates,
            "duplicate_rows": self.duplicate_rows if self.duplicate_rows is not None else self.duplicates,
            "outliers": self.outliers,
            "invalid_values": self.invalid_values,
            "warnings": self.warnings,
            "quality_score": round(float(self.quality_score), 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataQuality":
        return cls.model_validate(data)


class ColumnKnowledge(BaseModel):
    """Detailed semantic and statistical knowledge for a single column."""
    column_name: str
    data_type: str
    semantic_type: Union[SemanticType, str] = SemanticType.UNKNOWN
    role: str = "unknown"
    unique_count: int = 0
    missing_count: int = 0
    missing_percentage: float = 0.0
    sample_values: List[Any] = Field(default_factory=list)
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean: Optional[float] = None
    cardinality: Optional[int] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    concept: Optional[str] = None
    is_uncertain: bool = False
    evidence: List[Evidence] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return float(v)

    def to_dict(self) -> Dict[str, Any]:
        st = self.semantic_type.value if isinstance(self.semantic_type, SemanticType) else str(self.semantic_type)
        return {
            "column_name": self.column_name,
            "data_type": self.data_type,
            "semantic_type": st,
            "role": self.role,
            "unique_count": self.unique_count,
            "missing_count": self.missing_count,
            "missing_percentage": round(float(self.missing_percentage), 2),
            "sample_values": self.sample_values,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean": round(float(self.mean), 4) if self.mean is not None else None,
            "cardinality": self.cardinality if self.cardinality is not None else self.unique_count,
            "confidence": round(float(self.confidence), 4),
            "concept": self.concept,
            "is_uncertain": self.is_uncertain,
            "evidence": [e.to_dict() if isinstance(e, Evidence) else e for e in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnKnowledge":
        return cls.model_validate(data)


class DatasetKnowledge(BaseModel):
    """
    Unified, central semantic and analytical understanding of a dataset.
    Built once during dataset ingestion/profiling and passed as immutable context to all downstream agents.
    """
    dataset_id: str
    dataset_name: str = "dataset"
    dataset_type: str = "tabular"  # tabular, time_series, transactional, financial, customer, general
    row_count: int = 0
    column_count: int = 0
    columns: List[Union[str, ColumnKnowledge]] = Field(default_factory=list)
    column_knowledge: Dict[str, ColumnKnowledge] = Field(default_factory=dict)
    data_types: Dict[str, str] = Field(default_factory=dict)
    semantic_meanings: Dict[str, str] = Field(default_factory=dict)
    metrics: List[Union[str, SemanticMapping]] = Field(default_factory=list)
    dimensions: List[Union[str, SemanticMapping]] = Field(default_factory=list)
    identifiers: List[Union[str, SemanticMapping]] = Field(default_factory=list)
    date_columns: List[Union[str, SemanticMapping]] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    numerical_columns: List[str] = Field(default_factory=list)
    numeric_columns: List[str] = Field(default_factory=list)
    target_candidates: List[str] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    data_quality: Union[DataQuality, Dict[str, Any]] = Field(default_factory=dict)
    missing_values: Dict[str, Any] = Field(default_factory=dict)
    semantic_mappings: List[SemanticMapping] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence", "overall_confidence")
    @classmethod
    def validate_conf(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return float(v)

    @model_validator(mode="before")
    @classmethod
    def sync_legacy_and_modern_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Sync dataset_id / dataset_name
            if "dataset_name" in data and not data.get("dataset_id"):
                data["dataset_id"] = str(data["dataset_name"])
            elif "dataset_id" in data and not data.get("dataset_name"):
                data["dataset_name"] = str(data["dataset_id"])

            # Sync numeric_columns / numerical_columns
            if "numerical_columns" in data and not data.get("numeric_columns"):
                data["numeric_columns"] = list(data["numerical_columns"])
            elif "numeric_columns" in data and not data.get("numerical_columns"):
                data["numerical_columns"] = list(data["numeric_columns"])

            # Sync confidence / overall_confidence
            if "overall_confidence" in data and "confidence" not in data:
                data["confidence"] = float(data["overall_confidence"])
            elif "confidence" in data and "overall_confidence" not in data:
                data["overall_confidence"] = float(data["confidence"])

            # Sync row_count / column_count
            if "columns" in data and not data.get("column_count"):
                data["column_count"] = len(data["columns"])
        return data

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.numerical_columns and self.numeric_columns:
            self.numerical_columns = list(self.numeric_columns)
        if not self.numeric_columns and self.numerical_columns:
            self.numeric_columns = list(self.numerical_columns)

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

    def get_column_knowledge(self, column_name: str) -> Optional[ColumnKnowledge]:
        """Find the ColumnKnowledge object for a specific column name."""
        col_norm = column_name.strip().lower()
        if col_norm in self.column_knowledge:
            return self.column_knowledge[col_norm]
        for c in self.columns:
            if isinstance(c, ColumnKnowledge) and c.column_name.strip().lower() == col_norm:
                return c
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
        if isinstance(self.date_columns[0], str):
            return self.date_columns[0]
        sorted_dates = sorted(self.date_columns, key=lambda m: getattr(m, "confidence", 1.0), reverse=True)
        return sorted_dates[0].column_name if hasattr(sorted_dates[0], "column_name") else str(sorted_dates[0])

    def get_primary_metric(self) -> Optional[str]:
        """Return the primary numeric target metric with highest confidence."""
        if not self.metrics:
            return None
        if isinstance(self.metrics[0], str):
            return self.metrics[0]
        sorted_metrics = sorted(self.metrics, key=lambda m: getattr(m, "confidence", 1.0), reverse=True)
        return sorted_metrics[0].column_name if hasattr(sorted_metrics[0], "column_name") else str(sorted_metrics[0])

    def get_primary_dimension(self) -> Optional[str]:
        """Return the primary grouping dimension with highest confidence."""
        if not self.dimensions:
            return None
        if isinstance(self.dimensions[0], str):
            return self.dimensions[0]
        sorted_dims = sorted(self.dimensions, key=lambda m: getattr(m, "confidence", 1.0), reverse=True)
        return sorted_dims[0].column_name if hasattr(sorted_dims[0], "column_name") else str(sorted_dims[0])

    def is_time_series(self) -> bool:
        """Check if dataset has strong temporal structure."""
        return self.dataset_type == "time_series" or len(self.date_columns) > 0

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize DatasetKnowledge to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DatasetKnowledge to JSON-compatible dictionary."""
        col_list = [c.to_dict() if isinstance(c, ColumnKnowledge) else c for c in self.columns]
        metrics_list = [m.to_dict() if isinstance(m, SemanticMapping) else m for m in self.metrics]
        dims_list = [d.to_dict() if isinstance(d, SemanticMapping) else d for d in self.dimensions]
        dates_list = [t.to_dict() if isinstance(t, SemanticMapping) else t for t in self.date_columns]
        ids_list = [i.to_dict() if isinstance(i, SemanticMapping) else i for i in self.identifiers]
        sem_list = [s.to_dict() if isinstance(s, SemanticMapping) else s for s in self.semantic_mappings]
        dq_dict = self.data_quality.to_dict() if isinstance(self.data_quality, DataQuality) else self.data_quality

        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "dataset_type": self.dataset_type,
            "row_count": self.row_count,
            "column_count": self.column_count or len(self.columns),
            "columns": col_list,
            "column_knowledge": {k: v.to_dict() for k, v in self.column_knowledge.items()},
            "data_types": self.data_types,
            "semantic_meanings": self.semantic_meanings,
            "metrics": metrics_list,
            "dimensions": dims_list,
            "date_columns": dates_list,
            "identifiers": ids_list,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "numeric_columns": self.numeric_columns,
            "target_candidates": self.target_candidates,
            "relationships": self.relationships,
            "data_quality": dq_dict,
            "missing_values": self.missing_values,
            "semantic_mappings": sem_list,
            "confidence": round(float(self.confidence), 4),
            "overall_confidence": round(float(self.overall_confidence), 4),
            "confidence_scores": self.confidence_scores,
            "entities": self.entities,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetKnowledge":
        return cls.model_validate(data)


# Backward-compatibility alias for backend.app.core.dataset_knowledge
__all__ = ["DatasetKnowledge", "ColumnKnowledge", "DataQuality", "SemanticType"]
