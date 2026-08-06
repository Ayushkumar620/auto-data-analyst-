from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DatasetSchema:
    id: str
    name: str
    file_type: str
    rows: int
    columns: int
    created_at: str
    status: str = "uploaded"


@dataclass
class DatasetMetadata:
    name: str
    rows: int
    columns: int
    column_names: List[str]
    data_types: Dict[str, str]
    memory_usage: str
    missing_values: int
    duplicate_rows: int
    file_size: str


@dataclass
class DatasetResponse:
    dataset: DatasetSchema
    metadata: DatasetMetadata
    preview: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DatasetProfile:
    dataset_name: str
    rows: int
    columns: int
    column_names: List[str]
    missing_values: int
    duplicates: int
    preview: List[Dict[str, Any]] = field(default_factory=list)
    data_types: Dict[str, str] = field(default_factory=dict)
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    date_columns: List[str] = field(default_factory=list)
    memory_usage: str = ""
