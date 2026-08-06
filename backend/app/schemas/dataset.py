from dataclasses import dataclass, field
from typing import Any, List, Dict


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
