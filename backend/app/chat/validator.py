"""Validation for planned chat actions and their resulting evidence."""
from __future__ import annotations
from typing import Any
import pandas as pd

class ResultValidator:
    def validate_columns(self, dataframe: pd.DataFrame, *columns: str | None) -> str | None:
        available = ", ".join(map(str, dataframe.columns))
        for column in columns:
            if column and column not in dataframe.columns:
                return f"I can't determine this from the current dataset because there is no '{column}' column. Available columns: {available}."
        return None
    def valid_evidence(self, evidence: dict[str, Any]) -> bool:
        return bool(evidence) and all(value is not None for value in evidence.values() if isinstance(value, (int, float)))
