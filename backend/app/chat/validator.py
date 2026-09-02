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

    def validate_analytical_execution(self, query: str, response: Any) -> tuple[bool, str]:
        """
        Validate whether the analytical query was actually executed,
        distinguishing SUCCESS from INCOMPLETE (where only a preview was returned).
        """
        from agent.filter_engine import FilterEngine
        has_filter = FilterEngine.has_filter_intent(query)
        if not has_filter:
            return True, "SUCCESS"

        if isinstance(response, dict):
            if response.get("type") == "head" or response.get("intent") == "head":
                return False, "INCOMPLETE: only dataset preview returned for filtered analytical query"
            if "matching_rows" in response and "total_rows" in response:
                if response.get("matching_rows") == response.get("total_rows") and "No filter applied" in str(response.get("filter", "")):
                    return False, "INCOMPLETE: filter condition was not applied (returned all records)"
            if "matching_rows" in response or "filter" in response or response.get("type") == "filter_result":
                return True, "SUCCESS"
        if hasattr(response, "intent") and response.intent == "head":
            return False, "INCOMPLETE: only dataset preview returned for filtered analytical query"
        if hasattr(response, "evidence") and isinstance(response.evidence, dict):
            ev = response.evidence
            if ev.get("matching_rows") == ev.get("total_rows") and "No filter applied" in str(ev.get("filter", "")):
                return False, "INCOMPLETE: filter condition was not applied (returned all records)"
            if "matching_rows" in ev or "filter" in ev:
                return True, "SUCCESS"
        return True, "SUCCESS"
