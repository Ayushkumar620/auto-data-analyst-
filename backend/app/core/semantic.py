"""Semantic Schema Agent.

Classifies every column into a semantic role based on:
  - data type
  - sample values
  - cardinality / uniqueness
  - statistical characteristics
  - relationships (optional, passed in)
  - optional LLM reasoning (skipped when no API key is configured)

Never forces a classification when confidence is low.  All hints live in the
configurable semantic dictionary (SEMANTIC_HINTS) - the statistical detection
never depends on any particular dataset or column name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Configurable semantic dictionary.  Hints only: the classifier never
# *requires* a token to classify a column.
# ---------------------------------------------------------------------------
SEMANTIC_HINTS: dict[str, tuple[str, ...]] = {
    "identifier": (
        "id", "_id", "key", "code", "sku", "upc", "ean", "isbn", "uuid",
        "guid", "serial", "number", "no", "ref", "reference", "account",
    ),
    "entity": (
        "name", "company", "organization", "organisation", "vendor", "supplier",
        "customer", "client", "product", "brand", "store", "branch", "partner",
        "employee", "person", "owner", "category_name", "country", "city",
        "region", "state", "division", "department", "channel",
    ),
    "derived_metric": (
        "rate", "ratio", "percentage", "percent", "pct", "margin", "growth",
        "delta", "difference", "change", "share", "index", "yield", "efficiency",
        "density", "average_per", "per_unit",
    ),
    "metric_synonyms": (
        "amount", "total", "sum", "value", "price", "cost", "revenue", "income",
        "sales", "profit", "loss", "count", "quantity", "qty", "units", "volume",
        "balance", "fee", "charge", "payment", "score", "rating", "weight",
        "duration", "hours", "distance", "temp", "temperature", "salary", "wage",
        "gdp", "rate_value",
    ),
    "date_words": ("date", "day", "month", "year", "quarter", "yr", "dt"),
    "time_words": ("time", "hour", "minute", "second", "timestamp", "ts"),
    "dimension_words": ("type", "status", "segment", "group", "class", "tier",
                        "level", "source", "category", "method", "platform"),
    "text_words": ("text", "title", "description", "comment", "note", "message",
                   "content", "body", "summary", "address", "review"),
}


def _normalize(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _token_hits(name: str, tokens: tuple[str, ...]) -> list[str]:
    normalized = _normalize(name)
    hits: list[str] = []
    for token in tokens:
        token_norm = token.strip().lower()
        if token_norm == normalized:
            hits.append(token)
        elif f"_{token_norm}" in f"_{normalized}" or normalized.endswith(f"_{token_norm}"):
            hits.append(token)
    return hits


@dataclass
class ColumnSemantics:
    column: str
    role: str
    confidence: float
    candidates: list[dict[str, float]] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "role": self.role,
            "confidence": round(self.confidence, 3),
            "candidates": self.candidates,
            "rationale": self.rationale,
        }


def _is_integer_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(series)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


class SemanticSchemaAgent:
    """Deterministic semantic role classification with confidence."""

    ROLES = (
        "metric", "dimension", "identifier", "date", "time", "datetime",
        "text", "category", "target_candidate", "derived_metric", "entity",
        "constant", "unknown",
    )

    def __init__(self, hints: dict[str, tuple[str, ...]] | None = None) -> None:
        self.hints = hints or SEMANTIC_HINTS

    # -- public API ---------------------------------------------------------
    def classify(self, dataframe: pd.DataFrame,
                 profiles: dict[str, Any] | None = None,
                 relationships: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        per_column = profiles or self._fallback_profiles(dataframe)
        relationship_map = self._relationship_map(relationships or [])
        return [self._classify_column(dataframe[column], column,
                                      per_column.get(column, {}),
                                      relationship_map.get(column, []))
                for column in dataframe.columns]

    def roles_map(self, dataframe: pd.DataFrame,
                  classifications: list[dict[str, Any]] | None = None) -> dict[str, str]:
        items = classifications or self.classify(dataframe)
        return {item["column"]: item["role"] for item in items}

    def classify_summary(self, dataframe: pd.DataFrame) -> dict[str, list[str]]:
        summary: dict[str, list[str]] = {role: [] for role in self.ROLES}
        for item in self.classify(dataframe):
            summary.setdefault(item["role"], []).append(item["column"])
        return summary

    # -- internals ----------------------------------------------------------
    def _fallback_profiles(self, dataframe: pd.DataFrame) -> dict[str, dict[str, Any]]:
        profiles: dict[str, dict[str, Any]] = {}
        for column in dataframe.columns:
            series = dataframe[column].dropna()
            profiles[str(column)] = {
                "unique_values": int(series.nunique(dropna=True)),
                "missing_values": int(dataframe[column].isna().sum()),
                "total_rows": int(len(dataframe)),
            }
        return profiles

    def _relationship_map(self, relationships: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        mapping: dict[str, list[dict[str, Any]]] = {}
        for relationship in relationships:
            for column in relationship.get("columns", []):
                mapping.setdefault(str(column), []).append(relationship)
        return mapping
