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

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configurable semantic dictionary.
# These are *hints* only: the classifier must never *require* a token to
# classify a column.  They may be extended per deployment but the generic
# engine works without them.
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
