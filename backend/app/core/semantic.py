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

    def _classify_column(self, series: pd.Series, column: str,
                         profile: dict[str, Any],
                         relationships: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = _normalize(column)
        candidates: list[dict[str, float]] = []
        rationale: list[str] = []
        total = int(profile.get("total_rows") or len(series))
        non_null = series.dropna()
        unique = int(profile.get("unique_values") or non_null.nunique(dropna=True))
        unique_ratio = (unique / total) if total else 0.0

        # 1) Constant column
        if unique <= 1 and total > 0:
            return ColumnSemantics(column, "constant", 0.98,
                                   [{"role": "constant", "confidence": 0.98}],
                                   ["Single distinct value across the column"]).to_dict()

        # 2) Missing-only column
        if non_null.empty:
            return ColumnSemantics(column, "unknown", 0.4,
                                   [{"role": "constant", "confidence": 0.5}],
                                   ["No non-null values available"]).to_dict()

        # 3) Date / time / datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            has_time_component = bool(((non_null.dt.hour != 0) | (non_null.dt.minute != 0) |
                                       (non_null.dt.second != 0) | (non_null.dt.microsecond != 0)).any())
            role, conf = ("datetime", 0.95) if has_time_component else ("date", 0.95)
            candidates.append({"role": role, "confidence": conf})
            if _token_hits(normalized, self.hints.get("time_words", ())):
                candidates.append({"role": "time", "confidence": 0.4})
            rationale.append(f"Parsed as {role} by the temporal parser")
            return ColumnSemantics(column, role, conf, candidates, rationale).to_dict()

        # 4) Time-only columns (string columns that parse to time of day)
        if not _is_numeric(series):
            parsed_time = pd.to_datetime(non_null, errors="coerce", format="mixed")
            time_only = (parsed_time.notna().mean() >= 0.8 and
                         ((parsed_time.dt.hour != 0) | (parsed_time.dt.minute != 0) |
                          (parsed_time.dt.second != 0)).mean() > 0 and
                         parsed_time.dt.date.nunique() <= 1)
            if time_only:
                return ColumnSemantics(
                    column, "time", 0.85,
                    [{"role": "time", "confidence": 0.85},
                     {"role": "text", "confidence": 0.1}],
                    ["Values parse as time-of-day and carry no date component"]).to_dict()

        # 5) Year / month / quarter integer columns
        temporal_int = self._temporal_integer(series, normalized)
        if temporal_int is not None:
            return ColumnSemantics(
                column, temporal_int["role"], temporal_int["confidence"],
                temporal_int["candidates"], temporal_int["rationale"]).to_dict()

        # 6) Identifier detection (statistical + name hints)
        identifier_result = self._identifier_candidate(series, column, unique, total, unique_ratio)
        if identifier_result is not None:
            role, conf, cands, reasons = identifier_result
            return ColumnSemantics(column, role, conf, cands, reasons).to_dict()

        # 7) Numeric columns -> metrics / derived metrics
        if _is_numeric(series):
            values = pd.to_numeric(non_null, errors="coerce").dropna()
            if values.empty:
                return ColumnSemantics(column, "unknown", 0.3,
                                       [], ["Numeric dtype but no parseable values"]).to_dict()
            std = float(values.std()) if len(values) > 1 else 0.0
            if std > 0:
                candidates.append({"role": "metric", "confidence": 0.7})
                candidates.append({"role": "target_candidate", "confidence": 0.55})
                rationale.append(f"Numeric column with variance (std={std:.4g})")
                derived_hits = _token_hits(normalized, self.hints.get("derived_metric", ()))
                strict_derived = [h for h in derived_hits if _normalize(h) in normalized.split("_")]
                if strict_derived:
                    candidates.append({"role": "derived_metric", "confidence": 0.6})
                    rationale.append(f"Name hints ({', '.join(strict_derived)}) suggest a derived measure")
                    return ColumnSemantics(column, "derived_metric", 0.72, candidates, rationale).to_dict()
                if derived_hits:
                    candidates.append({"role": "derived_metric", "confidence": 0.5})
                    rationale.append(f"Name hints ({', '.join(derived_hits)}) suggest a derived measure")
                return ColumnSemantics(column, "metric", 0.6, candidates, rationale).to_dict()
            if unique <= 2:
                return ColumnSemantics(column, "category", 0.6,
                                       [{"role": "category", "confidence": 0.6},
                                        {"role": "metric", "confidence": 0.3}],
                                       ["Numeric column with very low cardinality; likely coded categories"]).to_dict()
            return ColumnSemantics(column, "metric", 0.5,
                                   [{"role": "metric", "confidence": 0.5},
                                    {"role": "constant", "confidence": 0.3}],
                                   ["Numeric column with no variance"]).to_dict()

        # 8) Categorical / text / entity columns
        if unique_ratio >= 0.9:
            entity_hits = _token_hits(normalized, self.hints.get("entity", ()))
            text_hits = _token_hits(normalized, self.hints.get("text_words", ()))
            if entity_hits:
                return ColumnSemantics(column, "entity", 0.7,
                                       [{"role": "entity", "confidence": 0.7}],
                                       [f"Name hints ({', '.join(entity_hits)}) and near-unique values suggest an entity"]).to_dict()
            if text_hits:
                return ColumnSemantics(column, "text", 0.7,
                                       [{"role": "text", "confidence": 0.7}],
                                       [f"Name hints ({', '.join(text_hits)}) and near-unique values suggest free text"]).to_dict()
            avg_len = float(non_null.astype(str).str.len().mean())
            if avg_len >= 20:
                return ColumnSemantics(column, "text", 0.6,
                                       [{"role": "text", "confidence": 0.6},
                                        {"role": "entity", "confidence": 0.3}],
                                       ["Near-unique values with long content; treated as free text"]).to_dict()
            return ColumnSemantics(column, "unknown", 0.4,
                                   [{"role": "text", "confidence": 0.4},
                                    {"role": "dimension", "confidence": 0.4}],
                                   ["High-cardinality string column; no reliable role"]).to_dict()

        # 9) Low / medium cardinality categorical columns
        if unique <= 2:
            candidates = [{"role": "category", "confidence": 0.8},
                          {"role": "dimension", "confidence": 0.7}]
            rationale = ["Binary-valued column"]
        elif unique_ratio <= 0.05 and unique >= 2:
            candidates = [{"role": "dimension", "confidence": 0.75},
                          {"role": "category", "confidence": 0.7}]
            rationale = ["Low-cardinality string column suitable as a grouping dimension"]
        else:
            candidates = [{"role": "category", "confidence": 0.5},
                          {"role": "text", "confidence": 0.4}]
            rationale = ["String column with moderate cardinality"]

        dim_hits = _token_hits(normalized, self.hints.get("dimension_words", ()))
        entity_hits = _token_hits(normalized, self.hints.get("entity", ()))
        if dim_hits:
            candidates.append({"role": "dimension", "confidence": 0.85})
            rationale.append(f"Name hints ({', '.join(dim_hits)}) suggest a dimension")
        if entity_hits:
            candidates.append({"role": "entity", "confidence": 0.7})
            rationale.append(f"Name hints ({', '.join(entity_hits)}) suggest an entity")
            if candidates[0]["role"] == "category":
                return ColumnSemantics(column, "entity", 0.6, candidates, rationale).to_dict()

        best = max(candidates, key=lambda item: item["confidence"])
        return ColumnSemantics(column, best["role"], best["confidence"],
                               candidates, rationale).to_dict()

    def _temporal_integer(self, series: pd.Series, normalized: str) -> dict[str, Any] | None:
        if not _is_integer_dtype(series):
            return None
        non_null = series.dropna()
        if non_null.empty:
            return None
        unique = int(non_null.nunique())
        year_hits = _token_hits(normalized, ("year", "yr"))
        month_hits = _token_hits(normalized, ("month", "mth"))
        quarter_hits = _token_hits(normalized, ("quarter", "qtr"))

        if year_hits and unique <= 60:
            values = non_null.astype(int)
            if ((values >= 1900) & (values <= 2100)).mean() >= 0.95:
                return {"role": "year", "confidence": 0.9,
                        "candidates": [{"role": "year", "confidence": 0.9},
                                       {"role": "category", "confidence": 0.2}],
                        "rationale": ["Integer column with year-scale values and matching name hint"]}
        if month_hits and unique <= 12:
            values = non_null.astype(int)
            if ((values >= 1) & (values <= 12)).mean() >= 0.95:
                return {"role": "month", "confidence": 0.9,
                        "candidates": [{"role": "month", "confidence": 0.9},
                                       {"role": "category", "confidence": 0.2}],
                        "rationale": ["Integer column with 1..12 values and matching name hint"]}
        if quarter_hits and unique <= 4:
            values = non_null.astype(int)
            if ((values >= 1) & (values <= 4)).mean() >= 0.95:
                return {"role": "quarter", "confidence": 0.9,
                        "candidates": [{"role": "quarter", "confidence": 0.9},
                                       {"role": "category", "confidence": 0.2}],
                        "rationale": ["Integer column with 1..4 values and matching name hint"]}
        return None

    def _identifier_candidate(self, series: pd.Series, column: str, unique: int,
                              total: int, unique_ratio: float) -> tuple[str, float, list[dict[str, float]], list[str]] | None:
        normalized = _normalize(column)
        name_hits = _token_hits(normalized, self.hints.get("identifier", ()))
        exact_id = normalized in {"id", "identifier", "key", "sku", "uuid", "guid"}
        if not (name_hits or exact_id):
            return None
        if unique_ratio < 0.5:
            return None
        dtype_score = 0.0
        if _is_numeric(series):
            if _is_integer_dtype(series):
                dtype_score = 0.2
        else:
            avg_len = float(series.dropna().astype(str).str.len().mean()) if series.notna().any() else 99
            dtype_score = 0.25 if avg_len <= 24 else 0.0
        conf = 0.5 + 0.25 * unique_ratio + dtype_score
        conf = min(0.98, conf)
        return ("identifier", conf,
                [{"role": "identifier", "confidence": conf},
                 {"role": "text", "confidence": max(0.0, 1.0 - conf)}],
                [f"Name hints ({', '.join(name_hits) or normalized}) and uniqueness ratio {unique_ratio:.2f}"])


def detect_identifiers(dataframe: pd.DataFrame) -> list[str]:
    """Statistical identifier detection without name hints.

    Identifier columns typically have near-unique values, no repeated
    pattern, integer dtype (or short string), and no natural ordering.
    """
    identified: list[str] = []
    for column in dataframe.columns:
        series = dataframe[column]
        total = len(series)
        if total == 0:
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        unique_ratio = non_null.nunique() / total
        if unique_ratio < 0.9:
            continue
        if pd.api.types.is_float_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            values = non_null.astype(int)
            if ((values >= 1900) & (values <= 2100)).mean() >= 0.9:
                continue
            identified.append(str(column))
        else:
            avg_len = float(non_null.astype(str).str.len().mean())
            if avg_len <= 40:
                identified.append(str(column))
    return identified
