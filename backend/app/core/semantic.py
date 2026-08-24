"""Semantic Schema Agent.

Classifies every column into a semantic role based on:
  - data type
  - sample values
  - cardinality / uniqueness
  - statistical characteristics
  - relationships (optional, passed in)
  - domain concept matching (revenue, profit, cost, volume, price, churn, etc.)
  - contextual disambiguation (distinguishing similar concepts using confidence and data distribution)

Never forces a classification when confidence is low. All hints live in the
configurable semantic dictionary (SEMANTIC_HINTS) and BUSINESS_CONCEPTS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent.schemas import ClaimType, Evidence, SemanticMapping
from backend.app.core.dataset_knowledge import DatasetKnowledge

# ---------------------------------------------------------------------------
# Business concepts registry for contextual semantic matching.
# Maps canonical concept keys to their category, default aliases, and properties.
# ---------------------------------------------------------------------------
BUSINESS_CONCEPTS: Dict[str, Dict[str, Any]] = {
    "revenue": {
        "category": "financial",
        "aliases": (
            "sales", "sales_amount", "gross_revenue", "net_revenue", "turnover",
            "top_line", "receipts", "gross_sales", "income", "total_revenue",
            "net_sales", "billings", "revenue_amount", "revenue_usd",
        ),
        "description": "Financial inflow generated from goods or services.",
        "expected_type": "numeric",
        "typically_positive": True,
    },
    "cost": {
        "category": "financial",
        "aliases": (
            "cogs", "cost_of_goods", "cost_of_sales", "expenses", "expenditure",
            "operating_expenses", "opex", "capex", "spend", "spending",
            "total_cost", "unit_cost", "direct_costs", "overhead",
        ),
        "description": "Financial expenditure incurred in business operations.",
        "expected_type": "numeric",
        "typically_positive": True,
    },
    "profit": {
        "category": "financial",
        "aliases": (
            "net_profit", "gross_profit", "margin", "net_income", "ebitda",
            "operating_income", "earnings", "bottom_line", "profit_amount",
            "profit_margin", "operating_profit", "ebit",
        ),
        "description": "Financial gain representing revenue minus expenses.",
        "expected_type": "numeric",
        "typically_positive": False,  # can be negative (loss)
    },
    "quantity": {
        "category": "volume",
        "aliases": (
            "qty", "units", "volume", "units_sold", "count", "item_count",
            "number_of_items", "order_quantity", "quantity_sold", "stock_qty",
        ),
        "description": "Discrete or continuous count of units/items.",
        "expected_type": "numeric",
        "typically_positive": True,
    },
    "price": {
        "category": "pricing",
        "aliases": (
            "unit_price", "rate", "retail_price", "selling_price", "list_price",
            "price_per_unit", "mrp", "standard_price", "base_price",
        ),
        "description": "Per-unit financial charge.",
        "expected_type": "numeric",
        "typically_positive": True,
    },
    "discount": {
        "category": "pricing",
        "aliases": (
            "discount_amount", "discount_rate", "rebate", "coupon_discount",
            "price_cut", "promo_discount", "markdown", "pct_discount",
        ),
        "description": "Reduction applied to standard price or billings.",
        "expected_type": "numeric",
        "typically_positive": True,
    },
    "salary": {
        "category": "hr",
        "aliases": (
            "wage", "compensation", "payroll", "bonus", "hourly_rate",
            "annual_salary", "base_pay", "total_compensation", "remuneration",
        ),
        "description": "Personnel compensation and wage figures.",
        "expected_type": "numeric",
        "typically_positive": True,
    },
    "customer": {
        "category": "customer",
        "aliases": (
            "client", "user", "account", "subscriber", "member", "shopper",
            "buyer", "customer_name", "client_name", "user_name",
        ),
        "description": "Customer, client, or user entity.",
        "expected_type": "entity",
    },
    "churn": {
        "category": "customer",
        "aliases": (
            "attrition", "cancellation", "drop_off", "unsubscribe",
            "is_churned", "churn_flag", "churn_rate", "exited",
        ),
        "description": "Customer attrition or cancellation indicator.",
        "expected_type": "boolean_or_numeric",
    },
    "tenure": {
        "category": "customer",
        "aliases": (
            "duration", "customer_age", "membership_duration", "months_active",
            "days_active", "account_age", "service_years",
        ),
        "description": "Duration or length of active relationship.",
        "expected_type": "numeric",
        "typically_positive": True,
    },
    "rating": {
        "category": "quality",
        "aliases": (
            "score", "feedback_score", "review_rating", "nps", "csat",
            "stars", "sentiment_score", "satisfaction", "grade",
        ),
        "description": "Numerical or ordinal rating and satisfaction score.",
        "expected_type": "numeric",
    },
    "location": {
        "category": "geography",
        "aliases": (
            "country", "state", "city", "region", "postal_code", "zipcode",
            "latitude", "longitude", "territory", "zone", "province", "county",
        ),
        "description": "Geographic or spatial dimension.",
        "expected_type": "dimension",
    },
    "timestamp": {
        "category": "temporal",
        "aliases": (
            "date", "datetime", "created_at", "order_date", "transaction_date",
            "time", "timestamp_utc", "updated_at", "event_time", "period",
        ),
        "description": "Temporal point or time reference.",
        "expected_type": "datetime",
    },
}

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

    def match_concept(
        self,
        column_name: str,
        series: Optional[pd.Series] = None,
        relationships: Optional[List[Dict[str, Any]]] = None,
    ) -> SemanticMapping:
        """Contextually match a column to a business semantic concept with confidence and evidence."""
        norm = _normalize(column_name)
        best_concept = "unknown"
        best_category = "general"
        best_confidence = 0.4
        best_aliases: List[str] = []
        best_description = ""
        matched_token = ""

        for concept_key, concept_info in BUSINESS_CONCEPTS.items():
            aliases = concept_info["aliases"]
            cat = concept_info["category"]
            desc = concept_info["description"]

            # Exact key match
            if norm == concept_key:
                score = 0.95
                matched_token = concept_key
            # Exact alias match
            elif norm in aliases:
                score = 0.92
                matched_token = norm
            else:
                # Token-level hits (e.g. sales in monthly_sales or sales_usd)
                tokens = norm.split("_")
                key_in_tokens = concept_key in tokens
                alias_in_tokens = [a for a in aliases if a in tokens or (f"_{a}" in f"_{norm}" and len(a) > 2)]

                if key_in_tokens:
                    score = 0.88
                    matched_token = concept_key
                elif alias_in_tokens:
                    # Longer alias matches get higher score
                    best_alias = max(alias_in_tokens, key=len)
                    score = 0.82 if len(best_alias) >= 4 else 0.72
                    matched_token = best_alias
                elif any(a in norm for a in aliases if len(a) >= 4):
                    score = 0.68
                    matched_token = [a for a in aliases if a in norm and len(a) >= 4][0]
                else:
                    continue

            # Contextual statistical validation if series is provided
            if series is not None and not series.dropna().empty:
                exp_type = concept_info.get("expected_type")
                if exp_type == "numeric":
                    if _is_numeric(series):
                        score = min(0.98, score + 0.05)
                        if concept_info.get("typically_positive") and len(series.dropna()) > 5:
                            vals = pd.to_numeric(series.dropna(), errors="coerce")
                            if (vals >= 0).mean() >= 0.95:
                                score = min(0.99, score + 0.02)
                    else:
                        # Expected numeric but is string/object -> downweight
                        score = max(0.2, score - 0.3)
                elif exp_type == "datetime":
                    if pd.api.types.is_datetime64_any_dtype(series):
                        score = min(0.99, score + 0.1)
                    else:
                        parsed = pd.to_datetime(series.dropna(), errors="coerce")
                        if parsed.notna().mean() >= 0.8:
                            score = min(0.95, score + 0.05)

            if score > best_confidence:
                best_confidence = score
                best_concept = concept_key
                best_category = cat
                best_aliases = list(aliases)
                best_description = desc

        # If no business concept matched with high confidence, use general classification
        if best_confidence <= 0.45 and series is not None:
            if _is_numeric(series):
                best_concept = "metric"
                best_category = "numeric"
                best_confidence = 0.6
                best_description = "General numeric measure."
            elif pd.api.types.is_datetime64_any_dtype(series):
                best_concept = "timestamp"
                best_category = "temporal"
                best_confidence = 0.9
                best_description = "Temporal datetime column."
            elif series.dropna().nunique() / max(1, len(series.dropna())) <= 0.2:
                best_concept = "dimension"
                best_category = "categorical"
                best_confidence = 0.7
                best_description = "Categorical grouping dimension."
            else:
                best_concept = "attribute"
                best_category = "text"
                best_confidence = 0.5
                best_description = "General textual attribute."

        evidence = [
            Evidence(
                source="SemanticSchemaAgent",
                method="concept_matching",
                data_ref={
                    "column": column_name,
                    "matched_concept": best_concept,
                    "matched_token": matched_token,
                    "category": best_category,
                },
                confidence=round(best_confidence, 3),
                claim_type=ClaimType.FACT if best_confidence >= 0.9 else ClaimType.OBSERVATION,
            )
        ]

        return SemanticMapping(
            column_name=column_name,
            semantic_concept=best_concept,
            concept_category=best_category,
            confidence=round(best_confidence, 3),
            evidence=evidence,
            aliases=best_aliases,
            description=best_description or f"Identified as {best_concept} ({best_category})",
        )

    def extract_semantic_mappings(
        self,
        dataframe: pd.DataFrame,
        relationships: Optional[List[Dict[str, Any]]] = None,
    ) -> List[SemanticMapping]:
        """Generate a SemanticMapping for each column in the dataframe."""
        mappings: List[SemanticMapping] = []
        for column in dataframe.columns:
            series = dataframe[column]
            mapping = self.match_concept(column, series=series, relationships=relationships)
            mappings.append(mapping)
        return mappings

    def build_knowledge(
        self,
        dataframe: pd.DataFrame,
        dataset_id: str = "dataset_001",
        dataset_type: str = "auto",
    ) -> DatasetKnowledge:
        """
        Construct a complete DatasetKnowledge object representing the semantic
        and analytical profile of the dataset.
        """
        from backend.app.core.quality import DataQualityEngine
        from backend.app.core.relationships import RelationshipDiscoveryEngine

        # 1. Column semantic classifications & mappings
        rel_engine = RelationshipDiscoveryEngine()
        relationships_data = rel_engine.discover(dataframe).get("relationships", [])
        roles = self.roles_map(dataframe)
        mappings = self.extract_semantic_mappings(dataframe, relationships=relationships_data)
        mapping_by_col = {m.column_name: m for m in mappings}

        # 2. Categorize column types
        metrics: List[SemanticMapping] = []
        dimensions: List[SemanticMapping] = []
        date_columns: List[SemanticMapping] = []
        identifiers: List[SemanticMapping] = []
        categorical_cols: List[str] = []
        numeric_cols: List[str] = []
        data_types: Dict[str, str] = {}
        missing_values: Dict[str, Any] = {}
        confidence_scores: Dict[str, float] = {}
        semantic_meanings: Dict[str, str] = {}

        for col in dataframe.columns:
            series = dataframe[col]
            dtype_str = str(series.dtype)
            data_types[str(col)] = dtype_str
            missing_count = int(series.isna().sum())
            missing_pct = round((missing_count / max(1, len(dataframe))) * 100, 2)
            missing_values[str(col)] = {"count": missing_count, "percentage": missing_pct}

            mapping = mapping_by_col.get(str(col))
            confidence = mapping.confidence if mapping else 0.5
            confidence_scores[str(col)] = round(float(confidence), 3)
            semantic_meanings[str(col)] = mapping.description if mapping else roles.get(str(col), "attribute")

            role = roles.get(str(col), "unknown")
            is_num = _is_numeric(series)

            if is_num:
                numeric_cols.append(str(col))
            else:
                categorical_cols.append(str(col))

            if role in ("metric", "derived_metric", "target_candidate") or (mapping and mapping.concept_category in ("financial", "volume", "pricing", "hr") and is_num):
                if mapping:
                    metrics.append(mapping)
            elif role in ("date", "time", "datetime", "year", "month", "quarter") or (mapping and mapping.concept_category == "temporal"):
                if mapping:
                    date_columns.append(mapping)
            elif role == "identifier" or (mapping and mapping.concept_category == "identifier"):
                if mapping:
                    identifiers.append(mapping)
            else:
                if mapping:
                    dimensions.append(mapping)

        # 3. Assess Data Quality
        quality_engine = DataQualityEngine()
        quality_data = quality_engine.assess(
            dataframe,
            semantic_roles=roles,
            identifiers=[i.column_name for i in identifiers],
        )

        # 4. Infer Dataset Type if auto
        inferred_type = dataset_type
        if dataset_type == "auto":
            if date_columns and numeric_cols:
                inferred_type = "time_series"
            elif any(m.concept_category == "financial" for m in mappings):
                inferred_type = "financial"
            elif any(m.concept_category in ("volume", "pricing") for m in mappings):
                inferred_type = "transactional"
            elif any(m.concept_category == "customer" for m in mappings):
                inferred_type = "customer"
            else:
                inferred_type = "tabular"

        # 5. Overall confidence calculation
        avg_confidence = float(np.mean(list(confidence_scores.values()))) if confidence_scores else 1.0
        quality_factor = float(quality_data.get("quality_score", 100)) / 100.0
        overall_confidence = round(min(1.0, max(0.1, avg_confidence * 0.7 + quality_factor * 0.3)), 3)

        metadata = {
            "row_count": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "memory_usage_bytes": int(dataframe.memory_usage(deep=True).sum()),
            "roles": roles,
        }

        return DatasetKnowledge(
            dataset_id=dataset_id,
            dataset_type=inferred_type,
            columns=[str(c) for c in dataframe.columns],
            data_types=data_types,
            semantic_meanings=semantic_meanings,
            metrics=metrics,
            dimensions=dimensions,
            date_columns=date_columns,
            identifiers=identifiers,
            semantic_mappings=mappings,
            categorical_columns=categorical_cols,
            numeric_columns=numeric_cols,
            relationships=relationships_data,
            missing_values=missing_values,
            data_quality=quality_data,
            confidence_scores=confidence_scores,
            entities=[],
            overall_confidence=overall_confidence,
            metadata=metadata,
        )

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
