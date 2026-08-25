"""
Semantic Schema Agent - Multi-Agent Dataset Intelligence and Semantic Schema Profiler.

Inspects dataset structure, statistical characteristics, cardinality, naming patterns,
and sample values to determine semantic types (METRIC, DIMENSION, IDENTIFIER, DATE, etc.)
with deterministic confidence scoring and traceable mathematical evidence.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from agent.base import BaseAgent
from agent.dataset_knowledge import (
    ColumnKnowledge,
    DataQuality,
    DatasetKnowledge,
    SemanticType,
)
from agent.schemas import AgentResult, ClaimType, Evidence, SemanticMapping
from backend.app.core.semantic import BUSINESS_CONCEPTS, SEMANTIC_HINTS, _normalize, _token_hits
from backend.app.profilers.dataset_profiler import DatasetProfiler


class SemanticSchemaAgent(BaseAgent):
    """
    Autonomous agent for dataset semantic profiling and ontology construction.
    Produces a complete DatasetKnowledge artifact for all downstream agents.
    """

    name = "Semantic Schema Agent"
    description = "Inspects schemas, column distributions, and semantic roles."
    role = "schema_profiler"

    def __init__(self, hints: Optional[Dict[str, Tuple[str, ...]]] = None):
        super().__init__()
        self.hints = hints or SEMANTIC_HINTS
        self.profiler = DatasetProfiler()

    # ------------------------------------------------------------------
    # Core Pipeline
    # ------------------------------------------------------------------
    def analyze_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        dataset_id: Optional[str] = None,
    ) -> DatasetKnowledge:
        """
        Execute full semantic profiling pipeline on a pandas DataFrame.
        Produces a rich, immutable DatasetKnowledge object.
        """
        d_id = dataset_id or dataset_name
        n_rows, n_cols = df.shape

        # 1. Run raw structural and data quality profiling via DatasetProfiler
        profile_dict = self.profiler.profile(
            dataframe=df,
            filename=dataset_name,
            file_type="dataframe",
            file_size=f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB",
        )

        data_quality_info = self._extract_data_quality(df, profile_dict)

        # 2. Classify every individual column
        columns_knowledge: List[ColumnKnowledge] = []
        column_knowledge_map: Dict[str, ColumnKnowledge] = {}
        semantic_mappings: List[SemanticMapping] = []
        all_evidence: List[Evidence] = []

        metrics_list: List[SemanticMapping] = []
        dims_list: List[SemanticMapping] = []
        dates_list: List[SemanticMapping] = []
        ids_list: List[SemanticMapping] = []
        categorical_cols: List[str] = []
        numerical_cols: List[str] = []
        target_candidates: List[str] = []
        data_types: Dict[str, str] = {}
        semantic_meanings: Dict[str, str] = {}
        confidence_scores: Dict[str, float] = {}

        for col in df.columns:
            series = df[col]
            ck, sm, ev_list = self._classify_column(series, col, df)
            columns_knowledge.append(ck)
            column_knowledge_map[col] = ck
            semantic_mappings.append(sm)
            all_evidence.extend(ev_list)

            data_types[col] = str(series.dtype)
            semantic_meanings[col] = f"{sm.semantic_concept.capitalize()} ({sm.concept_category})"
            confidence_scores[col] = ck.confidence

            # Categorize into high-level lists
            st = ck.semantic_type
            if st == SemanticType.METRIC:
                metrics_list.append(sm)
                numerical_cols.append(col)
            elif st in (SemanticType.DIMENSION, SemanticType.CATEGORY):
                dims_list.append(sm)
                categorical_cols.append(col)
            elif st in (SemanticType.DATE, SemanticType.DATETIME):
                dates_list.append(sm)
            elif st == SemanticType.IDENTIFIER:
                ids_list.append(sm)
            elif st == SemanticType.BOOLEAN:
                categorical_cols.append(col)

            # Identify target candidates (e.g. churn, status, price, score, default)
            if ck.role in ("target_candidate", "metric") and ck.concept in ("churn", "revenue", "profit", "rating", "salary", "conversion"):
                target_candidates.append(col)

        # 3. Overall confidence score
        overall_conf = round(float(np.mean([ck.confidence for ck in columns_knowledge])), 4) if columns_knowledge else 1.0

        # 4. Infer dataset type (financial, time_series, customer, transactional, tabular)
        dataset_type = self._infer_dataset_type(dates_list, metrics_list, semantic_mappings)

        # 5. Build unified DatasetKnowledge
        dk = DatasetKnowledge(
            dataset_id=d_id,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            row_count=n_rows,
            column_count=n_cols,
            columns=columns_knowledge,
            column_knowledge=column_knowledge_map,
            data_types=data_types,
            semantic_meanings=semantic_meanings,
            metrics=metrics_list,
            dimensions=dims_list,
            identifiers=ids_list,
            date_columns=dates_list,
            categorical_columns=categorical_cols,
            numerical_columns=numerical_cols,
            numeric_columns=numerical_cols,
            target_candidates=target_candidates,
            relationships=[],
            data_quality=data_quality_info,
            missing_values={col: int(df[col].isna().sum()) for col in df.columns},
            semantic_mappings=semantic_mappings,
            confidence=overall_conf,
            overall_confidence=overall_conf,
            confidence_scores=confidence_scores,
            entities=[],
            metadata={"profiler_version": "2.0"},
        )
        return dk

    # ------------------------------------------------------------------
    # Agent Execution Interface
    # ------------------------------------------------------------------
    def run(self, task: Union[Dict[str, Any], pd.DataFrame]) -> AgentResult:
        """
        Execute task returning standardized AgentResult.
        Accepts raw DataFrame or task dict {'data': df, 'name': 'sales.csv'}.
        """
        self._start()
        try:
            df = None
            name = "dataset"
            if isinstance(task, pd.DataFrame):
                df = task
            elif isinstance(task, dict):
                df = task.get("data")
                name = task.get("name") or task.get("source") or task.get("dataset_name", "dataset")
            
            if df is None or not isinstance(df, pd.DataFrame):
                return self._error(
                    message="SemanticSchemaAgent requires a valid pandas DataFrame.",
                    code="INVALID_DATAFRAME_INPUT",
                )

            dk = self.analyze_dataset(df, dataset_name=name)
            
            # Collect evidence items
            evidence_items: List[Evidence] = []
            for col_k in dk.columns:
                if isinstance(col_k, ColumnKnowledge):
                    evidence_items.extend(col_k.evidence)

            return self._finish(
                result={"dataset_knowledge": dk.to_dict()},
                message=f"SemanticSchemaAgent successfully profiled '{name}' ({dk.row_count} rows, {dk.column_count} columns).",
                evidence=evidence_items,
                confidence=dk.confidence,
                metadata={"dataset_id": dk.dataset_id, "dataset_type": dk.dataset_type},
            )
        except Exception as exc:
            return self._error(
                message=f"Failed to profile dataset semantics: {str(exc)}",
                code="PROFILING_ERROR",
                details={"exception": str(exc)},
            )

    # ------------------------------------------------------------------
    # Semantic Detection Logic
    # ------------------------------------------------------------------
    def _classify_column(
        self,
        series: pd.Series,
        col_name: str,
        df: pd.DataFrame,
    ) -> Tuple[ColumnKnowledge, SemanticMapping, List[Evidence]]:
        """Determine the semantic type, role, concept, and confidence for a column."""
        n_rows = len(df)
        valid_series = series.dropna()
        n_valid = len(valid_series)
        n_unique = int(series.nunique(dropna=True))
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / n_rows) * 100, 2) if n_rows > 0 else 0.0
        unique_ratio = (n_unique / n_valid) if n_valid > 0 else 0.0
        sample_vals = [self._clean_sample_val(v) for v in valid_series.head(5).tolist()]

        norm_name = _normalize(col_name)
        is_num = pd.api.types.is_numeric_dtype(series)
        is_dt = pd.api.types.is_datetime64_any_dtype(series)
        is_bool = pd.api.types.is_bool_dtype(series) or (n_unique <= 2 and set(sample_vals).issubset({True, False, 0, 1, "0", "1", "true", "false", "yes", "no", "y", "n"}))

        # Initial candidates
        sem_type = SemanticType.UNKNOWN
        role = "unknown"
        concept = norm_name
        category = "general"
        confidence = 0.50
        rationale: List[str] = []
        aliases: List[str] = []

        # 1. Check Date / Datetime / Temporal (including integer Year / Quarter)
        if is_dt or self._is_potential_date(series, norm_name):
            sem_type = SemanticType.DATETIME if is_dt or "time" in norm_name else SemanticType.DATE
            role = "date"
            concept = "year" if any(t in norm_name for t in ("year", "fy", "cy", "yr")) else "quarter" if any(t in norm_name for t in ("quarter", "qtr")) else "timestamp"
            category = "temporal"
            confidence = 0.95 if is_dt else 0.90
            rationale.append(f"Column '{col_name}' detected as temporal/chronological dimension.")
            aliases = ["date", "datetime", "timestamp", "year", "quarter", "period"]

        # 2. Check Identifier
        elif self._is_identifier(col_name, series, n_rows, unique_ratio, is_num):
            sem_type = SemanticType.IDENTIFIER
            role = "identifier"
            concept = "identifier"
            category = "system"
            confidence = 0.95 if any(norm_name.endswith(f"_{t}") or norm_name == t for t in ("id", "key", "uuid", "sku", "code")) else 0.85
            rationale.append(f"Column '{col_name}' has uniqueness ratio {unique_ratio:.2f} and identifier naming pattern.")
            aliases = ["id", "key", "code"]

        # 3. Check Boolean Flag
        elif is_bool and not (is_num and n_unique > 2):
            sem_type = SemanticType.BOOLEAN
            role = "dimension"
            concept = "boolean_flag"
            category = "indicator"
            confidence = 0.92
            rationale.append(f"Column '{col_name}' contains binary/boolean indicator values.")
            aliases = ["flag", "indicator"]

        # 4. Check Business Concepts Dictionary (e.g. revenue, profit, cost, budget, actual, churn, rating)
        else:
            matched_concept, b_conf, b_cat = self._match_business_concept(col_name, series)
            if matched_concept:
                concept = matched_concept
                category = b_cat
                confidence = b_conf
                aliases = list(BUSINESS_CONCEPTS[matched_concept].get("aliases", []))
                exp_type = BUSINESS_CONCEPTS[matched_concept].get("expected_type", "general")

                if exp_type in ("numeric", "volume", "pricing") or (is_num and exp_type != "temporal"):
                    sem_type = SemanticType.METRIC
                    role = "metric"
                    rationale.append(f"Column '{col_name}' matches financial/numerical business concept '{matched_concept}'.")
                elif exp_type in ("temporal", "datetime"):
                    sem_type = SemanticType.DATE
                    role = "date"
                    category = "temporal"
                    rationale.append(f"Column '{col_name}' matches temporal concept '{matched_concept}'.")
                elif matched_concept == "churn":
                    sem_type = SemanticType.TARGET
                    role = "target_candidate"
                    rationale.append(f"Column '{col_name}' matches churn/attrition target concept.")
                elif matched_concept == "location":
                    sem_type = SemanticType.DIMENSION
                    role = "dimension"
                    rationale.append(f"Column '{col_name}' matches geographic location dimension.")
                else:
                    sem_type = SemanticType.DIMENSION
                    role = "dimension"
                    rationale.append(f"Column '{col_name}' matches domain entity concept '{matched_concept}'.")

            # 5. Fallback Numeric Check (METRIC vs DIMENSION)
            elif is_num:
                if n_unique <= 5 and n_rows > 20 and not pd.api.types.is_float_dtype(series):
                    sem_type = SemanticType.DIMENSION
                    role = "dimension"
                    category = "categorical_numeric"
                    confidence = 0.75
                    rationale.append(f"Low cardinality numeric column ({n_unique} unique) classified as grouping dimension.")
                else:
                    sem_type = SemanticType.METRIC
                    role = "metric"
                    category = "numerical"
                    confidence = 0.85 if pd.api.types.is_float_dtype(series) else 0.78
                    rationale.append(f"Continuous numeric distribution with mean {float(series.mean()):.2f} classified as METRIC.")

            # 6. Fallback Categorical / Text Check
            else:
                if unique_ratio > 0.80 and not any(t in norm_name for t in ("category", "type", "group", "class", "region", "city", "status", "segment", "country")):
                    sem_type = SemanticType.UNKNOWN
                    role = "uncertain"
                    category = "ambiguous"
                    confidence = 0.50
                    rationale.append(f"High uniqueness string column ({n_unique}/{n_valid}) without semantic hints; flagged as uncertain.")
                elif n_unique <= 50 or unique_ratio < 0.20:
                    sem_type = SemanticType.DIMENSION
                    role = "dimension"
                    category = "categorical"
                    confidence = 0.85
                    rationale.append(f"String column with {n_unique} categories classified as grouping DIMENSION.")
                else:
                    # Long text vs Ambiguous category
                    mean_len = float(series.astype(str).str.len().mean()) if n_valid > 0 else 0
                    if mean_len > 40:
                        sem_type = SemanticType.TEXT
                        role = "text"
                        category = "freeform_text"
                        confidence = 0.85
                        rationale.append(f"Freeform text column (avg length {mean_len:.1f} chars).")
                    else:
                        sem_type = SemanticType.DIMENSION
                        role = "dimension"
                        category = "categorical"
                        confidence = 0.55
                        rationale.append(f"High-cardinality nominal column ({n_unique} unique values).")

        # Check for ambiguity & mark uncertain
        is_uncertain = confidence < 0.60
        if is_uncertain:
            rationale.append("Low confidence classification; flagged as uncertain.")

        # Compute stats for numeric columns
        min_v = float(series.min()) if is_num and not series.empty and pd.notna(series.min()) else None
        max_v = float(series.max()) if is_num and not series.empty and pd.notna(series.max()) else None
        mean_v = float(series.mean()) if is_num and not series.empty and pd.notna(series.mean()) else None

        # Build evidence
        ev = self.make_evidence(
            method="semantic_classifier",
            data_ref={
                "column": col_name,
                "dtype": str(series.dtype),
                "unique_count": n_unique,
                "missing_count": missing_count,
                "sample_values": sample_vals,
            },
            confidence=confidence,
            claim_type=ClaimType.FACT if confidence >= 0.90 else ClaimType.OBSERVATION,
            raw_value={"semantic_type": sem_type.value, "role": role, "concept": concept},
            metadata={"rationale": " | ".join(rationale)},
            dataset_name=df.attrs.get("name", "dataset"),
            columns=[col_name],
            operation=f"classify_{sem_type.value.lower()}",
            calculation=f"P({concept}|{col_name}) = {confidence:.2f}",
        )

        # Build SemanticMapping
        sm = SemanticMapping(
            column_name=col_name,
            semantic_concept=concept,
            concept_category=category,
            confidence=confidence,
            evidence=[ev],
            aliases=aliases,
            description=" | ".join(rationale),
        )

        # Build ColumnKnowledge
        ck = ColumnKnowledge(
            column_name=col_name,
            data_type=str(series.dtype),
            semantic_type=sem_type,
            role=role,
            unique_count=n_unique,
            missing_count=missing_count,
            missing_percentage=missing_pct,
            sample_values=sample_vals,
            min_value=min_v,
            max_value=max_v,
            mean=mean_v,
            cardinality=n_unique,
            confidence=confidence,
            concept=concept,
            is_uncertain=is_uncertain,
            evidence=[ev],
        )

        return ck, sm, [ev]

    # ------------------------------------------------------------------
    # Heuristics & Matching Helpers
    # ------------------------------------------------------------------
    def _match_business_concept(self, col_name: str, series: pd.Series) -> Tuple[Optional[str], float, str]:
        """Match column name against BUSINESS_CONCEPTS ontology."""
        norm_name = _normalize(col_name)

        # 1. Exact match with canonical concept key
        if norm_name in BUSINESS_CONCEPTS:
            return norm_name, 0.96, BUSINESS_CONCEPTS[norm_name]["category"]

        # 2. Exact match with alias
        for concept_key, spec in BUSINESS_CONCEPTS.items():
            aliases = spec.get("aliases", ())
            if norm_name in aliases:
                return concept_key, 0.94, spec["category"]

        # 3. Substring / token matching
        best_match = None
        highest_conf = 0.0
        best_cat = "general"

        for concept_key, spec in BUSINESS_CONCEPTS.items():
            # Check concept key in name
            if concept_key in norm_name:
                conf = 0.90
                if conf > highest_conf:
                    highest_conf = conf
                    best_match = concept_key
                    best_cat = spec["category"]

            # Check aliases in name
            for alias in spec.get("aliases", ()):
                if f"_{alias}" in f"_{norm_name}" or norm_name.endswith(f"_{alias}") or norm_name.startswith(f"{alias}_"):
                    conf = 0.88
                    if conf > highest_conf:
                        highest_conf = conf
                        best_match = concept_key
                        best_cat = spec["category"]

        if best_match:
            return best_match, highest_conf, best_cat
        return None, 0.50, "general"

    def _is_potential_date(self, series: pd.Series, norm_name: str) -> bool:
        """Safely test if a column represents date/timestamps/years without mutating data."""
        # 1. Check naming tokens
        date_tokens = ("date", "time", "created_at", "updated_at", "timestamp", "dob", "year", "month", "dt", "quarter", "qtr", "fy", "cy", "period")
        tokens = norm_name.split("_")
        name_hint = any(token == norm_name or norm_name.endswith(f"_{token}") or norm_name.startswith(f"{token}_") or token in tokens for token in date_tokens)

        # 2. Check numeric years, quarters, months
        if pd.api.types.is_numeric_dtype(series):
            s_clean = series.dropna()
            if not s_clean.empty:
                min_v, max_v = s_clean.min(), s_clean.max()
                # Integer years (e.g. 2020-2030 or FiscalYear)
                if min_v >= 1800 and max_v <= 2150 and (name_hint or (pd.api.types.is_integer_dtype(series) and max_v - min_v <= 100)):
                    return True
                # Quarters (1-4) with quarter naming hint
                if min_v >= 1 and max_v <= 4 and any(t in norm_name for t in ("quarter", "qtr")):
                    return True
                # Months (1-12) with month naming hint
                if min_v >= 1 and max_v <= 12 and any(t in norm_name for t in ("month", "mon")):
                    return True
            return False

        if not name_hint and not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(series):
            return False

        # 3. Sample value inspection for string/object columns
        sample = series.dropna().head(10)
        if sample.empty:
            return name_hint

        parsed_count = 0
        for val in sample:
            s_val = str(val).strip()
            if not s_val or len(s_val) < 2:
                continue
            # Regex check for YYYY, YYYY-MM-DD, DD/MM/YYYY, ISO timestamps, Q1-2024, 2024Q1
            if re.match(r"^\d{4}$", s_val) and 1800 <= int(s_val) <= 2150:
                parsed_count += 1
            elif re.match(r"^Q[1-4][-_\s]?\d{2,4}", s_val, re.I) or re.match(r"^\d{2,4}[-_\s]?Q[1-4]", s_val, re.I):
                parsed_count += 1
            elif re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", s_val) or re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", s_val):
                parsed_count += 1
            else:
                try:
                    pd.to_datetime(s_val)
                    parsed_count += 1
                except (ValueError, TypeError, OverflowError):
                    pass

        return parsed_count >= len(sample) * 0.70

    def _is_identifier(
        self,
        col_name: str,
        series: pd.Series,
        n_rows: int,
        unique_ratio: float,
        is_numeric: bool,
    ) -> bool:
        """Detect identifiers while preventing standard continuous numeric metrics from misclassification."""
        norm_name = _normalize(col_name)

        # 1. Token hints
        id_suffixes = ("_id", "_key", "_code", "_sku", "_uuid", "_guid", "_ref", "_number", "_no")
        id_exact = ("id", "key", "code", "sku", "uuid", "guid", "ref", "ssn", "isbn")

        has_id_name = norm_name in id_exact or any(norm_name.endswith(sfx) for sfx in id_suffixes)

        # 2. If name clearly says ID and uniqueness is high
        if has_id_name and (unique_ratio > 0.50 or n_rows < 10):
            return True

        # 3. String columns with 100% uniqueness and alphanumeric format
        if not is_numeric and unique_ratio > 0.98 and n_rows >= 10:
            if any(t in norm_name for t in ("id", "code", "account", "user", "order", "item", "trans")):
                return True

        return False

    def _extract_data_quality(self, df: pd.DataFrame, profile_dict: Dict[str, Any]) -> DataQuality:
        """Build a DataQuality model from DataFrame and profiler results."""
        missing_dict = {col: int(df[col].isna().sum()) for col in df.columns}
        n_dups = int(df.duplicated().sum())

        prof = profile_dict.get("profile", {})
        q_score = float(prof.get("quality_score", 100.0))

        warnings: List[str] = []
        if n_dups > 0:
            warnings.append(f"Detected {n_dups} duplicate rows in dataset.")
        total_missing = sum(missing_dict.values())
        if total_missing > 0:
            warnings.append(f"Detected {total_missing} missing values across columns.")

        return DataQuality(
            missing_values=missing_dict,
            duplicates=n_dups,
            duplicate_rows=n_dups,
            outliers={},
            invalid_values={},
            warnings=warnings,
            quality_score=q_score,
        )

    def _infer_dataset_type(
        self,
        date_cols: List[SemanticMapping],
        metrics: List[SemanticMapping],
        all_mappings: List[SemanticMapping],
    ) -> str:
        """Classify dataset domain archetype."""
        has_dates = len(date_cols) > 0
        concepts = {m.semantic_concept for m in all_mappings}

        if {"revenue", "profit", "cost"}.intersection(concepts):
            return "financial"
        if {"customer", "churn", "tenure"}.intersection(concepts):
            return "customer"
        if has_dates and len(metrics) > 0:
            return "time_series"
        if {"transaction_id", "order_id"}.intersection(concepts) or any("transaction" in m.column_name.lower() for m in all_mappings):
            return "transactional"
        return "tabular"

    def _clean_sample_val(self, val: Any) -> Any:
        """Convert numpy/pandas scalar values to JSON-serializable types."""
        if pd.isna(val):
            return None
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, (np.floating, float)):
            return round(float(val), 4)
        if isinstance(val, (pd.Timestamp, datetime)):
            return val.isoformat()
        return str(val)
