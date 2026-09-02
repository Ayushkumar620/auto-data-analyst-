"""
Filter Engine - Autonomous analytical filter parsing, compilation, and execution engine.
Parses natural-language filters (numeric, categorical, compound AND/OR, and temporal/dates),
grouping dimensions (single and multi-column combinations), rankings, extreme values,
and secondary aggregations. Applies safe vectorized boolean indexing, compiles to a
structured AST, and executes aggregations and dimensional statistics.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from backend.app.core.temporal import TemporalIntelligenceEngine

logger = logging.getLogger(__name__)

# Reserved logical modifiers that must NEVER be parsed as categorical values
LOGICAL_MODIFIERS = {
    "either", "neither", "any", "one", "both", "all", "can", "be", "of", "an",
    "either of", "one of", "any of",
}

# Pure grammatical function words that can never be categorical values
GRAMMAR_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "these", "those",
    "this", "that", "where", "with", "from", "is", "are", "be", "by", "do", "not",
}

# Analytical instruction keywords that must NEVER be accepted as categorical values
ANALYTICAL_INSTRUCTION_KEYWORDS = {
    "show", "calculate", "compute", "rank", "identify", "break down", "average", "total",
    "highest", "lowest", "combination", "combinations", "every", "by", "for each", "ranking",
    "sorting", "percent", "percentage", "records", "rows", "sales", "units", "highest to lowest",
    "lowest to highest", "asc", "desc", "ascending", "descending", "combinations by",
}


@dataclass
class FilterCondition:
    """Represents a single atomic filter comparison condition on a column."""
    column: str
    operator: str  # '<=', '>=', '<', '>', '==', '!=', 'in', 'not in'
    value: Any
    raw_expression: str = ""
    is_date: bool = False

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate condition against DataFrame and return boolean mask."""
        if self.column not in df.columns:
            matching_cols = [c for c in df.columns if c.lower() == self.column.lower()]
            if matching_cols:
                col_name = matching_cols[0]
            else:
                return pd.Series(True, index=df.index)
        else:
            col_name = self.column

        series = df[col_name]

        # 1. Date / Temporal comparisons
        if self.is_date or self._looks_like_date(self.value):
            dt_series = pd.to_datetime(series, errors="coerce")
            try:
                val_dt = pd.to_datetime(self.value)
                if not pd.isna(val_dt):
                    v_date = val_dt.date()
                    if self.operator == "==":
                        return dt_series.dt.date == v_date
                    elif self.operator == "!=":
                        return dt_series.dt.date != v_date
                    elif self.operator == "<=":
                        return dt_series.dt.date <= v_date
                    elif self.operator == ">=":
                        return dt_series.dt.date >= v_date
                    elif self.operator == "<":
                        return dt_series.dt.date < v_date
                    elif self.operator == ">":
                        return dt_series.dt.date > v_date
            except Exception:
                pass

        # 2. Numeric comparisons
        if self.operator in ("<=", ">=", "<", ">", "==", "!="):
            num_series = pd.to_numeric(series, errors="coerce")
            if isinstance(self.value, (int, float)) or (isinstance(self.value, str) and self._is_numeric_str(self.value)):
                val = float(self.value)
                if self.operator == "<=":
                    return num_series <= val
                elif self.operator == ">=":
                    return num_series >= val
                elif self.operator == "<":
                    return num_series < val
                elif self.operator == ">":
                    return num_series > val
                elif self.operator == "==":
                    return num_series == val
                elif self.operator == "!=":
                    return num_series != val

            # String comparisons
            str_series = series.astype(str).str.strip().str.lower()
            val_str = str(self.value).strip().lower()
            if self.operator == "==":
                return str_series == val_str
            elif self.operator == "!=":
                return str_series != val_str

        # 3. Categorical 'in' / 'not in'
        if self.operator == "in":
            if isinstance(self.value, (list, tuple, set)):
                val_set = {str(v).strip().lower() for v in self.value}
                return series.astype(str).str.strip().str.lower().isin(val_set)
            else:
                val_str = str(self.value).strip().lower()
                return series.astype(str).str.strip().str.lower() == val_str
        elif self.operator == "not in":
            if isinstance(self.value, (list, tuple, set)):
                val_set = {str(v).strip().lower() for v in self.value}
                return ~series.astype(str).str.strip().str.lower().isin(val_set)
            else:
                val_str = str(self.value).strip().lower()
                return series.astype(str).str.strip().str.lower() != val_str

        return pd.Series(True, index=df.index)

    def to_ast(self) -> Dict[str, Any]:
        """Convert condition to abstract syntax tree representation."""
        return {
            "type": "comparison",
            "column": self.column,
            "operator": self.operator,
            "value": self.value,
            "is_date": self.is_date,
            "expression": self.to_expression(),
        }

    def to_expression(self) -> str:
        """Render condition as human-readable boolean expression."""
        if self.raw_expression:
            return self.raw_expression
        if self.operator == "in" and isinstance(self.value, (list, tuple, set)):
            val_strs = [f"'{v}'" if isinstance(v, str) else str(v) for v in self.value]
            return f"({ ' OR '.join([f'{self.column} == {v}' for v in val_strs]) })"
        val_str = f"'{self.value}'" if isinstance(self.value, str) and not self.is_date else str(self.value)
        return f"{self.column} {self.operator} {val_str}"

    @staticmethod
    def _is_numeric_str(val: str) -> bool:
        try:
            float(val)
            return True
        except ValueError:
            return False

    @staticmethod
    def _looks_like_date(val: Any) -> bool:
        if not isinstance(val, str):
            return False
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", val.strip()) or re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", val.strip()))


@dataclass
class CompoundFilter:
    """Represents a compound logical filter AST node (AND / OR)."""
    conditions: List[Union[FilterCondition, 'CompoundFilter']] = field(default_factory=list)
    logical_op: str = "AND"  # 'AND' or 'OR'

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if not self.conditions:
            return pd.Series(True, index=df.index)

        mask = None
        for cond in self.conditions:
            curr_mask = cond.evaluate(df)
            if mask is None:
                mask = curr_mask
            else:
                if self.logical_op.upper() == "OR":
                    mask = mask | curr_mask
                else:
                    mask = mask & curr_mask

        return mask if mask is not None else pd.Series(True, index=df.index)

    def to_ast(self) -> Dict[str, Any]:
        """Convert compound filter to abstract syntax tree representation."""
        return {
            "type": "compound",
            "logical_op": self.logical_op,
            "conditions": [c.to_ast() for c in self.conditions],
        }

    def to_expression(self) -> str:
        """Render compound filter as boolean expression respecting precedence."""
        if not self.conditions:
            return "All Records"
        parts = [c.to_expression() for c in self.conditions]
        sep = f" {self.logical_op.upper()} "
        expr = sep.join(parts)
        if self.logical_op.upper() == "OR" and len(self.conditions) > 1:
            return f"({expr})"
        return expr


@dataclass
class AggregationRequest:
    """Represents an aggregation request on a specific column."""
    column: str
    function: str  # 'sum', 'mean', 'count', 'max', 'min'
    display_name: str = ""


@dataclass
class BreakdownRequest:
    """Represents a grouped breakdown request on a categorical dimension."""
    dimension: str
    metrics: List[Tuple[str, str]] = field(default_factory=list)  # [(column, function)]
    find_highest: bool = False
    find_lowest: bool = False
    extreme_metric: str = ""


@dataclass
class AnalyticalQueryPlan:
    """Structured analytical plan separating filter, group_by, aggregations, ranking, and extremes."""
    filter: Optional[Union[FilterCondition, CompoundFilter]] = None
    group_by: List[str] = field(default_factory=list)
    aggregations: List[AggregationRequest] = field(default_factory=list)
    ranking: Optional[Dict[str, str]] = None  # {"column": "sales", "direction": "desc"}
    extremes: List[str] = field(default_factory=list)  # ["highest", "lowest"]
    secondary_analysis: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filter": self.filter.to_ast() if self.filter else None,
            "group_by": self.group_by,
            "aggregations": [{"column": a.column, "function": a.function} for a in self.aggregations],
            "ranking": self.ranking,
            "extremes": self.extremes,
            "secondary_analysis": self.secondary_analysis,
        }


@dataclass
class FilterExecutionResult:
    """Structured analytical result of executing filter + aggregations + breakdowns."""
    filter_description: str
    matching_rows: int
    total_rows: int
    aggregations: Dict[str, Dict[str, Any]]
    filtered_df: pd.DataFrame
    columns: List[str]
    markdown_response: str
    breakdowns: Dict[str, Any] = field(default_factory=dict)
    filter_ast: Dict[str, Any] = field(default_factory=dict)
    highest_record: Optional[Dict[str, Any]] = None
    lowest_record: Optional[Dict[str, Any]] = None
    query_plan: Optional[Dict[str, Any]] = None
    group_by: List[str] = field(default_factory=list)
    grouped_records: Optional[List[Dict[str, Any]]] = None
    secondary_results: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filter_description": self.filter_description,
            "matching_rows": self.matching_rows,
            "total_rows": self.total_rows,
            "aggregations": self.aggregations,
            "columns": self.columns,
            "markdown_response": self.markdown_response,
            "breakdowns": self.breakdowns,
            "filter_ast": self.filter_ast,
            "highest_record": self.highest_record,
            "lowest_record": self.lowest_record,
            "query_plan": self.query_plan,
            "group_by": self.group_by,
            "grouped_records": self.grouped_records,
            "secondary_results": self.secondary_results,
        }


class FilterEngine:
    """
    Parser and execution engine for natural language filtering, compound expressions,
    date/temporal filters, aggregations, group combinations, rankings, and secondary statistics.
    """

    DATE_TOKEN_REGEX = r"(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})"

    # Comparison operator patterns for numbers
    NUMERIC_PATTERNS = [
        (re.compile(r"(?:is\s+)?(?:less\s+than\s+or\s+equal\s+to|at\s+most|no\s+more\s+than|<=)\s*([0-9]+(?:\.[0-9]+)?)\%?", re.I), "<="),
        (re.compile(r"(?:is\s+)?(?:greater\s+than\s+or\s+equal\s+to|at\s+least|no\s+less\s+than|>=)\s*([0-9]+(?:\.[0-9]+)?)\%?", re.I), ">="),
        (re.compile(r"(?:is\s+)?([0-9]+(?:\.[0-9]+)?)\%?\s+or\s+less\b", re.I), "<="),
        (re.compile(r"(?:is\s+)?([0-9]+(?:\.[0-9]+)?)\%?\s+or\s+fewer\b", re.I), "<="),
        (re.compile(r"(?:is\s+)?([0-9]+(?:\.[0-9]+)?)\%?\s+or\s+more\b", re.I), ">="),
        (re.compile(r"(?:is\s+)?([0-9]+(?:\.[0-9]+)?)\%?\s+or\s+greater\b", re.I), ">="),
        (re.compile(r"(?:is\s+)?(?:strictly\s+)?(?:less\s+than|below|under|<)\s*([0-9]+(?:\.[0-9]+)?)\%?", re.I), "<"),
        (re.compile(r"(?:is\s+)?(?:strictly\s+)?(?:greater\s+than|above|over|more\s+than|exceeding|>)\s*([0-9]+(?:\.[0-9]+)?)\%?", re.I), ">"),
        (re.compile(r"(?:is\s+)?(?:not\s+equal\s+to|not\s+equals?|!=)\s*([0-9]+(?:\.[0-9]+)?)\%?", re.I), "!="),
        (re.compile(r"(?:is\s+)?(?:equal\s+to|equals?|==|=)\s*([0-9]+(?:\.[0-9]+)?)\%?", re.I), "=="),
    ]

    # Filter indicator keywords - strictly discriminates genuine filters from grouping/ranking
    FILTER_INDICATORS = [
        re.compile(r"\bwhere\b", re.I),
        re.compile(r"\bfilter(?:ed)?\b", re.I),
        re.compile(r"\bonly\s+(?:the\s+)?records\b", re.I),
        re.compile(r"\bonly\s+(?:the\s+)?rows\b", re.I),
        re.compile(r"\bwhich\s+have\b", re.I),
        re.compile(r"\bhaving\b", re.I),
        re.compile(r"\bwith\s+[a-z0-9_]+\s*(?:<=|>=|<|>|=|!=|is\b|in\b)", re.I),
        re.compile(r"[a-z0-9_]+\s*(?:<=|>=|<|>|!=)\s*[0-9]+", re.I),
        re.compile(r"[a-z0-9_]+\s+(?:is\s+)?[0-9]+\s+or\s+less", re.I),
        re.compile(r"[a-z0-9_]+\s+(?:is\s+)?[0-9]+\s+or\s+more", re.I),
        re.compile(r"[a-z0-9_]+\s+(?:is\s+)?[0-9]+\s+or\s+fewer", re.I),
        re.compile(r"[a-z0-9_]+\s+(?:is\s+)?at\s+most\s+[0-9]+", re.I),
        re.compile(r"[a-z0-9_]+\s+(?:is\s+)?at\s+least\s+[0-9]+", re.I),
        re.compile(r"\beither\b.+\bor\b", re.I),
        re.compile(r"\b(?:is|in|equals?|can\s+be|one\s+of)\s+(?:either\s+)?[a-zA-Z0-9_'\"]+\s+or\s+[a-zA-Z0-9_'\"]+", re.I),
        # Date filter indicators - require an actual date token following preposition
        re.compile(r"\bfrom\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|\d{4}-\d{2}-\d{2})\b", re.I),
        re.compile(r"\b(?:on|between|after|before|through|until|since)\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|\d{4}-\d{2}-\d{2})\b", re.I),
        re.compile(r"\b\d{4}-\d{2}-\d{2}\b", re.I),
        re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b", re.I),
        re.compile(r"\b(?:records|sales|data)\s+(?:from|between|on|after|before)\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b", re.I),
    ]

    # Explicit preview indicators
    LEGITIMATE_PREVIEW_INDICATORS = [
        re.compile(r"\b(?:show|view|display|give)\s+(?:me\s+)?(?:the\s+)?first\s+\d+\s*(?:rows|records|items)?\b", re.I),
        re.compile(r"\b(?:show|view|display|give)\s+(?:me\s+)?(?:the\s+)?top\s+\d+\s*(?:rows|records|items)?\b", re.I),
        re.compile(r"\b(?:show|view|display|give)\s+(?:me\s+)?(?:a\s+)?(?:sample|preview)\b", re.I),
        re.compile(r"^head(?:\s+\d+)?$", re.I),
        re.compile(r"^preview(?:\s+data(?:set)?)?$", re.I),
    ]

    @classmethod
    def has_filter_intent(cls, query: str) -> bool:
        """Return True if query contains any analytical filtering intent."""
        q = query.strip()
        return any(pat.search(q) for pat in cls.FILTER_INDICATORS)

    @classmethod
    def is_legitimate_preview_request(cls, query: str) -> bool:
        """
        Return True ONLY if query is a genuine request for dataset preview / head,
        and does NOT contain analytical filtering or aggregation conditions.
        """
        if cls.has_filter_intent(query):
            return False

        q_lower = query.lower().strip()
        if any(w in q_lower for w in ("calculate", "total", "sum", "average", "mean", "min", "max", "count", "group by", "forecast", "predict")):
            return False

        return any(pat.search(query) for pat in cls.LEGITIMATE_PREVIEW_INDICATORS)

    @classmethod
    def detect_date_column(
        cls,
        df: Optional[pd.DataFrame] = None,
        query: str = "",
        columns: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect the appropriate temporal column from DataFrame or column list.
        Returns (resolved_column_name, error_or_clarification_message).
        """
        col_list = list(df.columns) if df is not None else (columns or [])
        if not col_list:
            return None, "No columns available in the dataset to filter by date."

        detected_temporal_cols: List[str] = []

        # 1. Use TemporalIntelligenceEngine if DataFrame is provided
        if df is not None and not df.empty:
            try:
                engine = TemporalIntelligenceEngine()
                fields = engine.detect_fields(df)
                if fields:
                    detected_temporal_cols = [f["column"] for f in fields]
            except Exception:
                pass

        # 2. Fallback: identify candidate temporal columns by name/type
        if not detected_temporal_cols:
            for c in col_list:
                c_low = c.lower()
                if df is not None and c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c]):
                    detected_temporal_cols.append(c)
                elif any(kw in c_low for kw in ("date", "time", "timestamp", "datetime", "created_at", "updated_at", "order_date", "sale_date", "transaction_date")):
                    detected_temporal_cols.append(c)

        if not detected_temporal_cols:
            return None, f"I detected a date filter in your request, but could not find a date or timestamp column in the current dataset. Available columns: {', '.join(col_list)}."

        # 3. Check if user explicitly mentioned one of the detected temporal columns in query
        mentioned = []
        for tc in detected_temporal_cols:
            if re.search(rf"\b{re.escape(tc)}\b", query, re.I):
                mentioned.append(tc)

        if len(mentioned) == 1:
            return mentioned[0], None
        elif len(mentioned) > 1:
            return None, f"I detected a date filter, and multiple date columns ({', '.join(mentioned)}) are mentioned in your query. Please clarify which date column to filter on."

        # 4. If none explicitly mentioned, but exactly 1 temporal column exists
        if len(detected_temporal_cols) == 1:
            return detected_temporal_cols[0], None

        # 5. Multiple temporal columns exist and query did not specify which one
        return None, f"I detected a date filter, but the dataset contains multiple date columns ({', '.join(detected_temporal_cols)}). Please specify which date column to filter on."

    @classmethod
    def _parse_iso_date(cls, val_str: str, default_year: Optional[Union[str, int]] = None) -> Optional[str]:
        """Convert a date string (e.g. 'January 3, 2025' or 'January 3') to ISO YYYY-MM-DD."""
        s = val_str.strip().rstrip(".,")
        m_no_yr = re.match(r"^([a-zA-Z]+)\s+(\d{1,2})(?:st|nd|rd|th)?$", s)
        if m_no_yr and default_year:
            s = f"{m_no_yr.group(1)} {m_no_yr.group(2)}, {default_year}"
        try:
            dt = pd.to_datetime(s, errors="coerce")
            if pd.isna(dt):
                return None
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    @classmethod
    def parse_date_filter(
        cls,
        query: str,
        columns: Optional[List[str]] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> Tuple[Optional[Union[FilterCondition, CompoundFilter]], Optional[str], Optional[str]]:
        """
        Extract date conditions from natural language query.
        Returns (filter_node, date_column, error_message).
        """
        token = cls.DATE_TOKEN_REGEX

        # 1. Range: between <d1> and <d2>
        m_between = re.search(rf"\bbetween\s+({token})\s+and\s+({token})\b", query, re.I)
        if m_between:
            d1_raw, d2_raw = m_between.group(1), m_between.group(2)
            y2 = re.search(r"\b(20\d\d|19\d\d)\b", d2_raw)
            def_y = y2.group(1) if y2 else None
            iso1 = cls._parse_iso_date(d1_raw, def_y)
            iso2 = cls._parse_iso_date(d2_raw)
            if iso1 and iso2:
                col_name, err = cls.detect_date_column(dataframe, query, columns)
                if err:
                    return None, None, err
                cond_start = FilterCondition(column=col_name, operator=">=", value=iso1, is_date=True, raw_expression=f"{col_name} >= {iso1}")
                cond_end = FilterCondition(column=col_name, operator="<=", value=iso2, is_date=True, raw_expression=f"{col_name} <= {iso2}")
                return CompoundFilter(conditions=[cond_start, cond_end], logical_op="AND"), col_name, None

        # 2. Range: from <d1> (through|to|until|–|-) <d2> (inclusive)?
        m_range = re.search(rf"\b(?:from\s+)?({token})\s*(?:through|to|until|thru|–|-)\s*({token})(?:\s+inclusive)?\b", query, re.I)
        if m_range:
            d1_raw, d2_raw = m_range.group(1), m_range.group(2)
            y2 = re.search(r"\b(20\d\d|19\d\d)\b", d2_raw)
            def_y = y2.group(1) if y2 else None
            iso1 = cls._parse_iso_date(d1_raw, def_y)
            iso2 = cls._parse_iso_date(d2_raw)
            if iso1 and iso2:
                col_name, err = cls.detect_date_column(dataframe, query, columns)
                if err:
                    return None, None, err
                cond_start = FilterCondition(column=col_name, operator=">=", value=iso1, is_date=True, raw_expression=f"{col_name} >= {iso1}")
                cond_end = FilterCondition(column=col_name, operator="<=", value=iso2, is_date=True, raw_expression=f"{col_name} <= {iso2}")
                return CompoundFilter(conditions=[cond_start, cond_end], logical_op="AND"), col_name, None

        # 3. Single date: on or after / after / since
        m_after = re.search(rf"\b(?:on\s+or\s+after|after|since)\s+({token})\b", query, re.I)
        if m_after:
            iso = cls._parse_iso_date(m_after.group(1))
            if iso:
                col_name, err = cls.detect_date_column(dataframe, query, columns)
                if err:
                    return None, None, err
                return FilterCondition(column=col_name, operator=">=", value=iso, is_date=True, raw_expression=f"{col_name} >= {iso}"), col_name, None

        # 4. Single date: on or before / before / until
        m_before = re.search(rf"\b(?:on\s+or\s+before|before|until|prior\s+to)\s+({token})\b", query, re.I)
        if m_before:
            iso = cls._parse_iso_date(m_before.group(1))
            if iso:
                col_name, err = cls.detect_date_column(dataframe, query, columns)
                if err:
                    return None, None, err
                return FilterCondition(column=col_name, operator="<=", value=iso, is_date=True, raw_expression=f"{col_name} <= {iso}"), col_name, None

        # 5. Single date: on / from / for / at <date>
        m_on = re.search(rf"\b(?:on|from|for|at|records\s+from)\s+({token})\b", query, re.I)
        if m_on:
            iso = cls._parse_iso_date(m_on.group(1))
            if iso:
                col_name, err = cls.detect_date_column(dataframe, query, columns)
                if err:
                    return None, None, err
                return FilterCondition(column=col_name, operator="==", value=iso, is_date=True, raw_expression=f"{col_name} == {iso}"), col_name, None

        # 6. Fallback: lone date mention
        m_lone = re.search(rf"\b({token})\b", query, re.I)
        if m_lone:
            iso = cls._parse_iso_date(m_lone.group(1))
            if iso:
                col_name, err = cls.detect_date_column(dataframe, query, columns)
                if err:
                    return None, None, err
                return FilterCondition(column=col_name, operator="==", value=iso, is_date=True, raw_expression=f"{col_name} == {iso}"), col_name, None

        # Check if user mentioned an invalid date
        m_inv = re.search(r"\b(?:from|on|between|before|after)\s+([A-Za-z]+ \d{1,2}(?:, \d{4})?)\b", query, re.I)
        if m_inv:
            raw_cand = m_inv.group(1)
            if not cls._parse_iso_date(raw_cand):
                return None, None, f"I detected a date filter, but could not parse a valid date from: '{raw_cand}'."

        return None, None, None

    @classmethod
    def _extract_filter_clause(cls, query: str) -> str:
        """Extract the specific substring containing filter conditions respecting sentence boundaries."""
        where_match = re.search(r"\b(?:where|filter(?:ed)?(?:\s+by|\s+on)?|having|with)\s+([^.;!\n]+)", query, re.I)
        if where_match:
            clause = where_match.group(1).strip()
            end_match = re.search(
                r"\b(?:calculate|compute|aggregate|break\s+down|showing|and\s+calculate|then\s+calculate|do\s+not\s+include|rank|group\s+by)\b",
                clause,
                re.I,
            )
            if end_match:
                clause = clause[:end_match.start()].strip()
            return clause
        return query

    @classmethod
    def parse_filters(
        cls,
        query: str,
        columns: Optional[List[str]] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> Optional[CompoundFilter]:
        """
        Extract structured filter AST from a natural language query against dataset columns.
        Returns None if no explicit filter condition is present.
        """
        if not cls.has_filter_intent(query):
            return None

        col_map = {c.lower(): c for c in columns} if columns else {}
        if dataframe is not None and not col_map:
            col_map = {c.lower(): c for c in dataframe.columns}
            columns = list(dataframe.columns)

        conditions: List[Union[FilterCondition, CompoundFilter]] = []

        # 1. Parse Date Conditions
        date_cond, date_col, date_err = cls.parse_date_filter(query, columns=columns, dataframe=dataframe)
        if date_cond:
            if isinstance(date_cond, CompoundFilter):
                conditions.extend(date_cond.conditions)
            else:
                conditions.append(date_cond)

        # 2. Parse Numeric & Categorical Conditions from filter clause
        clause_text = cls._extract_filter_clause(query)
        parts = cls._split_conjunctions(clause_text)

        for part in parts:
            part_cleaned = part.strip().strip("()")
            # Skip if this part was only the date expression already parsed
            if date_cond and (
                "january" in part_cleaned.lower() or "february" in part_cleaned.lower()
                or re.search(r"\b\d{4}-\d{2}-\d{2}\b", part_cleaned)
            ):
                if not any(kw in part_cleaned.lower() for kw in ("<", ">", "=", "discount", "sales", "region", "product")):
                    continue

            atomic_num = cls._parse_numeric_condition(part_cleaned, col_map, columns)
            if atomic_num:
                conditions.append(atomic_num)
                continue

            cat_or = cls._parse_categorical_condition(part_cleaned, col_map, columns, dataframe)
            if cat_or:
                conditions.append(cat_or)
                continue

            if re.search(r"\bor\b", part_cleaned, re.I):
                or_cond = cls._parse_compound_or_condition(part_cleaned, col_map, columns, dataframe)
                if or_cond:
                    conditions.append(or_cond)
                    continue

        if not conditions:
            # Fallback: scan whole query text for explicit numeric comparison patterns with column names
            for op_regex, op_symbol in cls.NUMERIC_PATTERNS:
                pat = re.compile(rf"\b([a-zA-Z_][a-zA-Z0-9_]*)\b\s*{op_regex.pattern}", re.I)
                for m in pat.finditer(query):
                    col_name = m.group(1)
                    if columns and col_name.lower() not in col_map:
                        continue
                    val_num = float(m.group(2))
                    if val_num.is_integer():
                        val_num = int(val_num)
                    resolved_col = col_map.get(col_name.lower(), col_name)
                    conditions.append(FilterCondition(
                        column=resolved_col,
                        operator=op_symbol,
                        value=val_num,
                        raw_expression=f"{resolved_col} {op_symbol} {val_num}",
                    ))

        if not conditions:
            return None

        compound = CompoundFilter(conditions=conditions, logical_op="AND")
        return compound

    @classmethod
    def _split_conjunctions(cls, text: str) -> List[str]:
        """Split text by 'AND' while preserving parentheses groups."""
        tokens = []
        current = []
        depth = 0
        for word in text.split():
            if "(" in word:
                depth += word.count("(")
            if ")" in word:
                depth -= word.count(")")
            if word.lower() == "and" and depth == 0:
                tokens.append(" ".join(current))
                current = []
            else:
                current.append(word)
        if current:
            tokens.append(" ".join(current))
        return [t.strip() for t in tokens if t.strip()]

    @classmethod
    def _parse_numeric_condition(
        cls,
        text: str,
        col_map: Dict[str, str],
        columns: Optional[List[str]],
    ) -> Optional[FilterCondition]:
        """Parse atomic numeric comparison (e.g. 'the discount is 5% or less' or 'sales > 10000')."""
        clean_text = re.sub(r"^(?:the|a|an|where|that)\s+", "", text.strip(), flags=re.I).strip("()")

        for op_regex, op_symbol in cls.NUMERIC_PATTERNS:
            pat = re.compile(rf"\b([a-zA-Z_][a-zA-Z0-9_]*)\b\s*{op_regex.pattern}", re.I)
            m = pat.search(clean_text)
            if m:
                col_name = m.group(1)
                if columns and col_name.lower() not in col_map:
                    continue
                val_num = float(m.group(2))
                if val_num.is_integer():
                    val_num = int(val_num)
                resolved_col = col_map.get(col_name.lower(), col_name)
                return FilterCondition(
                    column=resolved_col,
                    operator=op_symbol,
                    value=val_num,
                    raw_expression=f"{resolved_col} {op_symbol} {val_num}",
                )
        return None

    @classmethod
    def _parse_categorical_condition(
        cls,
        text: str,
        col_map: Dict[str, str],
        columns: Optional[List[str]],
        dataframe: Optional[pd.DataFrame] = None,
    ) -> Optional[FilterCondition]:
        """Parse natural-language categorical conditions (e.g. 'region is either North or West')."""
        clean_text = re.sub(r"^(?:the|a|an|where|that)\s+", "", text.strip(), flags=re.I).strip("()[]")

        col_name = None
        rest = None

        # Connective is mandatory: 'is', 'can be', 'equals', '==', 'in', 'one of', '!='
        if col_map:
            for c_low, c_orig in col_map.items():
                pat = re.compile(
                    rf"^\b{re.escape(c_low)}\b\s+(?:is|can\s+be|equals?|==?|!=|in|one\s+of)\s+(.+)",
                    re.I,
                )
                m = pat.match(clean_text)
                if m:
                    col_name = c_orig
                    rest = m.group(1).strip()
                    break

        if not col_name:
            gen_pat = re.compile(
                r"^([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:is|can\s+be|equals?|==?|!=|in|one\s+of)\s+(.+)",
                re.I,
            )
            m = gen_pat.match(clean_text)
            if m:
                potential_col = m.group(1)
                if not columns or potential_col.lower() in col_map:
                    col_name = col_map.get(potential_col.lower(), potential_col)
                    rest = m.group(2).strip()

        if not col_name or not rest:
            return None

        # Strip logical prefixes
        rest = re.sub(r"^(?:either|one\s+of|any\s+of|either\s+of)\s+", "", rest, flags=re.I).strip("()[]")

        # Reject if rest contains sentence punctuation indicating analytical prose
        if any(punct in rest for punct in (".", ";", "!")):
            return None

        raw_vals = [
            v.strip().strip("'\"()")
            for v in re.split(r",?\s+\bor\b\s*|,\s*", rest, flags=re.I)
            if v.strip()
        ]

        actual_uniques = {}
        if dataframe is not None and col_name in dataframe.columns:
            actual_uniques = {str(x).strip().lower(): str(x) for x in dataframe[col_name].dropna().unique()}

        clean_vals = []
        for v in raw_vals:
            v_low = v.lower()
            if v_low in LOGICAL_MODIFIERS:
                continue

            # Check actual dataset uniques first!
            if actual_uniques:
                if v_low in actual_uniques:
                    clean_vals.append(actual_uniques[v_low])
                continue

            # Fallback if dataframe not provided:
            v_tokens = set(v_low.split())
            if v_tokens.intersection(ANALYTICAL_INSTRUCTION_KEYWORDS):
                continue
            if len(v_tokens) > 3:
                continue
            if v_low not in GRAMMAR_STOPWORDS:
                clean_vals.append(v)

        if not clean_vals:
            return None

        for v in clean_vals:
            if v.lower() in LOGICAL_MODIFIERS:
                return None

        if len(clean_vals) > 1:
            expr_str = f"({ ' OR '.join([f'{col_name} == {repr(v)}' for v in clean_vals]) })"
            return FilterCondition(
                column=col_name,
                operator="in",
                value=clean_vals,
                raw_expression=expr_str,
            )
        else:
            return FilterCondition(
                column=col_name,
                operator="==",
                value=clean_vals[0],
                raw_expression=f"{col_name} == '{clean_vals[0]}'",
            )

    @classmethod
    def _parse_compound_or_condition(
        cls,
        text: str,
        col_map: Dict[str, str],
        columns: Optional[List[str]],
        dataframe: Optional[pd.DataFrame] = None,
    ) -> Optional[CompoundFilter]:
        """Parse multiple distinct expressions connected by OR."""
        sub_exprs = re.split(r"\s+\bor\b\s+", text, flags=re.I)
        sub_conds = []
        for se in sub_exprs:
            clean_se = se.strip().strip("()")
            cond = cls._parse_numeric_condition(clean_se, col_map, columns)
            if not cond:
                cond = cls._parse_categorical_condition(clean_se, col_map, columns, dataframe)
            if cond:
                sub_conds.append(cond)

        if sub_conds:
            if len(sub_conds) == 1:
                return sub_conds[0]
            return CompoundFilter(conditions=sub_conds, logical_op="OR")
        return None

    @classmethod
    def parse_group_by(cls, query: str, columns: Optional[List[str]] = None) -> List[str]:
        """
        Extract grouping dimensions from query (supports multi-dimension combinations and single dimensions).
        """
        col_map = {c.lower(): c for c in columns} if columns else {}
        if not col_map:
            return []

        # 1. Multi-dimension: 'by <col1> and <col2>' or 'by <col1>, <col2>'
        m_by_multi = re.search(r"\bby\s+([a-zA-Z0-9_]+)(?:\s+and\s+|,\s*)([a-zA-Z0-9_]+)\b", query, re.I)
        if m_by_multi:
            c1, c2 = m_by_multi.group(1).lower(), m_by_multi.group(2).lower()
            if c1 in col_map and c2 in col_map:
                return [col_map[c1], col_map[c2]]

        # 2. Multi-dimension combination: 'every <col1>-<col2> combination' or '<col1>-<col2> combinations'
        m_comb = re.search(r"\b([a-zA-Z0-9_]+)[-–/]([a-zA-Z0-9_]+)\s+combinations?\b", query, re.I)
        if m_comb:
            c1, c2 = m_comb.group(1).lower(), m_comb.group(2).lower()
            if c1 in col_map and c2 in col_map:
                return [col_map[c1], col_map[c2]]

        # 3. Single dimension: 'by <col>'
        m_by_single = re.search(r"\bby\s+([a-zA-Z0-9_]+)\b", query, re.I)
        if m_by_single:
            c = m_by_single.group(1).lower()
            if c in col_map and c not in ("day", "date", "time", "month", "year", "total", "sales", "units"):
                return [col_map[c]]

        # 4. Group by 'for each <col>' if not followed by a comparison
        m_for_each = re.search(r"\bfor\s+each\s+([a-zA-Z0-9_]+)\b", query, re.I)
        if m_for_each:
            c = m_for_each.group(1).lower()
            if c in col_map:
                return [col_map[c]]

        return []

    @classmethod
    def parse_query_plan(
        cls,
        query: str,
        columns: Optional[List[str]] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> AnalyticalQueryPlan:
        """
        Construct a structured AnalyticalQueryPlan separating FILTER, GROUP_BY,
        AGGREGATIONS, RANKING, EXTREMES, and SECONDARY_ANALYSIS.
        """
        cols = list(dataframe.columns) if dataframe is not None else (columns or [])
        col_map = {c.lower(): c for c in cols}

        # 1. Group By detection
        group_by = cls.parse_group_by(query, columns=cols)

        # 2. Filter parsing (Step 3: If group_by exists and query contains no explicit filter words, filter = None)
        has_explicit_filter = bool(re.search(r"\b(?:where|filter(?:ed)?(?:\s+by|\s+on)?|having|with\s+[a-z0-9_]+\s*(?:<=|>=|<|>|=|!=|is\b|in\b)|only\s+(?:the\s+)?records?\s+where)\b", query, re.I))
        if group_by and not has_explicit_filter:
            filter_node = None
        else:
            filter_node = cls.parse_filters(query, columns=cols, dataframe=dataframe)

        # 3. Aggregations
        aggs = cls.parse_aggregations(query, columns=cols)

        # 4. Ranking
        ranking = None
        if re.search(r"\b(rank|sort|order)\b", query, re.I):
            direction = "desc"
            if re.search(r"\b(?:lowest\s+to\s+highest|ascending|from\s+lowest|bottom)\b", query, re.I):
                direction = "asc"
            m_col = None
            for c in ("sales", "units", "amount", "revenue", "salary", "profit", "cost"):
                if c in col_map and c in query.lower():
                    m_col = col_map[c]
                    break
            if not m_col and aggs:
                m_col = aggs[0].column
            if m_col:
                ranking = {"column": m_col, "direction": direction}

        # 5. Extremes
        extremes = []
        if re.search(r"\b(?:highest|max|maximum|top)\b", query, re.I):
            extremes.append("highest")
        if re.search(r"\b(?:lowest|min|minimum|bottom)\b", query, re.I):
            extremes.append("lowest")

        # 6. Secondary Analysis (e.g. 'average sales for each product')
        secondary = []
        m_sec = re.search(r"\b(?:average|mean)\s+([a-zA-Z0-9_]+)\s+for\s+each\s+([a-zA-Z0-9_]+)\b", query, re.I)
        if m_sec:
            m_met, m_dim = m_sec.group(1).lower(), m_sec.group(2).lower()
            if m_met in col_map and m_dim in col_map:
                sec_dim_col = col_map[m_dim]
                sec_met_col = col_map[m_met]
                # If primary grouping is already identical, skip secondary
                if group_by != [sec_dim_col]:
                    secondary.append({
                        "group_by": [sec_dim_col],
                        "metric": sec_met_col,
                        "function": "mean",
                        "display": f"Average {sec_met_col.title()}",
                    })

        return AnalyticalQueryPlan(
            filter=filter_node,
            group_by=group_by,
            aggregations=aggs,
            ranking=ranking,
            extremes=extremes,
            secondary_analysis=secondary,
        )

    @classmethod
    def parse_aggregations(cls, query: str, columns: Optional[List[str]] = None) -> List[AggregationRequest]:
        """Extract AggregationRequest(s) from a natural language query."""
        q = query.strip()
        aggregations: List[AggregationRequest] = []
        col_map = {c.lower(): c for c in columns} if columns else {}

        if columns:
            candidates = list(col_map.values())
        else:
            raw_tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", q)
            candidates = [c for c in raw_tokens if c.lower() not in GRAMMAR_STOPWORDS]

        for col in candidates:
            c_esc = re.escape(col)
            # Sum / Total patterns
            sum_pat = re.compile(rf"\b(?:total|sum(?:\s+of)?)\s+{c_esc}\b|\b{c_esc}\s+(?:total|sum)\b", re.I)
            if sum_pat.search(q):
                aggregations.append(AggregationRequest(
                    column=col_map.get(col.lower(), col),
                    function="sum",
                    display_name=f"Total {col_map.get(col.lower(), col).title()}",
                ))
                continue

            # Average / Mean patterns (only if not a regional breakdown instruction)
            avg_pat = re.compile(rf"\b(?:average|mean(?:\s+of)?|avg)\s+{c_esc}\b|\b{c_esc}\s+(?:average|mean)\b", re.I)
            if avg_pat.search(q) and not re.search(rf"\baverage\s+{c_esc}\s+for\s+each\b", q, re.I):
                aggregations.append(AggregationRequest(
                    column=col_map.get(col.lower(), col),
                    function="mean",
                    display_name=f"Average {col_map.get(col.lower(), col).title()}",
                ))
                continue

            # Maximum patterns
            max_pat = re.compile(rf"\b(?:max|maximum|highest)\s+{c_esc}\b|\b{c_esc}\s+(?:max|maximum)\b", re.I)
            if max_pat.search(q):
                aggregations.append(AggregationRequest(
                    column=col_map.get(col.lower(), col),
                    function="max",
                    display_name=f"Maximum {col_map.get(col.lower(), col).title()}",
                ))
                continue

            # Minimum patterns
            min_pat = re.compile(rf"\b(?:min|minimum|lowest)\s+{c_esc}\b|\b{c_esc}\s+(?:min|minimum)\b", re.I)
            if min_pat.search(q):
                aggregations.append(AggregationRequest(
                    column=col_map.get(col.lower(), col),
                    function="min",
                    display_name=f"Minimum {col_map.get(col.lower(), col).title()}",
                ))
                continue

        return aggregations

    @classmethod
    def parse_breakdowns(cls, query: str, columns: Optional[List[str]] = None) -> List[BreakdownRequest]:
        """Extract group breakdown and dimensional analysis requests from query."""
        q = query.strip()
        requests: List[BreakdownRequest] = []
        col_map = {c.lower(): c for c in columns} if columns else {}

        bd_match = re.search(r"\b(?:break\s+down(?:\s+the\s+filtered\s+results?)?|group(?:\s+by)?)\s+by\s+([a-zA-Z0-9_]+)", q, re.I)
        if bd_match:
            raw_dim = bd_match.group(1).lower()
            resolved_dim = col_map.get(raw_dim, raw_dim)
            metrics = []
            if re.search(r"\btotal\s+sales\b", q, re.I):
                metrics.append(("sales", "sum"))
            if re.search(r"\btotal\s+units\b", q, re.I):
                metrics.append(("units", "sum"))
            if not metrics:
                metrics = [("sales", "sum")]
            requests.append(BreakdownRequest(dimension=resolved_dim, metrics=metrics))

        avg_match = re.search(r"\baverage\s+([a-zA-Z0-9_]+)\s+for\s+each\s+(?:matching\s+)?([a-zA-Z0-9_]+)", q, re.I)
        if avg_match:
            met_name = avg_match.group(1).lower()
            raw_dim = avg_match.group(2).lower()
            resolved_dim = col_map.get(raw_dim, raw_dim)
            resolved_met = col_map.get(met_name, met_name)

            find_high = bool(re.search(rf"\b(highest|max|maximum)\s+(?:average\s+)?{re.escape(met_name)}\b", q, re.I))
            find_low = bool(re.search(rf"\b(lowest|min|minimum)\s+(?:average\s+)?{re.escape(met_name)}\b", q, re.I))

            requests.append(BreakdownRequest(
                dimension=resolved_dim,
                metrics=[(resolved_met, "mean")],
                find_highest=find_high,
                find_lowest=find_low,
                extreme_metric=resolved_met,
            ))

        return requests

    @classmethod
    def execute(
        cls,
        df: pd.DataFrame,
        query: str,
        compound_filter: Optional[CompoundFilter] = None,
        aggregations: Optional[List[AggregationRequest]] = None,
        breakdown_requests: Optional[List[BreakdownRequest]] = None,
    ) -> FilterExecutionResult:
        """
        Executes analytical query plan (filter, group_by, aggregations, ranking, extremes, secondary analysis).
        """
        cols = list(df.columns)
        total_rows = len(df)

        # Check for date filter errors (never silently ignore date filter)
        _, date_col_detected, date_err = cls.parse_date_filter(query, columns=cols, dataframe=df)
        if date_err:
            return FilterExecutionResult(
                filter_description=date_err,
                matching_rows=0,
                total_rows=total_rows,
                aggregations={},
                filtered_df=df.iloc[0:0].copy(),
                columns=cols,
                markdown_response=f"⚠️ {date_err}",
                breakdowns={},
                filter_ast={},
            )

        plan = cls.parse_query_plan(query, columns=cols, dataframe=df)
        if compound_filter:
            plan.filter = compound_filter
        if aggregations:
            plan.aggregations = aggregations

        # Execute Filter
        if plan.filter:
            mask = plan.filter.evaluate(df)
            filtered_df = df[mask].copy()
            filter_desc = plan.filter.to_expression()
            filter_ast = plan.filter.to_ast()
        else:
            filtered_df = df.copy()
            if cls.has_filter_intent(query) and not plan.group_by:
                filter_desc = "Could not reliably parse requested filter from query."
            else:
                filter_desc = "None"
            filter_ast = {}

        matching_rows = len(filtered_df)

        # Step 1 Temporary Structured Trace Logging
        logger.info(
            "\n[ANALYTICAL_EXECUTION_TRACE]\n"
            "  RAW USER QUERY: %r\n"
            "  INTENT: %s\n"
            "  EXTRACTED FILTER: %s\n"
            "  FILTER AST: %s\n"
            "  GROUP BY COLUMNS: %s\n"
            "  AGGREGATIONS: %s\n"
            "  SORT/RANK: %s\n"
            "  EXTREME REQUEST: %s\n"
            "  SELECTED COLUMNS: %s\n"
            "  DATASET ROW COUNT BEFORE FILTER: %d\n"
            "  DATASET ROW COUNT AFTER FILTER: %d\n"
            "  FINAL EXECUTION PLAN: %s\n",
            query,
            "grouping_and_ranking" if plan.group_by else ("filtering" if plan.filter else "aggregation"),
            filter_desc,
            filter_ast,
            plan.group_by,
            [f"{a.function}({a.column})" for a in plan.aggregations],
            plan.ranking,
            plan.extremes,
            cols,
            total_rows,
            matching_rows,
            plan.to_dict(),
        )

        # Top-level aggregations
        aggs = plan.aggregations
        if not aggs and matching_rows > 0:
            default_aggs = []
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in ("id", "index", "discount")]
            for nc in num_cols[:2]:
                default_aggs.append(AggregationRequest(column=nc, function="sum", display_name=f"Total {nc.title()}"))
                if nc.lower() in ("sales", "revenue", "amount", "price", "salary") and not plan.group_by:
                    default_aggs.append(AggregationRequest(column=nc, function="mean", display_name=f"Average {nc.title()}"))
            aggs = default_aggs

        agg_results: Dict[str, Dict[str, Any]] = {}
        for agg in aggs:
            c = agg.column
            resolved_col = next((col for col in filtered_df.columns if col.lower() == c.lower()), c)
            if resolved_col in filtered_df.columns:
                series = pd.to_numeric(filtered_df[resolved_col], errors="coerce").dropna()
                val = 0.0
                if agg.function == "sum":
                    val = float(series.sum())
                elif agg.function == "mean":
                    val = float(series.mean()) if not series.empty else 0.0
                elif agg.function == "max":
                    val = float(series.max()) if not series.empty else 0.0
                elif agg.function == "min":
                    val = float(series.min()) if not series.empty else 0.0
                elif agg.function == "count":
                    val = int(len(series))

                formatted_val = f"{int(val):,}" if val.is_integer() else f"{val:,.2f}"
                key = resolved_col if resolved_col not in agg_results else f"{resolved_col}_{agg.function}"
                agg_results[key] = {
                    "function": agg.function,
                    "value": val,
                    "formatted": formatted_val,
                    "display_name": agg.display_name or f"{agg.function.title()} of {resolved_col}",
                }

        # Multi-dimension Grouping & Ranking Execution
        grouped_records = None
        highest_record_dict = None
        lowest_record_dict = None
        grouped_summary_lines = []

        if plan.group_by and matching_rows > 0:
            agg_map = {}
            for agg in aggs:
                agg_map[agg.column] = agg.function
            if not agg_map:
                if "sales" in filtered_df.columns:
                    agg_map["sales"] = "sum"
                if "units" in filtered_df.columns:
                    agg_map["units"] = "sum"

            grp_df = filtered_df.groupby(plan.group_by, dropna=False).agg(agg_map).reset_index()

            # Ranking
            if plan.ranking and plan.ranking["column"] in grp_df.columns:
                grp_df = grp_df.sort_values(
                    by=plan.ranking["column"],
                    ascending=(plan.ranking["direction"] == "asc")
                ).reset_index(drop=True)

            grp_df.insert(0, "Rank", range(1, len(grp_df) + 1))
            grouped_records = grp_df.to_dict(orient="records")

            # Extremes
            if "highest" in plan.extremes and not grp_df.empty:
                h_row = grp_df.iloc[0]
                comb_name = " / ".join(str(h_row[c]) for c in plan.group_by)
                s_val = float(h_row["sales"]) if "sales" in h_row else 0.0
                u_val = float(h_row["units"]) if "units" in h_row else 0.0
                highest_record_dict = {
                    "combination": comb_name,
                    "sales": s_val,
                    "units": u_val,
                    "formatted_sales": f"{int(s_val):,}" if s_val.is_integer() else f"{s_val:,.2f}",
                }
            if "lowest" in plan.extremes and not grp_df.empty:
                l_row = grp_df.iloc[-1]
                comb_name = " / ".join(str(l_row[c]) for c in plan.group_by)
                s_val = float(l_row["sales"]) if "sales" in l_row else 0.0
                u_val = float(l_row["units"]) if "units" in l_row else 0.0
                lowest_record_dict = {
                    "combination": comb_name,
                    "sales": s_val,
                    "units": u_val,
                    "formatted_sales": f"{int(s_val):,}" if s_val.is_integer() else f"{s_val:,.2f}",
                }

            # Format Combination Table
            title_dims = " & ".join(c.title() for c in plan.group_by)
            grouped_summary_lines.append(f"\n### {title_dims} Combinations (Ranked by Total Sales):\n")
            headers = ["Rank"] + [c.title() for c in plan.group_by] + [f"Total {k.title()}" for k in agg_map.keys()]
            alignments = [":---"] + [":---"] * len(plan.group_by) + ["---:"] * len(agg_map)
            grouped_summary_lines.append("| " + " | ".join(headers) + " |")
            grouped_summary_lines.append("| " + " | ".join(alignments) + " |")

            for _, r in grp_df.iterrows():
                row_vals = [str(r["Rank"])]
                for c in plan.group_by:
                    row_vals.append(str(r[c]))
                for k in agg_map.keys():
                    val = r[k]
                    fmt = f"{int(val):,}" if isinstance(val, (int, float, np.number)) and float(val).is_integer() else f"{val:,.2f}"
                    row_vals.append(fmt)
                grouped_summary_lines.append("| " + " | ".join(row_vals) + " |")

            if highest_record_dict or lowest_record_dict:
                grouped_summary_lines.append("\n### Extreme Combinations:")
                if highest_record_dict:
                    grouped_summary_lines.append(f"- **Highest Sales Combination**: **{highest_record_dict['combination']}** (Sales: **{highest_record_dict['formatted_sales']}**, Units: **{int(highest_record_dict['units'])}**)")
                if lowest_record_dict:
                    grouped_summary_lines.append(f"- **Lowest Sales Combination**: **{lowest_record_dict['combination']}** (Sales: **{lowest_record_dict['formatted_sales']}**, Units: **{int(lowest_record_dict['units'])}**)")

        # Highest record for non-grouping queries
        elif matching_rows > 0 and "sales" in filtered_df.columns:
            try:
                max_idx = filtered_df["sales"].idxmax()
                max_row = filtered_df.loc[max_idx]
                d_col, _ = cls.detect_date_column(df, query, cols)
                d_val = str(max_row[d_col]) if d_col and d_col in filtered_df.columns else "N/A"
                s_val = float(max_row["sales"])
                highest_record_dict = {
                    "date": d_val,
                    "sales": s_val,
                    "formatted_sales": f"{int(s_val):,}" if s_val.is_integer() else f"{s_val:,.2f}",
                }
            except Exception:
                pass

        # Secondary Analysis Execution (e.g. average sales for each product)
        secondary_lines = []
        sec_results = []
        if plan.secondary_analysis and matching_rows > 0:
            for sec in plan.secondary_analysis:
                sec_dim = sec["group_by"][0]
                sec_met = sec["metric"]
                sec_fn = sec["function"]
                sec_df = filtered_df.groupby(sec_dim)[sec_met].agg(sec_fn).reset_index()

                sec_records = []
                secondary_lines.append(f"\n### {sec['display']} by {sec_dim.title()}:\n")
                for _, r in sec_df.iterrows():
                    val = float(r[sec_met])
                    fmt = f"{int(val):,}" if val.is_integer() else f"{val:,.2f}"
                    secondary_lines.append(f"- **{r[sec_dim]}**: **{fmt}**")
                    sec_records.append({sec_dim: str(r[sec_dim]), sec_met: val, "formatted": fmt})
                sec_results.append({
                    "group_by": sec_dim,
                    "metric": sec_met,
                    "function": sec_fn,
                    "display": sec["display"],
                    "records": sec_records,
                })

        # Compute dimensional breakdowns
        b_reqs = breakdown_requests if breakdown_requests is not None else cls.parse_breakdowns(query, columns=cols)
        breakdowns_output: Dict[str, Any] = {}
        for breq in b_reqs:
            dim_col = next((col for col in filtered_df.columns if col.lower() == breq.dimension.lower()), None)
            if not dim_col or filtered_df.empty:
                continue

            agg_dict = {}
            for met, fn in breq.metrics:
                m_col = next((col for col in filtered_df.columns if col.lower() == met.lower()), None)
                if m_col:
                    agg_dict[m_col] = fn

            if not agg_dict:
                continue

            grouped = filtered_df.groupby(dim_col).agg(agg_dict).reset_index()
            records = []
            for _, grow in grouped.iterrows():
                rec = {dim_col: str(grow[dim_col])}
                for m_col, fn in agg_dict.items():
                    val = float(grow[m_col])
                    rec[m_col] = val
                    rec[f"{m_col}_formatted"] = f"{int(val):,}" if val.is_integer() else f"{val:,.2f}"
                records.append(rec)

            b_data: Dict[str, Any] = {"dimension": dim_col, "records": records}

            if breq.find_highest and breq.extreme_metric:
                m_col = next((col for col in filtered_df.columns if col.lower() == breq.extreme_metric.lower()), None)
                if m_col and records:
                    highest_rec = max(records, key=lambda r: r.get(m_col, 0))
                    b_data["highest"] = {
                        dim_col: highest_rec[dim_col],
                        "metric": m_col,
                        "value": highest_rec[m_col],
                        "formatted": highest_rec[f"{m_col}_formatted"],
                    }

            breakdowns_output[dim_col] = b_data

        # Format markdown response
        lines = [
            f"🎯 **Analytical Query Result**:\n",
            f"- **Filter Applied**: `{filter_desc}`",
            f"- **Matching Records**: **{matching_rows:,}** (out of {total_rows:,} total rows)",
        ]

        for col, res in agg_results.items():
            lines.append(f"- **{res['display_name']}**: **{res['formatted']}**")

        if highest_record_dict and not plan.group_by:
            lines.append(f"- **Highest Sales Record**: **{highest_record_dict['date']}** (Sales: **{highest_record_dict['formatted_sales']}**)")

        if grouped_summary_lines:
            lines.extend(grouped_summary_lines)

        # Format breakdowns if not already covered by grouped_summary_lines
        if not grouped_summary_lines and breakdowns_output:
            for dim_col, b_data in breakdowns_output.items():
                records = b_data.get("records", [])
                if not records:
                    continue
                metric_keys = [k for k in records[0].keys() if k != dim_col and not k.endswith("_formatted")]
                if len(metric_keys) >= 2:
                    lines.append(f"\n### {dim_col.title()} Breakdown:\n")
                    headers = [dim_col.title()] + [k.title() for k in metric_keys]
                    lines.append("| " + " | ".join(headers) + " |")
                    lines.append("| :--- | " + " | ".join(["---:"] * len(metric_keys)) + " |")
                    for r in records:
                        row_vals = [r[dim_col]] + [r[f"{k}_formatted"] for k in metric_keys]
                        lines.append("| " + " | ".join(row_vals) + " |")
                else:
                    met_key = metric_keys[0] if metric_keys else "metric"
                    lines.append(f"\n### Regional Analysis ({dim_col.title()} Averages):\n")
                    for r in records:
                        lines.append(f"- **{r[dim_col]}**: Average {met_key.title()} = **{r[f'{met_key}_formatted']}**")
                    if "highest" in b_data:
                        h = b_data["highest"]
                        lines.append(f"- **Highest Average {h['metric'].title()} {dim_col.title()}**: **{h[dim_col]}** ({h['formatted']})")

        if secondary_lines:
            lines.extend(secondary_lines)

        if "causal" in query.lower():
            lines.append("\n> [!NOTE]\n> *Projections and aggregations reflect observed mathematical associations without asserting causal claims.*")

        # Add preview table if not already displaying grouped combinations table
        if not plan.group_by and not breakdowns_output and matching_rows > 0:
            lines.append("\n**Filtered Records Preview**:\n")
            preview_rows = filtered_df.head(10)
            headers = list(preview_rows.columns)
            alignments = []
            for h in headers:
                is_num = pd.api.types.is_numeric_dtype(preview_rows[h])
                alignments.append("---:" if is_num else ":---")

            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(alignments) + " |")

            for _, row in preview_rows.iterrows():
                row_vals = []
                for h in headers:
                    v = row[h]
                    if pd.isna(v):
                        row_vals.append("—")
                    elif isinstance(v, (int, np.integer)):
                        row_vals.append(f"{v:,}")
                    elif isinstance(v, (float, np.floating)):
                        row_vals.append(f"{v:,.2f}" if not v.is_integer() else f"{int(v):,}")
                    else:
                        row_vals.append(str(v))
                lines.append("| " + " | ".join(row_vals) + " |")

        markdown_resp = "\n".join(lines)

        result_obj = FilterExecutionResult(
            filter_description=filter_desc,
            matching_rows=matching_rows,
            total_rows=total_rows,
            aggregations=agg_results,
            filtered_df=filtered_df,
            columns=cols,
            markdown_response=markdown_resp,
            breakdowns=breakdowns_output,
            filter_ast=filter_ast,
            highest_record=highest_record_dict,
            lowest_record=lowest_record_dict if plan.group_by else None,
            query_plan=plan.to_dict(),
            group_by=plan.group_by,
            grouped_records=grouped_records,
            secondary_results=sec_results,
        )

        diag_exec = (
            f"\nEXECUTION_RESULT\n"
            f"type={type(result_obj).__name__}\n"
            f"keys={list(result_obj.to_dict().keys())}\n"
        )
        print(diag_exec)
        import logging
        logging.getLogger("diagnostic").info(diag_exec)

        return result_obj
