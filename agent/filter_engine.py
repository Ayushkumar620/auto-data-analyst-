"""
Filter Engine - Autonomous analytical filter parsing, compilation, and execution engine.
Parses natural-language filters (simple, compound with AND/OR, 'either...or', 'one of'),
applies safe vectorized boolean indexing, compiles to a structured AST, and executes
aggregations, group breakdowns, and dimensional statistics on the filtered subset.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

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


@dataclass
class FilterCondition:
    """Represents a single atomic filter comparison condition on a column."""
    column: str
    operator: str  # '<=', '>=', '<', '>', '==', '!=', 'in', 'not in'
    value: Any
    raw_expression: str = ""

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

        # Numeric comparisons
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

        # Categorical 'in' / 'not in'
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
            "expression": self.to_expression(),
        }

    def to_expression(self) -> str:
        """Render condition as human-readable boolean expression."""
        if self.raw_expression:
            return self.raw_expression
        if self.operator == "in" and isinstance(self.value, (list, tuple, set)):
            val_strs = [f"'{v}'" if isinstance(v, str) else str(v) for v in self.value]
            return f"({ ' OR '.join([f'{self.column} == {v}' for v in val_strs]) })"
        val_str = f"'{self.value}'" if isinstance(self.value, str) else str(self.value)
        return f"{self.column} {self.operator} {val_str}"

    @staticmethod
    def _is_numeric_str(val: str) -> bool:
        try:
            float(val)
            return True
        except ValueError:
            return False


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


class FilterEngine:
    """
    Parser and execution engine for natural language filtering, compound expressions,
    aggregations, and grouped dimensional breakdowns.
    """

    # Comparison operator patterns for numbers (ordered most specific to least specific)
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

    # Filter indicator keywords
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
    def _extract_filter_clause(cls, query: str) -> str:
        """Extract the specific substring containing the filter conditions."""
        where_match = re.search(r"\b(?:where|filter(?:ed)?(?:\s+by|\s+on)?|having|with)\s+([^.]+)", query, re.I)
        if where_match:
            clause = where_match.group(1).strip()
            # Trim trailing action instructions (e.g. "Calculate...", "Break down...", "Do not include...")
            end_match = re.search(
                r"\b(?:calculate|compute|aggregate|break\s+down|showing|and\s+calculate|then\s+calculate|do\s+not\s+include)\b",
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
        Supports numeric comparisons, categorical equality, 'either ... or', 'one of', and compound AND/OR logic.
        """
        if not cls.has_filter_intent(query):
            return None

        clause_text = cls._extract_filter_clause(query)
        col_map = {c.lower(): c for c in columns} if columns else {}

        # If dataframe provided, build column map from df columns
        if dataframe is not None and not col_map:
            col_map = {c.lower(): c for c in dataframe.columns}
            columns = list(dataframe.columns)

        # Split clause on 'AND' while respecting parentheses
        parts = cls._split_conjunctions(clause_text)

        conditions: List[Union[FilterCondition, CompoundFilter]] = []
        for part in parts:
            part_cleaned = part.strip().strip("()")

            # 1. First check if it's an atomic numeric condition (e.g. discount <= 5)
            atomic_num = cls._parse_numeric_condition(part_cleaned, col_map, columns)
            if atomic_num:
                conditions.append(atomic_num)
                continue

            # 2. Check if it's a categorical condition with OR / either / one of / in
            cat_or = cls._parse_categorical_condition(part_cleaned, col_map, columns, dataframe)
            if cat_or:
                conditions.append(cat_or)
                continue

            # 3. Check for multiple distinct expressions separated by OR: e.g. 'region = North or region = West'
            if re.search(r"\bor\b", part_cleaned, re.I):
                or_cond = cls._parse_compound_or_condition(part_cleaned, col_map, columns, dataframe)
                if or_cond:
                    conditions.append(or_cond)
                    continue

        if not conditions:
            # Fallback: scan whole query text for numeric comparison patterns
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

        # Step 1 Temporary Trace Logging
        logger.info(
            "\n[NATURAL_LANGUAGE_FILTER_PARSER_TRACE]\n"
            "  RAW QUERY: %r\n"
            "  EXTRACTED FILTER TEXT: %r\n"
            "  PARSED CONDITIONS: %s\n"
            "  FINAL FILTER AST/EXPRESSION: %s\n"
            "  EXECUTED FILTER: %s\n",
            query,
            clause_text,
            [c.to_ast() for c in conditions],
            compound.to_ast(),
            compound.to_expression(),
        )

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
        """
        Parse natural-language categorical conditions:
        - 'region is either North or West'
        - 'region is North or West'
        - 'region can be North or West'
        - 'region is one of North or West'
        - 'region is one of North, West'
        - 'region in North or West'
        - 'region equals North or West'
        - 'region is North, West'
        - 'region is either North, West, or South'
        - 'product is Laptop or Phone'
        """
        clean_text = re.sub(r"^(?:the|a|an|where|that)\s+", "", text.strip(), flags=re.I).strip("()[]")

        # Check which known column appears at or near the start
        col_name = None
        rest = None

        # Priority 1: Match against known columns in col_map
        if col_map:
            for c_low, c_orig in col_map.items():
                pat = re.compile(
                    rf"^\b{re.escape(c_low)}\b\s*(?:is|can\s+be|equals?|==?|in|one\s+of)?\s*(.+)",
                    re.I,
                )
                m = pat.match(clean_text)
                if m:
                    col_name = c_orig
                    rest = m.group(1).strip()
                    break

        # Priority 2: Generic token match if no col_map passed
        if not col_name:
            gen_pat = re.compile(
                r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:is|can\s+be|equals?|==?|in|one\s+of)\s*(.+)",
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

        # Strip leading logical modifiers from rest: e.g. 'either ', 'one of ', 'any of '
        rest = re.sub(r"^(?:either|one\s+of|any\s+of|either\s+of)\s+", "", rest, flags=re.I).strip("()[]")

        # Split on ', or', ' or ', ','
        raw_vals = [
            v.strip().strip("'\"()")
            for v in re.split(r",?\s+\bor\b\s*|,\s*", rest, flags=re.I)
            if v.strip()
        ]

        # Clean and filter candidate values (Step 7 & Step 10: NEVER allow 'either' or stopwords)
        clean_vals = [
            v for v in raw_vals
            if v.lower() not in LOGICAL_MODIFIERS and v.lower() not in STOPWORDS
        ]

        # Schema validation (Step 10: validate against dataframe unique values if available)
        if dataframe is not None and col_name in dataframe.columns:
            actual_uniques = {str(x).strip().lower(): str(x) for x in dataframe[col_name].dropna().unique()}
            validated_vals = []
            for cv in clean_vals:
                if cv.lower() in actual_uniques:
                    validated_vals.append(actual_uniques[cv.lower()])
                else:
                    validated_vals.append(cv)
            clean_vals = validated_vals

        if not clean_vals:
            return None

        # Step 10: Check for suspicious values like 'either'
        for v in clean_vals:
            if v.lower() in LOGICAL_MODIFIERS:
                logger.warning(
                    "Rejected suspicious modifier '%s' as categorical value for column '%s'",
                    v,
                    col_name,
                )
                return None

        if len(clean_vals) > 1:
            # Build proper expression: (col == 'A' OR col == 'B')
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
        """Parse multiple distinct expressions connected by OR (e.g. 'region = North or region = West')."""
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
    def parse_aggregations(cls, query: str, columns: Optional[List[str]] = None) -> List[AggregationRequest]:
        """
        Extract AggregationRequest(s) from a natural language query.
        e.g. 'Calculate the total sales and total units' -> [sales (sum), units (sum)]
        """
        q = query.strip()
        aggregations: List[AggregationRequest] = []
        col_map = {c.lower(): c for c in columns} if columns else {}

        if columns:
            candidates = list(col_map.values())
        else:
            raw_tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", q)
            candidates = [c for c in raw_tokens if c.lower() not in STOPWORDS]

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
        """
        Extract group breakdown and dimensional analysis requests from query:
        - 'Break down the filtered results by product, showing total sales and total units for each product'
        - 'calculate the average sales for each matching region and identify which region has the highest average sales'
        """
        q = query.strip()
        requests: List[BreakdownRequest] = []
        col_map = {c.lower(): c for c in columns} if columns else {}

        # 1. Look for explicit 'break down by <dim>' or 'by <dim>'
        bd_match = re.search(r"\b(?:break\s+down(?:\s+the\s+filtered\s+results?)?|group(?:\s+by)?)\s+by\s+([a-zA-Z0-9_]+)", q, re.I)
        if bd_match:
            raw_dim = bd_match.group(1).lower()
            resolved_dim = col_map.get(raw_dim, raw_dim)
            # Find what metrics to show for each dim item
            metrics = []
            if re.search(r"\btotal\s+sales\b", q, re.I):
                metrics.append(("sales", "sum"))
            if re.search(r"\btotal\s+units\b", q, re.I):
                metrics.append(("units", "sum"))
            if not metrics:
                metrics = [("sales", "sum")]
            requests.append(BreakdownRequest(dimension=resolved_dim, metrics=metrics))

        # 2. Look for 'average <metric> for each matching <dim>' or 'for each <dim>'
        avg_match = re.search(r"\baverage\s+([a-zA-Z0-9_]+)\s+for\s+each\s+(?:matching\s+)?([a-zA-Z0-9_]+)", q, re.I)
        if avg_match:
            met_name = avg_match.group(1).lower()
            raw_dim = avg_match.group(2).lower()
            resolved_dim = col_map.get(raw_dim, raw_dim)
            resolved_met = col_map.get(met_name, met_name)

            # Check if highest / lowest requested
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
        Executes filtering, aggregations, and dimensional breakdowns against DataFrame.
        """
        cols = list(df.columns)
        filter_expr = compound_filter or cls.parse_filters(query, columns=cols, dataframe=df)
        aggs = aggregations or cls.parse_aggregations(query, columns=cols)
        b_reqs = breakdown_requests if breakdown_requests is not None else cls.parse_breakdowns(query, columns=cols)

        total_rows = len(df)
        if filter_expr:
            mask = filter_expr.evaluate(df)
            filtered_df = df[mask].copy()
            filter_desc = filter_expr.to_expression()
            filter_ast = filter_expr.to_ast()
        else:
            filtered_df = df.copy()
            filter_desc = "All Records (No filter applied)"
            filter_ast = {}

        matching_rows = len(filtered_df)

        # Compute top-level aggregations
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
                agg_results[resolved_col] = {
                    "function": agg.function,
                    "value": val,
                    "formatted": formatted_val,
                    "display_name": agg.display_name or f"{agg.function.title()} of {resolved_col}",
                }

        # Compute dimensional breakdowns (Step 13)
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

            # Check extreme if requested
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

        # Format rich markdown response
        lines = [
            f"🎯 **Filtered Analysis Result**:\n",
            f"- **Filter Applied**: `{filter_desc}`",
            f"- **Matching Records**: **{matching_rows:,}** (out of {total_rows:,} total rows)",
        ]

        for col, res in agg_results.items():
            lines.append(f"- **{res['display_name']}**: **{res['formatted']}**")

        # Format breakdowns if computed
        for dim_col, b_data in breakdowns_output.items():
            records = b_data.get("records", [])
            if not records:
                continue

            # If breakdown has multiple metrics (e.g. sales & units), format table
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
                # Single metric average breakdown
                met_key = metric_keys[0] if metric_keys else "metric"
                lines.append(f"\n### Regional Analysis ({dim_col.title()} Averages):\n")
                for r in records:
                    lines.append(f"- **{r[dim_col]}**: Average {met_key.title()} = **{r[f'{met_key}_formatted']}**")
                if "highest" in b_data:
                    h = b_data["highest"]
                    lines.append(f"- **Highest Average {h['metric'].title()} {dim_col.title()}**: **{h[dim_col]}** ({h['formatted']})")

        # Avoid causal claims disclaimer if query requested
        if "causal" in query.lower():
            lines.append("\n> [!NOTE]\n> *Projections and aggregations reflect observed mathematical associations without asserting causal claims.*")

        markdown_resp = "\n".join(lines)

        return FilterExecutionResult(
            filter_description=filter_desc,
            matching_rows=matching_rows,
            total_rows=total_rows,
            aggregations=agg_results,
            filtered_df=filtered_df,
            columns=cols,
            markdown_response=markdown_resp,
            breakdowns=breakdowns_output,
            filter_ast=filter_ast,
        )
