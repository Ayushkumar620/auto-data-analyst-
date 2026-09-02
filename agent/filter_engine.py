"""
Filter Engine - Autonomous analytical filter parsing and execution engine.
Parses natural-language filters (simple and compound with AND/OR), applies numeric
and categorical comparisons, and executes aggregations on the filtered subset.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class FilterCondition:
    """Represents a single atomic filter condition on a column."""
    column: str
    operator: str  # '<=', '>=', '<', '>', '==', '!=', 'in', 'not in'
    value: Any
    raw_expression: str = ""

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate condition against DataFrame and return boolean mask."""
        if self.column not in df.columns:
            # Case-insensitive column resolution
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
            # Check if numeric
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

    @staticmethod
    def _is_numeric_str(val: str) -> bool:
        try:
            float(val)
            return True
        except ValueError:
            return False


@dataclass
class CompoundFilter:
    """Represents a compound logical filter (AND / OR)."""
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


@dataclass
class AggregationRequest:
    """Represents an aggregation request on a specific column."""
    column: str
    function: str  # 'sum', 'mean', 'count', 'max', 'min'
    display_name: str = ""


@dataclass
class FilterExecutionResult:
    """Structured analytical result of executing filter + aggregations."""
    filter_description: str
    matching_rows: int
    total_rows: int
    aggregations: Dict[str, Dict[str, Any]]
    filtered_df: pd.DataFrame
    columns: List[str]
    markdown_response: str


class FilterEngine:
    """
    Parser and execution engine for natural language filtering and aggregations.
    """

    # Comparison operator patterns for numbers
    # Ordered from most specific to least specific
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
        re.compile(r"\bwith\s+[a-z0-9_]+\s*(?:<=|>=|<|>|=|!=|is\b)", re.I),
        re.compile(r"[a-z0-9_]+\s*(?:<=|>=|<|>|!=)\s*[0-9]+", re.I),
        re.compile(r"[a-z0-9_]+\s+(?:is\s+)?[0-9]+\s+or\s+less", re.I),
        re.compile(r"[a-z0-9_]+\s+(?:is\s+)?[0-9]+\s+or\s+more", re.I),
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
        # If user asks to calculate or aggregate, it's not a simple preview
        if any(w in q_lower for w in ("calculate", "total", "sum", "average", "mean", "min", "max", "count", "group by", "forecast", "predict")):
            return False

        return any(pat.search(query) for pat in cls.LEGITIMATE_PREVIEW_INDICATORS)

    @classmethod
    def parse_filters(cls, query: str, columns: Optional[List[str]] = None) -> Optional[CompoundFilter]:
        """
        Extract FilterCondition(s) from a natural language query against dataset columns.
        Supports numeric comparisons, categorical equality, and compound AND/OR logic.
        """
        if not cls.has_filter_intent(query):
            return None

        q = query.strip()
        compound = CompoundFilter(logical_op="AND")

        # Map available columns (case-insensitive lookup)
        col_map = {c.lower(): c for c in columns} if columns else {}

        # 1. Look for numeric conditions on columns
        # e.g., "discount is 5 or less", "discount <= 5", "sales > 10000"
        found_conditions: List[FilterCondition] = []

        # Find potential column names in query
        target_cols = []
        if columns:
            for c in columns:
                # check if column name appears in query
                c_pattern = re.compile(rf"\b{re.escape(c)}\b", re.I)
                if c_pattern.search(q):
                    target_cols.append(c)
        else:
            # Infer tokens that might be column names before operators
            tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", q)
            target_cols = tokens

        for col in target_cols:
            col_escaped = re.escape(col)
            # Find context immediately following the column name
            pattern = re.compile(rf"\b{col_escaped}\b\s*(?:is\s+)?([^,.;\n]+)", re.I)
            match = pattern.search(q)
            if not match:
                continue

            predicate_text = match.group(1).strip()

            # Try numeric operator matches
            matched_numeric = False
            for op_regex, op_symbol in cls.NUMERIC_PATTERNS:
                num_match = op_regex.match(predicate_text) or op_regex.search(predicate_text[:40])
                if num_match:
                    num_val = float(num_match.group(1))
                    if num_val.is_integer():
                        num_val = int(num_val)
                    cond = FilterCondition(
                        column=col_map.get(col.lower(), col),
                        operator=op_symbol,
                        value=num_val,
                        raw_expression=f"{col} {op_symbol} {num_val}",
                    )
                    found_conditions.append(cond)
                    matched_numeric = True
                    break

            if matched_numeric:
                continue

            # Check for categorical match e.g. "region is North or West" or "region = North"
            cat_match = re.search(r"^(?:is\s+|==?\s+)?([A-Za-z0-9_-]+(?:\s+or\s+[A-Za-z0-9_-]+)*)", predicate_text, re.I)
            if cat_match:
                raw_cats = cat_match.group(1).strip()
                # Split on 'or'
                cat_values = [v.strip() for v in re.split(r"\s+or\s+", raw_cats, flags=re.I) if v.strip()]
                # Exclude SQL or analytical keywords
                stopwords = {"calculate", "and", "the", "where", "total", "sum", "units", "sales", "filtered"}
                cat_values = [v for v in cat_values if v.lower() not in stopwords]
                if cat_values:
                    if len(cat_values) > 1:
                        cond = FilterCondition(
                            column=col_map.get(col.lower(), col),
                            operator="in",
                            value=cat_values,
                            raw_expression=f"{col} in ({', '.join(cat_values)})",
                        )
                    else:
                        cond = FilterCondition(
                            column=col_map.get(col.lower(), col),
                            operator="==",
                            value=cat_values[0],
                            raw_expression=f"{col} == '{cat_values[0]}'",
                        )
                    found_conditions.append(cond)

        if not found_conditions:
            # Try generic pattern: "where <col> <= <val>" or "<col> <= <val>"
            for op_regex, op_symbol in cls.NUMERIC_PATTERNS:
                generic_pat = re.compile(rf"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+{op_regex.pattern}", re.I)
                for g_match in generic_pat.finditer(q):
                    col = g_match.group(1)
                    num_val = float(g_match.group(2))
                    if num_val.is_integer():
                        num_val = int(num_val)
                    cond = FilterCondition(
                        column=col_map.get(col.lower(), col),
                        operator=op_symbol,
                        value=num_val,
                        raw_expression=f"{col} {op_symbol} {num_val}",
                    )
                    found_conditions.append(cond)

        if not found_conditions:
            return None

        # Check if query contains OR at top level between column conditions
        compound.conditions = found_conditions
        if " or " in q.lower() and not any(isinstance(c.value, list) for c in found_conditions):
            # Check if OR connects distinct columns
            pass

        return compound

    @classmethod
    def parse_aggregations(cls, query: str, columns: Optional[List[str]] = None) -> List[AggregationRequest]:
        """
        Extract AggregationRequest(s) from a natural language query.
        e.g. 'Calculate the total sales and total units' -> [sales (sum), units (sum)]
        """
        q = query.strip()
        aggregations: List[AggregationRequest] = []
        col_map = {c.lower(): c for c in columns} if columns else {}

        # Search for metrics in query: "total <col>", "sum of <col>", "average <col>"
        candidates = list(col_map.values()) if columns else re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", q)

        for col in candidates:
            c_esc = re.escape(col)
            # sum patterns
            sum_pat = re.compile(rf"\b(?:total|sum(?:\s+of)?)\s+{c_esc}\b|\b{c_esc}\s+(?:total|sum)\b", re.I)
            if sum_pat.search(q):
                aggregations.append(AggregationRequest(
                    column=col_map.get(col.lower(), col),
                    function="sum",
                    display_name=f"Total {col_map.get(col.lower(), col).title()}",
                ))
                continue

            # mean/average patterns
            avg_pat = re.compile(rf"\b(?:average|mean(?:\s+of)?|avg)\s+{c_esc}\b|\b{c_esc}\s+(?:average|mean)\b", re.I)
            if avg_pat.search(q):
                aggregations.append(AggregationRequest(
                    column=col_map.get(col.lower(), col),
                    function="mean",
                    display_name=f"Average {col_map.get(col.lower(), col).title()}",
                ))
                continue

            # max patterns
            max_pat = re.compile(rf"\b(?:max|maximum|highest)\s+{c_esc}\b|\b{c_esc}\s+(?:max|maximum)\b", re.I)
            if max_pat.search(q):
                aggregations.append(AggregationRequest(
                    column=col_map.get(col.lower(), col),
                    function="max",
                    display_name=f"Maximum {col_map.get(col.lower(), col).title()}",
                ))
                continue

            # min patterns
            min_pat = re.compile(rf"\b(?:min|minimum|lowest)\s+{c_esc}\b|\b{c_esc}\s+(?:min|minimum)\b", re.I)
            if min_pat.search(q):
                aggregations.append(AggregationRequest(
                    column=col_map.get(col.lower(), col),
                    function="min",
                    display_name=f"Minimum {col_map.get(col.lower(), col).title()}",
                ))
                continue

        # If user asks "Calculate the total sales and total units" and no column was passed
        if not aggregations and not columns:
            matches = re.findall(r"\b(?:total|sum)\s+([a-zA-Z_][a-zA-Z0-9_]*)", q, re.I)
            for m in matches:
                if m.lower() not in ("of", "the", "for", "records", "rows"):
                    aggregations.append(AggregationRequest(column=m, function="sum", display_name=f"Total {m.title()}"))

        return aggregations

    @classmethod
    def execute(
        cls,
        df: pd.DataFrame,
        query: str,
        compound_filter: Optional[CompoundFilter] = None,
        aggregations: Optional[List[AggregationRequest]] = None,
    ) -> FilterExecutionResult:
        """
        Executes filtering and aggregations against DataFrame.
        """
        cols = list(df.columns)
        filter_expr = compound_filter or cls.parse_filters(query, columns=cols)
        aggs = aggregations or cls.parse_aggregations(query, columns=cols)

        total_rows = len(df)
        if filter_expr:
            mask = filter_expr.evaluate(df)
            filtered_df = df[mask].copy()
            desc_parts = [c.raw_expression for c in filter_expr.conditions if hasattr(c, "raw_expression") and c.raw_expression]
            filter_desc = " AND ".join(desc_parts) if desc_parts else "Filtered Subset"
        else:
            filtered_df = df.copy()
            filter_desc = "All Records (No filter applied)"

        matching_rows = len(filtered_df)

        # Compute aggregations
        agg_results: Dict[str, Dict[str, Any]] = {}
        for agg in aggs:
            c = agg.column
            # resolve column
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

        # Build clean markdown response
        lines = [
            f"🎯 **Filtered Analysis Result**:\n",
            f"- **Filter Applied**: `{filter_desc}`",
            f"- **Matching Records**: **{matching_rows:,}** (out of {total_rows:,} total rows)",
        ]

        for col, res in agg_results.items():
            lines.append(f"- **{res['display_name']}**: **{res['formatted']}**")

        # If filtered rows exist, attach Markdown table of the filtered rows
        if matching_rows > 0:
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

        return FilterExecutionResult(
            filter_description=filter_desc,
            matching_rows=matching_rows,
            total_rows=total_rows,
            aggregations=agg_results,
            filtered_df=filtered_df,
            columns=cols,
            markdown_response=markdown_resp,
        )
