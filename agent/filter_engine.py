"""
Filter Engine - Autonomous analytical filter parsing and execution engine.
Parses natural-language filters (simple and compound with AND/OR), applies numeric
and categorical comparisons, and executes aggregations on the filtered subset.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
        re.compile(r"\bwith\s+[a-z0-9_]+\s*(?:<=|>=|<|>|=|!=|is\b)", re.I),
        re.compile(r"[a-z0-9_]+\s*(?:<=|>=|<|>|!=)\s*[0-9]+", re.I),
        re.compile(r"[a-z0-9_]+\s+(?:is\s+)?[0-9]+\s+or\s+less", re.I),
        re.compile(r"[a-z0-9_]+\s+(?:is\s+)?[0-9]+\s+or\s+more", re.I),
        re.compile(r"[a-z0-9_]+\s+(?:is\s+)?[0-9]+\s+or\s+fewer", re.I),
        re.compile(r"[a-z0-9_]+\s+(?:is\s+)?at\s+most\s+[0-9]+", re.I),
        re.compile(r"[a-z0-9_]+\s+(?:is\s+)?at\s+least\s+[0-9]+", re.I),
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
        # 1. Look for explicit where / filter by / with / having clause
        where_match = re.search(r"\b(?:where|filter(?:ed)?(?:\s+by|\s+on)?|having|with)\s+([^.]+)", query, re.I)
        if where_match:
            clause = where_match.group(1).strip()
            # If clause ends with a sentence stop or aggregation instruction, trim it
            # e.g., "discount is 5 or less. Calculate the total..." -> "discount is 5 or less"
            end_match = re.search(r"\b(?:calculate|compute|aggregate|and\s+calculate|then\s+calculate)\b", clause, re.I)
            if end_match:
                clause = clause[:end_match.start()].strip()
            return clause

        return query

    @classmethod
    def parse_filters(cls, query: str, columns: Optional[List[str]] = None) -> Optional[CompoundFilter]:
        """
        Extract FilterCondition(s) from a natural language query against dataset columns.
        Supports numeric comparisons, categorical equality, and compound AND/OR logic.
        """
        if not cls.has_filter_intent(query):
            return None

        clause_text = cls._extract_filter_clause(query)
        col_map = {c.lower(): c for c in columns} if columns else {}

        # Split clause on 'AND' while respecting parentheses
        # e.g., "discount <= 5 and (region = North or region = West)"
        parts = cls._split_conjunctions(clause_text)

        conditions: List[Union[FilterCondition, CompoundFilter]] = []
        for part in parts:
            part_cleaned = part.strip().strip("()")
            # 1. First check if it's an atomic condition (e.g. numeric comparison with 'or less' / 'or more')
            atomic = cls._parse_atomic_condition(part_cleaned, col_map, columns)
            if atomic:
                conditions.append(atomic)
                continue

            # 2. Check if this part contains an 'OR' compound:
            # e.g. "region is North or West" or "region = North or region = West"
            if re.search(r"\bor\b", part_cleaned, re.I):
                or_cond = cls._parse_or_condition(part_cleaned, col_map, columns)
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

        return CompoundFilter(conditions=conditions, logical_op="AND")

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
    def _parse_or_condition(
        cls,
        text: str,
        col_map: Dict[str, str],
        columns: Optional[List[str]],
    ) -> Optional[Union[FilterCondition, CompoundFilter]]:
        """Parse an OR condition: e.g. 'region is North or West' or 'region = North or region = West'."""
        # Check: single column multiple values: <col> (?:is|==?|in)? <val1> or <val2>
        col_pat = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b\s*(?:is\s+|==?\s+|in\s+)?(.+)", re.I)
        m = col_pat.match(text)
        if m:
            potential_col = m.group(1)
            rest = m.group(2).strip()
            if (not columns) or (potential_col.lower() in col_map):
                resolved_col = col_map.get(potential_col.lower(), potential_col)
                # Split rest on 'or'
                val_parts = [v.strip().strip("'\"()") for v in re.split(r"\s+\bor\b\s+", rest, flags=re.I) if v.strip()]
                # If each part is just a value (not <col> = <val>)
                if all("=" not in v and " is " not in v for v in val_parts):
                    # Filter out stop words
                    stopwords = {"calculate", "the", "total", "and", "units", "sales", "for", "filtered", "records", "rows"}
                    clean_vals = [v for v in val_parts if v.lower() not in stopwords]
                    if clean_vals:
                        if len(clean_vals) > 1:
                            return FilterCondition(
                                column=resolved_col,
                                operator="in",
                                value=clean_vals,
                                raw_expression=f"{resolved_col} in ({', '.join(clean_vals)})",
                            )
                        else:
                            return FilterCondition(
                                column=resolved_col,
                                operator="==",
                                value=clean_vals[0],
                                raw_expression=f"{resolved_col} == '{clean_vals[0]}'",
                            )

        # Multiple distinct expressions separated by OR: e.g. 'region = North or region = West'
        sub_exprs = re.split(r"\s+\bor\b\s+", text, flags=re.I)
        sub_conds = []
        for se in sub_exprs:
            cond = cls._parse_atomic_condition(se.strip(), col_map, columns)
            if cond:
                sub_conds.append(cond)

        if sub_conds:
            if len(sub_conds) == 1:
                return sub_conds[0]
            return CompoundFilter(conditions=sub_conds, logical_op="OR")

        return None

    @classmethod
    def _parse_atomic_condition(
        cls,
        text: str,
        col_map: Dict[str, str],
        columns: Optional[List[str]],
    ) -> Optional[FilterCondition]:
        """Parse a single atomic condition (e.g. 'discount is 5 or less' or 'region = North')."""
        clean_text = text.strip().strip("()")

        # 1. Numeric patterns
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

        # 2. Categorical equality (only if no 'or' conjunction inside)
        if not re.search(r"\bor\b", clean_text, re.I):
            cat_pat = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b\s*(?:is|==?|=)\s*['\"]?([A-Za-z0-9_-]+)['\"]?", re.I)
            m = cat_pat.search(clean_text)
            if m:
                col_name = m.group(1)
                val_cat = m.group(2).strip()
                if not columns or col_name.lower() in col_map:
                    resolved_col = col_map.get(col_name.lower(), col_name)
                    stopwords = {"calculate", "the", "total", "and", "units", "sales", "for", "filtered", "records", "rows"}
                    if val_cat.lower() not in stopwords:
                        return FilterCondition(
                            column=resolved_col,
                            operator="==",
                            value=val_cat,
                            raw_expression=f"{resolved_col} == '{val_cat}'",
                        )

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

        candidates = list(col_map.values()) if columns else re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", q)

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

            # Average / Mean patterns
            avg_pat = re.compile(rf"\b(?:average|mean(?:\s+of)?|avg)\s+{c_esc}\b|\b{c_esc}\s+(?:average|mean)\b", re.I)
            if avg_pat.search(q):
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

        # If user asks "Calculate the total sales and total units" and columns were not provided
        if not aggregations and not columns:
            matches = re.findall(r"\b(?:total|sum)\s+([a-zA-Z_][a-zA-Z0-9_]*)", q, re.I)
            for m in matches:
                if m.lower() not in ("of", "the", "for", "records", "rows", "these", "filtered"):
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

        # Format clean markdown response
        lines = [
            f"🎯 **Filtered Analysis Result**:\n",
            f"- **Filter Applied**: `{filter_desc}`",
            f"- **Matching Records**: **{matching_rows:,}** (out of {total_rows:,} total rows)",
        ]

        for col, res in agg_results.items():
            lines.append(f"- **{res['display_name']}**: **{res['formatted']}**")

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

        # Development debug logging (Step 2 requirement)
        logger.info(
            "\n[ANALYTICAL_FILTER_ENGINE_TRACE]\n"
            "  1. raw user query: %r\n"
            "  2. detected intent: FILTERING / AGGREGATION\n"
            "  3. extracted entities (columns): %s\n"
            "  4. detected filters: %s\n"
            "  5. generated execution plan: [1. Apply boolean filter mask -> 2. Aggregate filtered records -> 3. Format response]\n"
            "  6. selected agents/tools: FilterEngine (vectorized pandas indexing)\n"
            "  7. execution result: matching_rows=%d, aggregations=%s\n"
            "  8. final response type: filter_result\n",
            query,
            [a.column for a in aggs],
            filter_desc,
            matching_rows,
            {k: v["formatted"] for k, v in agg_results.items()},
        )

        return FilterExecutionResult(
            filter_description=filter_desc,
            matching_rows=matching_rows,
            total_rows=total_rows,
            aggregations=agg_results,
            filtered_df=filtered_df,
            columns=cols,
            markdown_response=markdown_resp,
        )
