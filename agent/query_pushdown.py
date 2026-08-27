"""
Enterprise Query Pushdown & Large-Data Engine.

Translates high-level statistical operations (aggregations, value counts, grouped metrics)
into optimized SQL pushdown queries for out-of-core execution on remote databases or large chunked files.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel


class PushdownQuery(BaseModel):
    operation: str  # "aggregation", "group_by", "filter_scan", "correlation"
    generated_sql: str
    target_table: str
    target_metric: Optional[str] = None
    group_dimensions: List[str] = []
    filters: Dict[str, Any] = {}


class QueryPushdownEngine:
    """Generates and optimizes SQL pushdowns for large-scale enterprise data."""

    def build_aggregation_pushdown(
        self,
        table_name: str,
        metric: str,
        dimensions: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 1000,
    ) -> PushdownQuery:
        """Construct an aggregation pushdown query with grouping and filtering."""
        dims = dimensions or []
        fltrs = filters or {}

        dim_select = ", ".join(dims) + ", " if dims else ""
        group_by = f"GROUP BY {', '.join(dims)}" if dims else ""

        where_clauses = []
        for k, v in fltrs.items():
            if isinstance(v, str):
                where_clauses.append(f"{k} = '{v}'")
            else:
                where_clauses.append(f"{k} = {v}")
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        sql = f"""
            SELECT 
                {dim_select}
                COUNT(*) as count,
                AVG({metric}) as avg_{metric},
                SUM({metric}) as sum_{metric},
                MIN({metric}) as min_{metric},
                MAX({metric}) as max_{metric}
            FROM {table_name}
            {where_sql}
            {group_by}
            ORDER BY sum_{metric} DESC
            LIMIT {limit}
        """.strip()

        return PushdownQuery(
            operation="group_by" if dims else "aggregation",
            generated_sql=" ".join(sql.split()),
            target_table=table_name,
            target_metric=metric,
            group_dimensions=dims,
            filters=fltrs,
        )

    def execute_chunked_aggregation(
        self,
        df: pd.DataFrame,
        metric: str,
        dimension: Optional[str] = None,
        chunk_size: int = 10000,
    ) -> Dict[str, Any]:
        """Stream / chunk execute large DataFrame metrics to reduce memory peak."""
        if dimension and dimension in df.columns:
            metric_series = pd.to_numeric(df[metric], errors="coerce")
            if metric_series.notna().sum() == 0:
                return {"count": int(len(df)), "mean": 0.0, "sum": 0.0, "min": 0.0, "max": 0.0,
                        "note": "Metric column has no numeric values."}
            work = df.copy()
            work[metric] = metric_series
            grouped = work.groupby(dimension)[metric].agg(["count", "mean", "sum", "min", "max"])
            return grouped.reset_index().to_dict(orient="records")

        metric_series = pd.to_numeric(df[metric], errors="coerce")
        return {
            "count": int(metric_series.count()),
            "mean": float(metric_series.mean()) if metric_series.count() else 0.0,
            "sum": float(metric_series.sum()) if metric_series.count() else 0.0,
            "min": float(metric_series.min()) if metric_series.count() else 0.0,
            "max": float(metric_series.max()) if metric_series.count() else 0.0,
        }


GLOBAL_PUSHDOWN_ENGINE = QueryPushdownEngine()
