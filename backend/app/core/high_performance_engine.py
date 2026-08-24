"""High-Performance Analytical Execution Layer (DuckDB / Polars / Vectorized NumPy).

Provides sub-second analytical querying, vectorized aggregations, multi-column group-bys,
SQL query execution, and high-speed statistical profiling for datasets ranging from
thousands to millions of records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import sqlite3
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Check optional engine accelerators
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


@dataclass
class AggregationResult:
    """Standardized output of a high-performance aggregation execution."""
    data: pd.DataFrame
    engine_used: str
    duration_ms: float
    rows_processed: int
    group_columns: List[str]
    metrics_computed: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_used": self.engine_used,
            "duration_ms": round(float(self.duration_ms), 3),
            "rows_processed": self.rows_processed,
            "group_columns": self.group_columns,
            "metrics_computed": self.metrics_computed,
            "record_count": len(self.data),
            "records": self.data.head(50).to_dict(orient="records"),
        }


@dataclass
class HighPerformanceStats:
    """High-speed statistical metrics computed across columns."""
    column_stats: Dict[str, Dict[str, float]]
    correlation_matrix: Dict[str, Dict[str, float]]
    quantiles: Dict[str, Dict[str, float]]
    duration_ms: float
    engine_used: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_stats": self.column_stats,
            "correlation_matrix": self.correlation_matrix,
            "quantiles": self.quantiles,
            "duration_ms": round(float(self.duration_ms), 3),
            "engine_used": self.engine_used,
        }


class HighPerformanceExecutionEngine:
    """Ultra-fast analytical computation engine with DuckDB, Polars, and Vectorized NumPy fallbacks."""

    def __init__(self, prefer_engine: str = "auto"):
        self.prefer_engine = prefer_engine

    def _determine_active_engine(self) -> str:
        if self.prefer_engine == "duckdb" and HAS_DUCKDB:
            return "duckdb"
        if self.prefer_engine == "polars" and HAS_POLARS:
            return "polars"
        if HAS_DUCKDB:
            return "duckdb"
        if HAS_POLARS:
            return "polars"
        return "vectorized_numpy"

    def aggregate(
        self,
        df: pd.DataFrame,
        group_by: Union[str, List[str]],
        aggregations: Dict[str, Union[str, List[str]]],
        having: Optional[str] = None,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        limit: Optional[int] = None,
    ) -> AggregationResult:
        """
        Execute multi-column grouped aggregations with sub-second latency.
        
        Args:
            df: Target pandas DataFrame.
            group_by: Column or list of columns to group by.
            aggregations: Dict mapping metric column to aggregation function(s) ('sum', 'mean', 'count', 'min', 'max', 'std').
            having: Optional filter predicate on aggregated results.
            sort_by: Column to sort by.
            ascending: Sort direction.
            limit: Top N rows to retain.
        """
        start_t = time.time()
        engine = self._determine_active_engine()
        group_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        rows_count = len(df)

        if engine == "duckdb" and HAS_DUCKDB:
            res_df = self._aggregate_duckdb(df, group_cols, aggregations, sort_by, ascending, limit)
        elif engine == "polars" and HAS_POLARS:
            res_df = self._aggregate_polars(df, group_cols, aggregations, sort_by, ascending, limit)
        else:
            engine = "vectorized_numpy"
            res_df = self._aggregate_numpy(df, group_cols, aggregations, sort_by, ascending, limit)

        duration = (time.time() - start_t) * 1000

        metrics_list = []
        for col, funcs in aggregations.items():
            f_list = [funcs] if isinstance(funcs, str) else funcs
            for f in f_list:
                metrics_list.append(f"{col}_{f}")

        return AggregationResult(
            data=res_df,
            engine_used=engine,
            duration_ms=duration,
            rows_processed=rows_count,
            group_columns=group_cols,
            metrics_computed=metrics_list,
        )

    def execute_sql(
        self,
        query: str,
        tables: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Execute an analytical SQL query against in-memory DataFrame tables.
        Uses DuckDB (zero-copy) if available, or SQLite in-memory virtual database as reliable fallback.
        """
        if HAS_DUCKDB:
            con = duckdb.connect(database=":memory:")
            for name, df in tables.items():
                con.register(name, df)
            res_df = con.execute(query).df()
            con.close()
            return res_df

        # Fallback: SQLite in-memory engine
        con = sqlite3.connect(":memory:")
        for name, df in tables.items():
            df.to_sql(name, con, index=False, if_exists="replace")
        res_df = pd.read_sql_query(query, con)
        con.close()
        return res_df

    def compute_fast_statistics(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
    ) -> HighPerformanceStats:
        """Compute parallel descriptive statistics, correlation matrix, and quantiles."""
        start_t = time.time()
        engine = self._determine_active_engine()

        if numeric_columns is None:
            num_df = df.select_dtypes(include=[np.number])
        else:
            valid_cols = [c for c in numeric_columns if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
            num_df = df[valid_cols]

        if num_df.empty:
            return HighPerformanceStats(
                column_stats={},
                correlation_matrix={},
                quantiles={},
                duration_ms=(time.time() - start_t) * 1000,
                engine_used=engine,
            )

        # 1. Vectorized Column Statistics
        stats_dict = {}
        for col in num_df.columns:
            series = num_df[col].dropna()
            if len(series) > 0:
                stats_dict[col] = {
                    "count": int(len(series)),
                    "mean": float(series.mean()),
                    "std": float(series.std()) if len(series) > 1 else 0.0,
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                    "sum": float(series.sum()),
                }

        # 2. Fast Pearson Correlation Matrix
        corr_matrix = {}
        if len(num_df.columns) >= 2:
            corr_df = num_df.corr(method="pearson").fillna(0.0)
            corr_matrix = corr_df.to_dict()

        # 3. Fast Quantiles (p10, p25, p50, p75, p90, p99)
        quantiles_dict = {}
        for col in num_df.columns:
            series = num_df[col].dropna()
            if len(series) > 0:
                q_vals = np.percentile(series, [10, 25, 50, 75, 90, 99])
                quantiles_dict[col] = {
                    "p10": float(q_vals[0]),
                    "p25": float(q_vals[1]),
                    "p50": float(q_vals[2]),
                    "p75": float(q_vals[3]),
                    "p90": float(q_vals[4]),
                    "p99": float(q_vals[5]),
                }

        duration = (time.time() - start_t) * 1000

        return HighPerformanceStats(
            column_stats=stats_dict,
            correlation_matrix=corr_matrix,
            quantiles=quantiles_dict,
            duration_ms=duration,
            engine_used=engine,
        )

    # ------------------------------------------------------------------
    # Engine-Specific Implementations
    # ------------------------------------------------------------------
    def _aggregate_duckdb(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        aggregations: Dict[str, Any],
        sort_by: Optional[str],
        ascending: bool,
        limit: Optional[int],
    ) -> pd.DataFrame:
        con = duckdb.connect(database=":memory:")
        con.register("input_data", df)

        select_parts = [f'"{c}"' for c in group_cols]
        for col, funcs in aggregations.items():
            f_list = [funcs] if isinstance(funcs, str) else funcs
            for f in f_list:
                f_upper = f.upper()
                if f_upper == "MEAN":
                    f_upper = "AVG"
                select_parts.append(f'{f_upper}("{col}") AS "{col}_{f.lower()}"')

        group_str = ", ".join(f'"{c}"' for c in group_cols)
        select_str = ", ".join(select_parts)
        query = f"SELECT {select_str} FROM input_data GROUP BY {group_str}"

        if sort_by:
            dir_str = "ASC" if ascending else "DESC"
            query += f' ORDER BY "{sort_by}" {dir_str}'

        if limit:
            query += f" LIMIT {int(limit)}"

        res = con.execute(query).df()
        con.close()
        return res

    def _aggregate_polars(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        aggregations: Dict[str, Any],
        sort_by: Optional[str],
        ascending: bool,
        limit: Optional[int],
    ) -> pd.DataFrame:
        pldf = pl.from_pandas(df)
        exprs = []
        for col, funcs in aggregations.items():
            f_list = [funcs] if isinstance(funcs, str) else funcs
            for f in f_list:
                fl = f.lower()
                if fl == "sum":
                    exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
                elif fl in ("mean", "avg"):
                    exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
                elif fl == "count":
                    exprs.append(pl.col(col).count().alias(f"{col}_count"))
                elif fl == "min":
                    exprs.append(pl.col(col).min().alias(f"{col}_min"))
                elif fl == "max":
                    exprs.append(pl.col(col).max().alias(f"{col}_max"))
                elif fl == "std":
                    exprs.append(pl.col(col).std().alias(f"{col}_std"))

        res_pl = pldf.group_by(group_cols).agg(exprs)
        if sort_by:
            res_pl = res_pl.sort(sort_by, descending=not ascending)
        if limit:
            res_pl = res_pl.head(limit)

        return res_pl.to_pandas()

    def _aggregate_numpy(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        aggregations: Dict[str, Any],
        sort_by: Optional[str],
        ascending: bool,
        limit: Optional[int],
    ) -> pd.DataFrame:
        agg_map = {}
        for col, funcs in aggregations.items():
            if col in df.columns:
                agg_map[col] = funcs

        res = df.groupby(group_cols, as_index=False, observed=True).agg(agg_map)

        # Flatten multi-index columns if created
        if isinstance(res.columns, pd.MultiIndex):
            res.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in res.columns]
        else:
            # Rename columns to standard col_metric format
            new_cols = []
            for c in res.columns:
                if c in agg_map:
                    f = agg_map[c]
                    f_name = f if isinstance(f, str) else "_".join(f)
                    new_cols.append(f"{c}_{f_name}")
                else:
                    new_cols.append(c)
            res.columns = new_cols

        if sort_by and sort_by in res.columns:
            res = res.sort_values(by=sort_by, ascending=ascending)
        elif len(res.columns) > len(group_cols):
            # Sort by first metric by default
            metric_col = [c for c in res.columns if c not in group_cols][0]
            res = res.sort_values(by=metric_col, ascending=ascending)

        if limit:
            res = res.head(limit)

        return res.reset_index(drop=True)


# Global singleton high performance engine
global_high_performance_engine = HighPerformanceExecutionEngine()

