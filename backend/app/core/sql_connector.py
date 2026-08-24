"""Live Enterprise SQL Database Connector & Multi-Table Schema Introspection Engine.

Provides live connectivity to relational databases (SQLite, PostgreSQL, MySQL, DuckDB),
automated schema introspection (tables, columns, types, PKs, FKs), automated foreign key
join graph construction, safe read-only SQL query execution, and zero-copy tabular ingestion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


@dataclass
class ColumnMetadata:
    """Introspected metadata for a database column."""
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool
    foreign_key_target: Optional[str] = None  # Format: "target_table.target_column"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": str(self.data_type),
            "is_nullable": self.is_nullable,
            "is_primary_key": self.is_primary_key,
            "foreign_key_target": self.foreign_key_target,
        }


@dataclass
class TableMetadata:
    """Introspected metadata for a database table."""
    table_name: str
    row_count: int
    columns: List[ColumnMetadata]
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "row_count": self.row_count,
            "columns": [c.to_dict() for c in self.columns],
            "primary_keys": self.primary_keys,
            "foreign_keys": self.foreign_keys,
        }


@dataclass
class RelationalSchemaGraph:
    """Complete relational schema graph with discovered tables and join relationships."""
    database_type: str
    tables: Dict[str, TableMetadata]
    relationships: List[Dict[str, str]]  # List of {from_table, from_col, to_table, to_col}
    total_tables: int
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "database_type": self.database_type,
            "tables": {k: v.to_dict() for k, v in self.tables.items()},
            "relationships": self.relationships,
            "total_tables": self.total_tables,
            "duration_ms": round(float(self.duration_ms), 3),
        }


class LiveSQLDatabaseConnector:
    """Connects to SQL databases, introspects multi-table schemas, and safely executes queries."""

    FORBIDDEN_SQL_KEYWORDS = re.compile(
        r"\b(DROP|DELETE|TRUNCATE|ALTER|UPDATE|INSERT|GRANT|REVOKE|EXEC|EXECUTE)\b",
        re.I
    )

    def __init__(self, connection_uri: str, read_only: bool = True):
        self.connection_uri = connection_uri
        self.read_only = read_only
        self._engine: Optional[Engine] = None

    def _get_engine(self) -> Engine:
        if self._engine is None:
            # Handle sqlite relative vs memory URIs
            uri = self.connection_uri
            if not uri.startswith(("sqlite:", "postgresql:", "mysql:", "duckdb:")):
                if uri.endswith((".db", ".sqlite", ".sqlite3")) or uri == ":memory:":
                    uri = f"sqlite:///{uri}"
            self._engine = create_engine(uri)
        return self._engine

    def introspect_schema(self) -> RelationalSchemaGraph:
        """Introspect all tables, column datatypes, primary keys, and foreign key relationships."""
        start_t = time.time()
        engine = self._get_engine()
        inspector = inspect(engine)
        db_type = engine.name

        table_names = inspector.get_table_names()
        tables_dict: Dict[str, TableMetadata] = {}
        relationships: List[Dict[str, str]] = []

        for t_name in table_names:
            pk_constraint = inspector.get_pk_constraint(t_name)
            pks = pk_constraint.get("constrained_columns", [])

            fk_constraints = inspector.get_foreign_keys(t_name)
            fks_list = []
            fk_col_map: Dict[str, str] = {}

            for fk in fk_constraints:
                referred_table = fk.get("referred_table")
                constrained_cols = fk.get("constrained_columns", [])
                referred_cols = fk.get("referred_columns", [])

                for c_col, r_col in zip(constrained_cols, referred_cols):
                    fk_target = f"{referred_table}.{r_col}"
                    fk_col_map[c_col] = fk_target
                    fks_list.append({
                        "from_table": t_name,
                        "from_column": c_col,
                        "to_table": referred_table,
                        "to_column": r_col,
                    })
                    relationships.append({
                        "from_table": t_name,
                        "from_column": c_col,
                        "to_table": referred_table,
                        "to_column": r_col,
                    })

            # Get columns
            cols_info = inspector.get_columns(t_name)
            columns_list: List[ColumnMetadata] = []
            for col in cols_info:
                c_name = col.get("name")
                c_type = str(col.get("type"))
                c_nullable = bool(col.get("nullable", True))
                is_pk = c_name in pks
                fk_target = fk_col_map.get(c_name)

                columns_list.append(ColumnMetadata(
                    name=c_name,
                    data_type=c_type,
                    is_nullable=c_nullable,
                    is_primary_key=is_pk,
                    foreign_key_target=fk_target,
                ))

            # Row count
            try:
                with engine.connect() as conn:
                    count_res = conn.execute(text(f'SELECT COUNT(*) FROM "{t_name}"')).scalar()
                    row_count = int(count_res or 0)
            except Exception:
                row_count = 0

            tables_dict[t_name] = TableMetadata(
                table_name=t_name,
                row_count=row_count,
                columns=columns_list,
                primary_keys=pks,
                foreign_keys=fks_list,
            )

        duration = (time.time() - start_t) * 1000

        return RelationalSchemaGraph(
            database_type=db_type,
            tables=tables_dict,
            relationships=relationships,
            total_tables=len(tables_dict),
            duration_ms=duration,
        )

    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        max_rows: Optional[int] = 50000,
    ) -> Tuple[pd.DataFrame, float]:
        """
        Safely execute a read-only SQL query against the database.
        
        Returns:
            Tuple of (DataFrame, duration_in_milliseconds)
        """
        start_t = time.time()
        q_clean = query.strip()

        if self.read_only:
            # Check for destructive keywords
            first_word = q_clean.split()[0].upper()
            if first_word not in ("SELECT", "WITH", "EXPLAIN", "PRAGMA", "SHOW", "DESCRIBE"):
                raise PermissionError(f"Read-only connector prohibits query starting with '{first_word}'.")

            if self.FORBIDDEN_SQL_KEYWORDS.search(q_clean):
                # Ensure no embedded destructive commands (e.g., '; DROP TABLE')
                if any(kw in q_clean.upper() for kw in ["DROP ", "DELETE ", "TRUNCATE ", "ALTER "]):
                    raise PermissionError("Query contains destructive SQL keywords (DROP, DELETE, TRUNCATE, ALTER).")

        engine = self._get_engine()
        with engine.connect() as conn:
            stmt = text(q_clean)
            if params:
                result = conn.execute(stmt, params)
            else:
                result = conn.execute(stmt)

            if result.returns_rows:
                rows = result.fetchmany(max_rows) if max_rows else result.fetchall()
                cols = list(result.keys())
                df = pd.DataFrame(rows, columns=cols)
            else:
                df = pd.DataFrame()

        duration = (time.time() - start_t) * 1000
        return df, duration

    def load_all_tables_as_dict(self, max_rows_per_table: int = 25000) -> Dict[str, pd.DataFrame]:
        """Ingest all database tables as a dictionary of DataFrames."""
        engine = self._get_engine()
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        tables_data: Dict[str, pd.DataFrame] = {}
        for t_name in table_names:
            df, _ = self.execute_query(f'SELECT * FROM "{t_name}"', max_rows=max_rows_per_table)
            tables_data[t_name] = df

        return tables_data

    def generate_smart_join_query(
        self,
        table_a: str,
        table_b: str,
        schema: Optional[RelationalSchemaGraph] = None,
    ) -> str:
        """Automatically synthesize an SQL JOIN query between two related tables."""
        if schema is None:
            schema = self.introspect_schema()

        # Search direct FK relationships
        for rel in schema.relationships:
            if rel["from_table"] == table_a and rel["to_table"] == table_b:
                return (
                    f'SELECT a.*, b.*\n'
                    f'FROM "{table_a}" a\n'
                    f'JOIN "{table_b}" b ON a."{rel["from_column"]}" = b."{rel["to_column"]}"'
                )
            if rel["from_table"] == table_b and rel["to_table"] == table_a:
                return (
                    f'SELECT a.*, b.*\n'
                    f'FROM "{table_a}" a\n'
                    f'JOIN "{table_b}" b ON a."{rel["to_column"]}" = b."{rel["from_column"]}"'
                )

        # Fallback: Match column names that match across tables (e.g. customer_id, id)
        t_a_cols = [c.name for c in schema.tables.get(table_a, TableMetadata(table_a, 0, [], [], [])).columns]
        t_b_cols = [c.name for c in schema.tables.get(table_b, TableMetadata(table_b, 0, [], [], [])).columns]

        common = [c for c in t_a_cols if c in t_b_cols]
        if common:
            join_col = common[0]
            return (
                f'SELECT a.*, b.*\n'
                f'FROM "{table_a}" a\n'
                f'JOIN "{table_b}" b ON a."{join_col}" = b."{join_col}"'
            )

        # Default cross / simple select
        return f'SELECT * FROM "{table_a}"'
