"""
Enterprise Live Database Connectors Engine.

Supports PostgreSQL, MySQL, SQLite, Snowflake, and generic SQLAlchemy database URLs.
Provides connection pooling, test validation, schema/table reflection, parameterized queries,
and direct ingestion into the Auto Data Analyst dataset format.
"""
from __future__ import annotations

import os
import uuid
import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, Field
import sqlalchemy
from sqlalchemy import create_engine, inspect, text


class DBConnectionConfig(BaseModel):
    """Configuration for an enterprise database connection."""
    connection_id: str = Field(default_factory=lambda: f"conn_{uuid.uuid4().hex[:8]}")
    name: str
    db_type: str  # "postgresql", "mysql", "sqlite", "snowflake", "generic"
    host: Optional[str] = "localhost"
    port: Optional[int] = 5432
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    ssl_mode: Optional[str] = "prefer"
    custom_url: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())

    def get_sqlalchemy_url(self) -> str:
        """Construct a standardized SQLAlchemy connection URL."""
        if self.custom_url:
            return self.custom_url
        if self.db_type == "sqlite":
            return f"sqlite:///{self.database}"
        if self.db_type == "postgresql":
            pwd = f":{self.password}" if self.password else ""
            user = self.username or "postgres"
            return f"postgresql+psycopg2://{user}{pwd}@{self.host}:{self.port}/{self.database}"
        if self.db_type == "mysql":
            pwd = f":{self.password}" if self.password else ""
            user = self.username or "root"
            return f"mysql+pymysql://{user}{pwd}@{self.host}:{self.port}/{self.database}"
        if self.db_type == "snowflake":
            pwd = f":{self.password}" if self.password else ""
            return f"snowflake://{self.username}{pwd}@{self.host}/{self.database}"
        return f"sqlite:///{self.database}"


class TableSchemaInfo(BaseModel):
    """Metadata schema for a remote table."""
    table_name: str
    row_count_estimate: Optional[int] = None
    columns: List[Dict[str, str]] = []  # [{"name": "col", "type": "INTEGER"}]


class QueryResult(BaseModel):
    """Result of a live database query execution."""
    columns: List[str]
    rows: List[Dict[str, Any]]
    total_rows: int
    execution_time_ms: float
    query: str


from sqlalchemy.pool import StaticPool


class EnterpriseConnectorManager:
    """Manages active database connections and execution pool."""

    def __init__(self):
        self._connections: Dict[str, DBConnectionConfig] = {}
        self._engines: Dict[str, Any] = {}
        # Pre-seed with an in-memory SQLite demo connection for immediate out-of-the-box usage
        demo_conn = DBConnectionConfig(
            connection_id="conn_demo_sqlite",
            name="Demo SQLite Database",
            db_type="sqlite",
            database=":memory:",
        )
        self.register_connection(demo_conn)
        self._init_demo_database(demo_conn)

    def _get_engine(self, conn_config: DBConnectionConfig):
        cid = conn_config.connection_id
        if cid in self._engines:
            return self._engines[cid]

        url = conn_config.get_sqlalchemy_url()
        if conn_config.db_type == "sqlite" and conn_config.database == ":memory:":
            engine = create_engine(
                url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            engine = create_engine(url, pool_pre_ping=True)

        self._engines[cid] = engine
        return engine

    def _init_demo_database(self, conn: DBConnectionConfig):
        """Seed demo tables inside the in-memory demo connection."""
        try:
            engine = self._get_engine(conn)
            with engine.connect() as c:
                c.execute(text("""
                    CREATE TABLE IF NOT EXISTS customer_transactions (
                        transaction_id INTEGER PRIMARY KEY,
                        customer_id TEXT,
                        region TEXT,
                        amount REAL,
                        category TEXT,
                        created_at TEXT
                    )
                """))
                # Insert seed rows if empty
                res = c.execute(text("SELECT COUNT(*) FROM customer_transactions")).scalar()
                if res == 0:
                    c.execute(text("""
                        INSERT INTO customer_transactions (transaction_id, customer_id, region, amount, category, created_at)
                        VALUES 
                        (101, 'CUST-001', 'North', 450.00, 'Enterprise', '2026-01-15'),
                        (102, 'CUST-002', 'South', 120.50, 'Retail', '2026-01-16'),
                        (103, 'CUST-003', 'North', 890.00, 'Enterprise', '2026-01-18'),
                        (104, 'CUST-004', 'West', 310.25, 'Mid-Market', '2026-01-20'),
                        (105, 'CUST-001', 'North', 620.00, 'Enterprise', '2026-01-22')
                    """))
                    c.commit()
        except Exception:
            pass

    def register_connection(self, config: DBConnectionConfig) -> DBConnectionConfig:
        self._connections[config.connection_id] = config
        return config

    def list_connections(self) -> List[DBConnectionConfig]:
        return list(self._connections.values())

    def get_connection(self, connection_id: str) -> Optional[DBConnectionConfig]:
        return self._connections.get(connection_id)

    def delete_connection(self, connection_id: str) -> bool:
        if connection_id in self._connections:
            del self._connections[connection_id]
            if connection_id in self._engines:
                del self._engines[connection_id]
            return True
        return False

    def test_connection(self, config: DBConnectionConfig) -> Tuple[bool, str]:
        """Test database connection validity."""
        try:
            url = config.get_sqlalchemy_url()
            if config.db_type == "sqlite" and config.database == ":memory:":
                engine = create_engine(url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
            else:
                engine = create_engine(url, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, "Connection successful"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def inspect_tables(self, connection_id: str) -> List[TableSchemaInfo]:
        """Inspect and return remote tables and their column definitions."""
        conn_config = self.get_connection(connection_id)
        if not conn_config:
            raise ValueError(f"Connection ID {connection_id} not found")

        engine = self._get_engine(conn_config)
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        tables = []
        for name in table_names:
            cols = []
            for col in inspector.get_columns(name):
                cols.append({"name": col["name"], "type": str(col["type"])})
            tables.append(TableSchemaInfo(table_name=name, columns=cols))
        return tables

    def execute_query(self, connection_id: str, sql_query: str, limit: int = 500) -> QueryResult:
        """Safely execute a SELECT query and return structured results."""
        conn_config = self.get_connection(connection_id)
        if not conn_config:
            raise ValueError(f"Connection ID {connection_id} not found")

        # Sanitize against destructive statements for read safety
        cleaned_sql = sql_query.strip().rstrip(";")
        lowered = cleaned_sql.lower()
        destructive = ["drop ", "delete ", "truncate ", "alter ", "update ", "insert "]
        for d in destructive:
            if d in lowered:
                raise ValueError(f"Destructive statement '{d.strip()}' not permitted in analytical queries")

        start_time = datetime.datetime.utcnow()
        engine = self._get_engine(conn_config)

        # Append LIMIT if not present
        if "limit " not in lowered:
            cleaned_sql = f"{cleaned_sql} LIMIT {limit}"

        df = pd.read_sql_query(cleaned_sql, engine)
        exec_ms = (datetime.datetime.utcnow() - start_time).total_seconds() * 1000.0

        return QueryResult(
            columns=df.columns.tolist(),
            rows=df.to_dict(orient="records"),
            total_rows=len(df),
            execution_time_ms=round(exec_ms, 2),
            query=cleaned_sql,
        )

    def ingest_to_dataframe(self, connection_id: str, sql_query: str, limit: int = 50000) -> pd.DataFrame:
        """Execute query and return full pandas DataFrame for dataset analysis."""
        conn_config = self.get_connection(connection_id)
        if not conn_config:
            raise ValueError(f"Connection ID {connection_id} not found")

        engine = self._get_engine(conn_config)
        return pd.read_sql_query(sql_query, engine)


# Global singleton instance
GLOBAL_CONNECTOR_MANAGER = EnterpriseConnectorManager()
