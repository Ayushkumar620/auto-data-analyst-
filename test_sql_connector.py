"""Tests for Live Enterprise SQL Database Connector & Schema Introspection Engine.

Verifies:
1. LiveSQLDatabaseConnector schema introspection (tables, columns, types, PKs, FKs)
2. Automated Relational Foreign Key Graph construction
3. Smart SQL Join query synthesis
4. Safe read-only execution with destructive query blocking
5. UniversalDatasetLoader multi-table database ingestion and orchestrator integration
"""
import sqlite3
import pandas as pd
import pytest

from backend.app.core.sql_connector import (
    LiveSQLDatabaseConnector,
    RelationalSchemaGraph,
    TableMetadata,
)
from backend.app.core.universal_loader import UniversalDatasetLoader
from agent.command_orchestrator import AutonomousCommandOrchestrator


@pytest.fixture
def sample_sqlite_db(tmp_path):
    """Create a sample relational SQLite database with Customers, Orders, and OrderItems."""
    db_path = str(tmp_path / "enterprise_crm.sqlite")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Enable FKs
    cur.execute("PRAGMA foreign_keys = ON;")

    # 1. Customers Table
    cur.execute("""
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        country TEXT NOT NULL,
        segment TEXT NOT NULL
    );
    """)

    # 2. Orders Table
    cur.execute("""
    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        order_date TEXT NOT NULL,
        total_amount REAL NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );
    """)

    # 3. Order Items Table
    cur.execute("""
    CREATE TABLE order_items (
        item_id INTEGER PRIMARY KEY,
        order_id INTEGER NOT NULL,
        product TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        price REAL NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    );
    """)

    # Insert sample data
    cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?);", [
        (1, "Alice Smith", "USA", "Enterprise"),
        (2, "Bob Jones", "Germany", "SMB"),
        (3, "Charlie Brown", "Japan", "Consumer"),
    ])

    cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?);", [
        (101, 1, "2025-01-10", 1500.0),
        (102, 1, "2025-02-15", 2500.0),
        (103, 2, "2025-01-20", 800.0),
        (104, 3, "2025-03-05", 300.0),
    ])

    cur.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?);", [
        (1, 101, "Server Hardware", 1, 1500.0),
        (2, 102, "Cloud License", 5, 500.0),
        (3, 103, "Support Plan", 2, 400.0),
        (4, 104, "Mouse & Keyboard", 3, 100.0),
    ])

    conn.commit()
    conn.close()
    return db_path


def test_sql_schema_introspection(sample_sqlite_db):
    """Verify discovery of tables, columns, primary keys, and foreign keys."""
    connector = LiveSQLDatabaseConnector(connection_uri=sample_sqlite_db)
    graph: RelationalSchemaGraph = connector.introspect_schema()

    assert isinstance(graph, RelationalSchemaGraph)
    assert graph.total_tables == 3
    assert "customers" in graph.tables
    assert "orders" in graph.tables
    assert "order_items" in graph.tables

    # Check Customers metadata
    cust = graph.tables["customers"]
    assert cust.primary_keys == ["customer_id"]
    assert cust.row_count == 3

    # Check Orders FK metadata
    ord_meta = graph.tables["orders"]
    assert len(ord_meta.foreign_keys) >= 1
    assert ord_meta.foreign_keys[0]["to_table"] == "customers"
    assert ord_meta.foreign_keys[0]["from_column"] == "customer_id"


def test_smart_join_query_generation(sample_sqlite_db):
    """Verify smart SQL join generation using discovered FK graph."""
    connector = LiveSQLDatabaseConnector(connection_uri=sample_sqlite_db)
    join_sql = connector.generate_smart_join_query("orders", "customers")

    assert "JOIN" in join_sql
    assert "customers" in join_sql
    assert "orders" in join_sql
    assert "customer_id" in join_sql


def test_safe_query_execution_and_guardrails(sample_sqlite_db):
    """Verify read-only query execution and protection against destructive commands."""
    connector = LiveSQLDatabaseConnector(connection_uri=sample_sqlite_db)

    # Valid SELECT query
    df, duration = connector.execute_query("SELECT customer_id, name, country FROM customers")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert duration >= 0

    # Destructive queries must be rejected
    with pytest.raises(PermissionError):
        connector.execute_query("DROP TABLE customers")

    with pytest.raises(PermissionError):
        connector.execute_query("DELETE FROM orders WHERE order_id = 101")


def test_universal_loader_and_orchestrator_integration(sample_sqlite_db):
    """Verify UniversalDatasetLoader.load_database_uri and auto-join execution."""
    tables_dict, mem_profile, schema_graph = UniversalDatasetLoader.load_database_uri(sample_sqlite_db)

    assert len(tables_dict) == 3
    assert "customers" in tables_dict
    assert "orders" in tables_dict

    # Execute orchestrator on multi-table relational dataset
    orchestrator = AutonomousCommandOrchestrator()
    res = orchestrator.execute_command(
        command="Find total revenue by country",
        dataframe=tables_dict,
        session_id="sql_e2e_sess",
    )

    assert res.user_intent in ("eda", "segmentation", "ranking")
    assert len(res.execution_steps) >= 1
    assert res.execution_graph is not None
