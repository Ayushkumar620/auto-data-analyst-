"""
Unit and Integration Tests for Enterprise Live Database Connectors.
"""
import pytest
from agent.connectors import (
    DBConnectionConfig,
    EnterpriseConnectorManager,
    GLOBAL_CONNECTOR_MANAGER,
)


def test_sqlite_in_memory_connection_test():
    manager = EnterpriseConnectorManager()
    cfg = DBConnectionConfig(
        name="Test In-Memory DB",
        db_type="sqlite",
        database=":memory:",
    )
    success, msg = manager.test_connection(cfg)
    assert success is True
    assert "Connection successful" in msg


def test_table_inspection_and_query_execution():
    manager = EnterpriseConnectorManager()
    # The default demo sqlite database has customer_transactions table
    tables = manager.inspect_tables("conn_demo_sqlite")
    table_names = [t.table_name for t in tables]
    assert "customer_transactions" in table_names

    # Test query execution
    res = manager.execute_query(
        connection_id="conn_demo_sqlite",
        sql_query="SELECT region, SUM(amount) as total_revenue FROM customer_transactions GROUP BY region",
    )
    assert res.total_rows > 0
    assert "region" in res.columns
    assert "total_revenue" in res.columns
    assert res.execution_time_ms >= 0.0


def test_destructive_query_rejection():
    manager = EnterpriseConnectorManager()
    with pytest.raises(ValueError, match="Destructive statement"):
        manager.execute_query(
            connection_id="conn_demo_sqlite",
            sql_query="DROP TABLE customer_transactions",
        )


def test_ingest_to_dataframe():
    manager = EnterpriseConnectorManager()
    df = manager.ingest_to_dataframe("conn_demo_sqlite", "SELECT * FROM customer_transactions")
    assert len(df) >= 5
    assert "customer_id" in df.columns

