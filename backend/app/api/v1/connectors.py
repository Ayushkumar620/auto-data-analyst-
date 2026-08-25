"""
FastAPI REST Router for Enterprise Live Database Connectors.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from agent.connectors import (
    DBConnectionConfig,
    EnterpriseConnectorManager,
    GLOBAL_CONNECTOR_MANAGER,
    QueryResult,
    TableSchemaInfo,
)

router = APIRouter(prefix="/connectors", tags=["Enterprise Database Connectors"])


class TestConnectionRequest(BaseModel):
    name: str = "Test DB"
    db_type: str = "sqlite"
    host: Optional[str] = "localhost"
    port: Optional[int] = 5432
    database: str = ":memory:"
    username: Optional[str] = None
    password: Optional[str] = None
    custom_url: Optional[str] = None


class RunQueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 500


class IngestDatasetRequest(BaseModel):
    query: str
    dataset_name: str
    limit: Optional[int] = 50000


@router.get("", response_model=List[DBConnectionConfig])
def list_database_connections():
    """List all registered database connections."""
    return GLOBAL_CONNECTOR_MANAGER.list_connections()


@router.post("/test")
def test_database_connection(req: TestConnectionRequest):
    """Test database connectivity with provided credentials."""
    cfg = DBConnectionConfig(
        name=req.name,
        db_type=req.db_type,
        host=req.host,
        port=req.port,
        database=req.database,
        username=req.username,
        password=req.password,
        custom_url=req.custom_url,
    )
    success, message = GLOBAL_CONNECTOR_MANAGER.test_connection(cfg)
    return {"success": success, "message": message}


@router.post("/create", response_model=DBConnectionConfig)
def create_database_connection(req: TestConnectionRequest):
    """Save and register a verified database connection profile."""
    cfg = DBConnectionConfig(
        name=req.name,
        db_type=req.db_type,
        host=req.host,
        port=req.port,
        database=req.database,
        username=req.username,
        password=req.password,
        custom_url=req.custom_url,
    )
    success, message = GLOBAL_CONNECTOR_MANAGER.test_connection(cfg)
    if not success:
        raise HTTPException(status_code=400, detail=f"Cannot save invalid connection: {message}")
    return GLOBAL_CONNECTOR_MANAGER.register_connection(cfg)


@router.get("/{connection_id}/tables", response_model=List[TableSchemaInfo])
def inspect_connection_tables(connection_id: str):
    """Inspect tables and schemas available on the remote database."""
    try:
        return GLOBAL_CONNECTOR_MANAGER.inspect_tables(connection_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{connection_id}/query", response_model=QueryResult)
def execute_sql_query(connection_id: str, req: RunQueryRequest):
    """Execute a read-only SQL query with limit guardrails."""
    try:
        return GLOBAL_CONNECTOR_MANAGER.execute_query(connection_id, req.query, limit=req.limit or 500)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{connection_id}")
def delete_database_connection(connection_id: str):
    """Delete a database connection profile."""
    deleted = GLOBAL_CONNECTOR_MANAGER.delete_connection(connection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Connection ID not found")
    return {"success": True, "deleted_id": connection_id}
