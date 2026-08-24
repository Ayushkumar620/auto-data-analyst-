"""Live SQL Database Gateway FastAPI Router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from backend.app.core.sql_connector import LiveSQLDatabaseConnector, RelationalSchemaGraph

router = APIRouter(prefix="/sql", tags=["Enterprise SQL"])


class SQLIntrospectRequest(BaseModel):
    connection_uri: str = Field(..., description="SQLAlchemy connection URI or sqlite path")


class SQLQueryRequest(BaseModel):
    connection_uri: str = Field(..., description="SQLAlchemy connection URI or sqlite path")
    query: str = Field(..., description="Read-only SQL query to execute")
    max_rows: Optional[int] = Field(5000, description="Max rows to return")


class SmartJoinRequest(BaseModel):
    connection_uri: str
    table_a: str
    table_b: str


@router.post("/introspect")
def introspect_database_schema(req: SQLIntrospectRequest) -> Dict[str, Any]:
    """Introspect all tables, column datatypes, primary keys, and foreign keys."""
    try:
        connector = LiveSQLDatabaseConnector(connection_uri=req.connection_uri)
        graph: RelationalSchemaGraph = connector.introspect_schema()
        return graph.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database introspection failed: {str(e)}")


@router.post("/query")
def execute_sql_query(req: SQLQueryRequest) -> Dict[str, Any]:
    """Safely execute a read-only SQL query against the target database."""
    try:
        connector = LiveSQLDatabaseConnector(connection_uri=req.connection_uri, read_only=True)
        df, duration_ms = connector.execute_query(query=req.query, max_rows=req.max_rows)
        return {
            "query": req.query,
            "rows_returned": len(df),
            "columns": list(df.columns),
            "records": df.to_dict(orient="records"),
            "duration_ms": round(float(duration_ms), 2),
        }
    except PermissionError as p_err:
        raise HTTPException(status_code=403, detail=str(p_err))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL execution failed: {str(e)}")


@router.post("/smart-join")
def synthesize_smart_join(req: SmartJoinRequest) -> Dict[str, str]:
    """Automatically synthesize an SQL join query between two related tables."""
    try:
        connector = LiveSQLDatabaseConnector(connection_uri=req.connection_uri)
        join_sql = connector.generate_smart_join_query(table_a=req.table_a, table_b=req.table_b)
        return {"smart_join_query": join_sql}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Smart join synthesis failed: {str(e)}")
