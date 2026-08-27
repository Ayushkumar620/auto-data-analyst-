"""
Universal Agent Orchestration FastAPI Router.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from agent.agent_result import AgentResult, AgentStatus
from agent.analytical_context import DEFAULT_SESSION_CONTEXT_MANAGER
from agent.orchestrator import UniversalOrchestrator

router = APIRouter(tags=["Universal Agent Orchestration"])


class OrchestrationRequest(BaseModel):
    dataset: Optional[List[Dict[str, Any]]] = Field(None, description="Tabular dataset records")
    command: str = Field(..., description="Natural language analytical command")
    target: Optional[str] = Field(None, description="Optional target column name")
    features: Optional[List[str]] = Field(None, description="Optional feature columns subset")
    time_column: Optional[str] = Field(None, description="Optional temporal/datetime column name")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional orchestration parameters")
    dataset_id: Optional[str] = Field(None, description="Optional dataset ID")
    session_id: Optional[str] = Field(None, description="Conversational session identifier")


@router.post("/orchestrate")
def orchestrate_command(req: OrchestrationRequest) -> Dict[str, Any]:
    """
    Execute autonomous end-to-end multi-agent orchestration for a natural language command.
    Supports session memory, follow-up reference resolution, and multi-turn workflows.
    Returns canonical AgentResult payload.
    """
    df = pd.DataFrame(req.dataset) if (req.dataset is not None and len(req.dataset) > 0) else None

    if df is None and not req.session_id:
        raise HTTPException(
            status_code=400,
            detail="Dataset records are required when session_id is not provided. Provided dataset is empty.",
        )

    orchestrator = UniversalOrchestrator()

    result: AgentResult = orchestrator.orchestrate(
        command=req.command,
        data=df,
        target=req.target,
        features=req.features,
        time_column=req.time_column,
        config=req.config,
        dataset_id=req.dataset_id,
        session_id=req.session_id,
    )

    if result.status == AgentStatus.ERROR:
        err_msg = result.error_message or "Orchestration failed."
        raise HTTPException(status_code=422, detail=err_msg)

    return result.to_dict()


@router.get("/orchestrate/context/{session_id}")
def get_session_context(session_id: str) -> Dict[str, Any]:
    """Retrieve structured analytical context for a given session."""
    ctx = DEFAULT_SESSION_CONTEXT_MANAGER.get_context(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Session context '{session_id}' not found.")
    return ctx.to_dict()


@router.delete("/orchestrate/context/{session_id}")
def clear_session_context(session_id: str) -> Dict[str, Any]:
    """Clear and invalidate session analytical context and memory."""
    DEFAULT_SESSION_CONTEXT_MANAGER.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}