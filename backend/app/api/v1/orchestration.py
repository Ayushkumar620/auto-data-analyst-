"""
Universal Agent Orchestration FastAPI Router.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from agent.agent_result import AgentResult, AgentStatus
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


@router.post("/orchestrate")
def orchestrate_command(req: OrchestrationRequest) -> Dict[str, Any]:
    """
    Execute autonomous end-to-end multi-agent orchestration for a natural language command.
    Returns canonical AgentResult payload.
    """
    if not req.dataset or len(req.dataset) == 0:
        raise HTTPException(
            status_code=400,
            detail="Dataset records are required for orchestration. Provided dataset is empty.",
        )

    df = pd.DataFrame(req.dataset)
    orchestrator = UniversalOrchestrator()

    result: AgentResult = orchestrator.orchestrate(
        command=req.command,
        data=df,
        target=req.target,
        features=req.features,
        time_column=req.time_column,
        config=req.config,
        dataset_id=req.dataset_id,
    )

    if result.status == AgentStatus.ERROR:
        err_msg = result.error_message or "Orchestration failed."
        raise HTTPException(status_code=422, detail=err_msg)

    return result.to_dict()