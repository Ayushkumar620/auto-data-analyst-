"""Autonomous Command-Driven Analysis FastAPI Router."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from agent.command_orchestrator import (
    AutonomousCommandOrchestrator,
    CommandExecutionResult,
    global_orchestrator,
)

router = APIRouter(prefix="/analyze", tags=["Autonomous Analysis"])


class AnalyzeCommandRequest(BaseModel):
    command: str = Field(..., description="Natural language analytical command")
    dataset: Optional[List[Dict[str, Any]]] = Field(None, description="In-memory dataset records")
    session_id: Optional[str] = Field("default_session", description="Conversational session ID")


class AnalyzeCommandResponse(BaseModel):
    command: str
    user_intent: str
    required_operations: List[str]
    final_explanation: str
    execution_graph: Any
    evidence: List[Dict[str, Any]]
    dataset_summary: Dict[str, Any]
    validation_summary: Dict[str, Any]
    duration_ms: float
    relationships: Optional[List[Dict[str, Any]]] = None
    top_relationships: Optional[List[Dict[str, Any]]] = None
    subgroup_analysis: Optional[Dict[str, Any]] = None
    correlation_matrix: Optional[Dict[str, Any]] = None


@router.post("", response_model=AnalyzeCommandResponse)
def execute_autonomous_command(req: AnalyzeCommandRequest) -> AnalyzeCommandResponse:
    """Execute a natural language analytical command against a dataset."""
    if not req.command.strip():
        raise HTTPException(status_code=400, detail="Command string cannot be empty.")

    if req.dataset:
        df = pd.DataFrame(req.dataset)
    else:
        # Default sample dataset if none provided
        df = pd.DataFrame({
            "Region": ["North", "South", "East", "West"],
            "Revenue": [120000.0, 85000.0, 94000.0, 110000.0],
            "Profit": [24000.0, 15000.0, 18500.0, 22000.0],
        })

    result: CommandExecutionResult = global_orchestrator.execute_command(
        command=req.command,
        dataframe=df,
        session_id=req.session_id or "default_session",
    )

    import logging
    logger = logging.getLogger(__name__)
    rels = result.relationships or []
    first_r = rels[0] if rels else {}
    logger.info(
        "\n[STATISTICAL_RUNTIME_FORENSICS_V1] (API /api/v1/analyze)\n"
        "  received user command: %r\n"
        "  detected intent: %s\n"
        "  selected tool: %s\n"
        "  selected agent: %s\n"
        "  engine class/function executed: %s\n"
        "  returned AgentResult.task_type: %s\n"
        "  top-level AgentResult.result keys: %s\n"
        "  pearson_r exists: %s\n"
        "  spearman_rho exists: %s\n"
        "  p_value exists: %s\n"
        "  adjusted_p_value exists: %s\n"
        "  evidence exists: %s\n",
        req.command,
        result.user_intent,
        "statistical_analysis_tool",
        "AutonomousCommandOrchestrator",
        "execute_command",
        "statistical_analysis",
        list(result.to_dict().keys()),
        "pearson_r" in first_r or "pearson" in first_r,
        "spearman_rho" in first_r or "spearman" in first_r,
        "p_value" in first_r,
        "adjusted_p_value" in first_r,
        bool(result.evidence),
    )

    return AnalyzeCommandResponse(
        command=result.command,
        user_intent=result.user_intent,
        required_operations=result.required_operations,
        final_explanation=result.final_explanation,
        execution_graph=result.execution_graph,
        evidence=result.evidence,
        dataset_summary=result.dataset_summary,
        validation_summary=result.validation_summary,
        duration_ms=result.duration_ms,
        relationships=result.relationships,
        top_relationships=result.top_relationships,
        subgroup_analysis=result.subgroup_analysis,
        correlation_matrix=result.correlation_matrix,
    )
