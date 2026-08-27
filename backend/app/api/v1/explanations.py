"""
Universal Analytical Explanation & Evidence Traceability FastAPI Router.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from agent.agent_result import AgentResult, AgentStatus
from agent.explanation_agent import ExplanationAgent
from agent.explanation_engine import ExplanationEngine

router = APIRouter(prefix="/explanations", tags=["Universal Analytical Explanation"])


class ExplanationRequest(BaseModel):
    result: Optional[Dict[str, Any]] = Field(None, description="Analytical result payload to explain")
    dataset: Optional[List[Dict[str, Any]]] = Field(None, description="Optional raw dataset records")
    command: Optional[str] = Field(None, description="Optional original user command")
    depth: Optional[str] = Field("detailed", description="Explanation detail level: 'detailed', 'summary', 'audit'")
    synthesis_report: Optional[Dict[str, Any]] = Field(None, description="Optional synthesis report")


def _execute_explanation(req: ExplanationRequest) -> Dict[str, Any]:
    df = pd.DataFrame(req.dataset) if (req.dataset is not None and len(req.dataset) > 0) else None
    agent = ExplanationAgent()

    payload = req.result or req.synthesis_report or {}
    if not payload and df is None:
        raise HTTPException(
            status_code=400,
            detail="Either analytical result payload or dataset records must be provided for explanation.",
        )

    res: AgentResult = agent.execute({
        "result": payload,
        "data": df,
        "command": req.command,
        "depth": req.depth or "detailed",
    })

    if res.status == AgentStatus.ERROR:
        err_msg = res.error_message or "Explanation generation failed."
        raise HTTPException(status_code=422, detail=err_msg)

    return res.to_dict()


@router.post("/explain")
def explain_analysis(req: ExplanationRequest) -> Dict[str, Any]:
    """
    Generate an auditable, evidence-backed analytical explanation for any analytical result or model.
    """
    return _execute_explanation(req)


@router.post("")
def create_explanation(req: ExplanationRequest) -> Dict[str, Any]:
    """
    Canonical endpoint to generate an analytical explanation with complete evidence traceability.
    """
    return _execute_explanation(req)