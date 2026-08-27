"""
Universal Statistical Relationship & Dependency Analysis FastAPI Router.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from agent.agent_result import AgentResult, AgentStatus
from agent.statistical_analysis_agent import StatisticalAnalysisAgent

router = APIRouter(prefix="/statistical-analysis", tags=["Statistical Relationship Analysis"])


class StatisticalAnalysisRequest(BaseModel):
    dataset: Optional[List[Dict[str, Any]]] = Field(None, description="Tabular dataset records")
    features: Optional[List[str]] = Field(None, description="Optional list of feature columns to evaluate")
    target: Optional[str] = Field(None, description="Optional target column to focus relationships against")
    alpha: Optional[float] = Field(0.05, description="Significance threshold for hypothesis tests")
    max_pairs: Optional[int] = Field(250, description="Maximum number of variable pairs to evaluate")


@router.post("/run")
@router.post("")
def run_statistical_analysis(req: StatisticalAnalysisRequest) -> Dict[str, Any]:
    """Execute autonomous dataset-agnostic relationship discovery, testing, and ranking."""
    if not req.dataset or len(req.dataset) == 0:
        raise HTTPException(
            status_code=400,
            detail="Dataset records are required for statistical analysis. Provided dataset is empty.",
        )

    df = pd.DataFrame(req.dataset)

    agent = StatisticalAnalysisAgent()
    task = {
        "data": df,
        "features": req.features,
        "target": req.target,
        "alpha": req.alpha if req.alpha is not None else 0.05,
        "max_pairs": req.max_pairs if req.max_pairs is not None else 250,
    }
    result: AgentResult = agent.run(task)

    if not result.is_success:
        err_msg = result.error_message or "Statistical relationship analysis failed."
        raise HTTPException(
            status_code=400 if result.status == AgentStatus.VALIDATION_FAILED else 422,
            detail=err_msg,
        )

    return result.to_dict()