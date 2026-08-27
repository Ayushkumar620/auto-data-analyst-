"""
Universal EDA, Data Profiling & Data Quality FastAPI Router.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from agent.agent_result import AgentResult, AgentStatus
from agent.eda_agent import EDAAgent

router = APIRouter(tags=["EDA & Data Profiling"])


class EDARequest(BaseModel):
    dataset: Optional[List[Dict[str, Any]]] = Field(None, description="Tabular dataset records")
    columns: Optional[List[str]] = Field(None, description="Optional column subset to profile")
    target: Optional[str] = Field(None, description="Optional target column to profile")
    max_categories: Optional[int] = Field(10, description="Maximum categorical frequency levels to report")


@router.post("/eda/profile")
@router.post("/eda")
@router.post("/eda/run")
@router.post("/data/profile")
def run_eda_profile(req: EDARequest) -> Dict[str, Any]:
    """Execute autonomous dataset profiling, schema inference, distributions, and quality assessment."""
    if not req.dataset or len(req.dataset) == 0:
        raise HTTPException(
            status_code=400,
            detail="Dataset records are required for EDA profiling. Provided dataset is empty.",
        )

    df = pd.DataFrame(req.dataset)

    agent = EDAAgent()
    task = {
        "data": df,
        "columns": req.columns,
        "target": req.target,
        "max_categories": req.max_categories if req.max_categories is not None else 10,
    }
    result: AgentResult = agent.run(task)

    if not result.is_success:
        err_msg = result.error_message or "EDA profiling failed."
        raise HTTPException(
            status_code=400 if result.status == AgentStatus.VALIDATION_FAILED else 422,
            detail=err_msg,
        )

    return result.to_dict()