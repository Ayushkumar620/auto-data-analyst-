"""
Universal Clustering & Segmentation FastAPI Router.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from agent.agent_result import AgentResult, AgentStatus
from agent.clustering_agent import ClusteringAgent
from agent.clustering_engine import ClusteringEngine

router = APIRouter(prefix="/clustering", tags=["Clustering & Segmentation"])


class ClusteringRequest(BaseModel):
    dataset: Optional[List[Dict[str, Any]]] = Field(None, description="Tabular dataset records")
    features: Optional[List[str]] = Field(None, description="Optional subset of feature column names")
    n_clusters: Optional[Union[str, int]] = Field("auto", description="'auto' or int >= 2")
    method: Optional[str] = Field(None, description="Optional clustering algorithm override")
    random_state: Optional[int] = Field(42, description="Random seed for reproducibility")


@router.post("/run")
@router.post("")
def run_clustering(req: ClusteringRequest) -> Dict[str, Any]:
    """Execute autonomous dataset-agnostic clustering and segment profiling."""
    if not req.dataset or len(req.dataset) == 0:
        raise HTTPException(
            status_code=400,
            detail="Dataset records are required for clustering. Provided dataset is empty.",
        )

    df = pd.DataFrame(req.dataset)

    agent = ClusteringAgent()
    task = {
        "data": df,
        "features": req.features,
        "n_clusters": req.n_clusters or "auto",
        "method": req.method,
        "random_state": req.random_state if req.random_state is not None else 42,
    }
    result: AgentResult = agent.run(task)

    if not result.is_success:
        err_msg = result.error_message or "Clustering failed."
        raise HTTPException(
            status_code=400 if result.status == AgentStatus.VALIDATION_FAILED else 422,
            detail=err_msg,
        )

    return result.to_dict()
