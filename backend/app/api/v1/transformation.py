"""
Universal Data Transformation & Feature Engineering FastAPI Router.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from agent.agent_result import AgentResult, AgentStatus
from agent.transformation_agent import TransformationAgent

router = APIRouter(tags=["Data Transformation & Preprocessing"])


class TransformationFitRequest(BaseModel):
    dataset: Optional[List[Dict[str, Any]]] = Field(None, description="Tabular training dataset records")
    target: Optional[str] = Field(None, description="Optional target column name")
    features: Optional[List[str]] = Field(None, description="Optional subset of features to transform")
    task_type: Optional[str] = Field(None, description="Task type (regression, classification, clustering, etc.)")
    config: Optional[Dict[str, Any]] = Field(None, description="Transformation parameters (imputation, scaling, etc.)")


class TransformationTransformRequest(BaseModel):
    dataset: Optional[List[Dict[str, Any]]] = Field(None, description="Tabular evaluation dataset records to transform")
    state: Optional[Dict[str, Any]] = Field(None, description="Fitted TransformationState dictionary")
    drift_policy: Optional[str] = Field("compatible", description="Schema drift handling policy: compatible, strict, permissive")


class TransformationRunRequest(BaseModel):
    dataset: Optional[List[Dict[str, Any]]] = Field(None, description="Tabular dataset records")
    target: Optional[str] = Field(None, description="Optional target column name")
    features: Optional[List[str]] = Field(None, description="Optional subset of features to transform")
    task_type: Optional[str] = Field(None, description="Task type (regression, classification, clustering, etc.)")
    config: Optional[Dict[str, Any]] = Field(None, description="Transformation parameters")


@router.post("/transformation")
@router.post("/transformation/run")
def run_fit_transform(req: TransformationRunRequest) -> Dict[str, Any]:
    """Execute complete fit_transform pipeline returning transformed schema, plan, and state."""
    if not req.dataset or len(req.dataset) == 0:
        raise HTTPException(
            status_code=400,
            detail="Dataset records are required for transformation. Provided dataset is empty.",
        )

    df = pd.DataFrame(req.dataset)
    agent = TransformationAgent()
    task = {
        "data": df,
        "target": req.target,
        "features": req.features,
        "task_type": req.task_type,
        "config": req.config or {},
        "action": "fit_transform",
    }
    result: AgentResult = agent.run(task)

    if not result.is_success:
        err_msg = result.error_message or "Transformation failed."
        raise HTTPException(
            status_code=400 if result.status == AgentStatus.VALIDATION_FAILED else 422,
            detail=err_msg,
        )

    return result.to_dict()


@router.post("/transformation/fit")
def run_fit(req: TransformationFitRequest) -> Dict[str, Any]:
    """Fit transformation parameters and produce serializable TransformationState."""
    if not req.dataset or len(req.dataset) == 0:
        raise HTTPException(
            status_code=400,
            detail="Dataset records are required for transformation fitting. Provided dataset is empty.",
        )

    df = pd.DataFrame(req.dataset)
    agent = TransformationAgent()
    task = {
        "data": df,
        "target": req.target,
        "features": req.features,
        "task_type": req.task_type,
        "config": req.config or {},
        "action": "fit",
    }
    result: AgentResult = agent.run(task)

    if not result.is_success:
        err_msg = result.error_message or "Fitting transformation failed."
        raise HTTPException(
            status_code=400 if result.status == AgentStatus.VALIDATION_FAILED else 422,
            detail=err_msg,
        )

    return result.to_dict()


@router.post("/transformation/transform")
def run_transform(req: TransformationTransformRequest) -> Dict[str, Any]:
    """Transform inference/test dataset using pre-fitted TransformationState."""
    if not req.dataset or len(req.dataset) == 0:
        raise HTTPException(
            status_code=400,
            detail="Dataset records are required for transformation. Provided dataset is empty.",
        )
    if not req.state:
        raise HTTPException(
            status_code=400,
            detail="Fitted transformation state is required for transformation.",
        )

    df = pd.DataFrame(req.dataset)
    agent = TransformationAgent()
    task = {
        "data": df,
        "state": req.state,
        "action": "transform",
    }
    result: AgentResult = agent.run(task)

    if not result.is_success:
        err_msg = result.error_message or "Transforming dataset failed."
        raise HTTPException(
            status_code=400 if result.status == AgentStatus.VALIDATION_FAILED else 422,
            detail=err_msg,
        )

    return result.to_dict()