"""
Universal Data Quality Gate FastAPI Router.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from agent.agent_result import AgentResult, AgentStatus
from agent.data_quality_agent import DataQualityAgent

router = APIRouter(tags=["Data Quality Gate & Validation"])


class DataQualityValidationRequest(BaseModel):
    dataset: Optional[List[Dict[str, Any]]] = Field(None, description="Tabular dataset records")
    task_type: Optional[str] = Field("eda", description="Target analytical task: regression, classification, forecasting, clustering, anomaly_detection, statistical_relationship, eda, transformation")
    target: Optional[str] = Field(None, description="Optional target column name")
    features: Optional[List[str]] = Field(None, description="Optional subset of features to evaluate")
    time_column: Optional[str] = Field(None, description="Optional temporal/datetime column name")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional validation configuration thresholds")


@router.post("/data-quality/validate")
@router.post("/data-quality/check")
def validate_dataset_quality(req: DataQualityValidationRequest) -> Dict[str, Any]:
    """
    Evaluate dataset readiness and data quality gate status before analytical task execution.
    Returns canonical AgentResult payload.
    """
    if not req.dataset or len(req.dataset) == 0:
        raise HTTPException(
            status_code=400,
            detail="Dataset records are required for data quality validation. Provided dataset is empty.",
        )

    df = pd.DataFrame(req.dataset)
    agent = DataQualityAgent()
    task = {
        "data": df,
        "task_type": req.task_type or "eda",
        "target": req.target,
        "features": req.features,
        "time_column": req.time_column,
        "config": req.config or {},
    }
    result: AgentResult = agent.run(task)

    if not result.is_success:
        err_msg = result.error_message or "Data quality validation failed."
        raise HTTPException(
            status_code=400 if result.status == AgentStatus.VALIDATION_FAILED else 422,
            detail=err_msg,
        )

    return result.to_dict()