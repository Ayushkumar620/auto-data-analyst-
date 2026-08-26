"""
Universal Anomaly Detection FastAPI Router.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from agent.agent_result import AgentResult, AgentStatus
from agent.anomaly_agent import AnomalyDetectionAgent
from agent.anomaly_detection_engine import AnomalyDetectionEngine

router = APIRouter(prefix="/anomalies", tags=["Anomaly Detection"])


class AnomalyDetectRequest(BaseModel):
    dataset: Optional[List[Dict[str, Any]]] = Field(None, description="Tabular dataset records")
    features: Optional[List[str]] = Field(None, description="Optional subset of features to inspect")
    contamination: Optional[Union[str, float]] = Field("auto", description="'auto' or float in (0, 0.5]")
    method: Optional[str] = Field(None, description="Optional algorithm override")


@router.post("/detect")
@router.post("")
def detect_anomalies(req: AnomalyDetectRequest) -> Dict[str, Any]:
    """Execute autonomous dataset-agnostic anomaly and outlier detection."""
    if not req.dataset or len(req.dataset) == 0:
        raise HTTPException(
            status_code=400,
            detail="Dataset records are required for anomaly detection. Provided dataset is empty.",
        )

    df = pd.DataFrame(req.dataset)

    agent = AnomalyDetectionAgent()
    task = {
        "data": df,
        "features": req.features,
        "contamination": req.contamination or "auto",
        "method": req.method,
    }
    result: AgentResult = agent.run(task)

    if not result.is_success:
        err_msg = result.error_message or "Anomaly detection failed."
        raise HTTPException(
            status_code=400 if result.status == AgentStatus.VALIDATION_FAILED else 422,
            detail=err_msg,
        )

    return result.to_dict()
