"""
FastAPI REST Router for Production Model Serving & Live Inferences.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from agent.model_serving import (
    GLOBAL_MODEL_SERVING,
    ModelDeployment,
    ServingPredictionResult,
)

router = APIRouter(prefix="/models/served", tags=["Enterprise Model Serving"])


class DeployModelRequest(BaseModel):
    model_id: str
    endpoint_name: Optional[str] = None


class EndpointPredictRequest(BaseModel):
    records: List[Dict[str, Any]]


@router.post("/deploy", response_model=ModelDeployment)
def deploy_model_endpoint(req: DeployModelRequest):
    """Deploy a trained model to a dedicated live REST endpoint."""
    try:
        return GLOBAL_MODEL_SERVING.deploy_model(req.model_id, endpoint_name=req.endpoint_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/endpoints", response_model=List[ModelDeployment])
def list_deployed_endpoints():
    """List all currently active deployed model REST endpoints."""
    return GLOBAL_MODEL_SERVING.list_deployments()


@router.post("/{deployment_id}/predict", response_model=ServingPredictionResult)
def predict_via_endpoint(deployment_id: str, req: EndpointPredictRequest):
    """Execute real-time prediction through the deployed endpoint."""
    if not req.records:
        raise HTTPException(status_code=400, detail="Input records cannot be empty")
    try:
        return GLOBAL_MODEL_SERVING.predict_endpoint(deployment_id, req.records)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{deployment_id}")
def undeploy_model_endpoint(deployment_id: str):
    """Undeploy and terminate a model REST endpoint."""
    deleted = GLOBAL_MODEL_SERVING.undeploy(deployment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Deployment ID not found")
    return {"success": True, "undeployed_id": deployment_id}
