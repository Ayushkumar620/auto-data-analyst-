"""FastAPI router for Model Registry management and live inference execution."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.ml.registry import ModelRegistry

router = APIRouter(prefix="/models", tags=["Model Registry"])
_registry = ModelRegistry()


class PredictRequest(BaseModel):
    data: Any = Field(..., description="DataFrame dict, single record dict, or list of record dicts.")


class StatusUpdateRequest(BaseModel):
    status: str = Field(..., description="New lifecycle status: active, staging, or archived.")


@router.get("", response_model=List[Dict[str, Any]])
def list_models(
    family: Optional[str] = Query(None, description="Filter by model family: traditional_ml, ann, cnn, forecasting"),
    problem_type: Optional[str] = Query(None, description="Filter by problem type: classification, regression"),
    status: Optional[str] = Query(None, description="Filter by status: active, staging, archived"),
):
    """Retrieve leaderboard of all registered ML, ANN, and CNN models."""
    return _registry.list_models(family=family, problem_type=problem_type, status=status)


@router.get("/{model_id}", response_model=Dict[str, Any])
def get_model_metadata(model_id: str):
    """Retrieve full metadata, loss curves, metrics, and schema for a specific registered model."""
    meta = _registry.get_metadata(model_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Model ID '{model_id}' not found.")
    return meta.to_dict()


@router.post("/{model_id}/predict", response_model=Dict[str, Any])
def run_model_inference(model_id: str, request: PredictRequest):
    """Execute live batch or single-record inference using a registered model artifact."""
    try:
        return _registry.predict(model_id, request.data)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(exc)}")


@router.patch("/{model_id}/status", response_model=Dict[str, Any])
def update_model_status(model_id: str, request: StatusUpdateRequest):
    """Update the deployment lifecycle status of a model (e.g. promote to active or archive)."""
    success = _registry.set_status(model_id, request.status)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model ID '{model_id}' not found.")
    return {"status": "success", "model_id": model_id, "new_status": request.status}


@router.delete("/{model_id}", response_model=Dict[str, Any])
def delete_model(model_id: str):
    """Delete a model artifact and remove it from the registry."""
    success = _registry.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model ID '{model_id}' not found.")
    return {"status": "success", "message": f"Model '{model_id}' deleted successfully."}

