from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.model_monitor_agent import ModelMonitorAgent
from agent.model_monitoring_schemas import DriftSeverity, DriftThresholdConfig
from backend.app.ml.registry import ModelRegistry

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

_registry = ModelRegistry()
_monitor_agent = ModelMonitorAgent(registry=_registry)
_monitoring_history: List[Dict[str, Any]] = []


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class MonitoringRunRequest(BaseModel):
    model_id: str = Field(..., description="Registered model ID to monitor")
    current_dataset: List[Dict[str, Any]] = Field(..., description="New production / evaluation dataset records")
    reference_dataset: Optional[List[Dict[str, Any]]] = Field(None, description="Optional reference dataset records")
    feature_columns: Optional[List[str]] = Field(None, description="Optional subset of features to evaluate")
    target_column: Optional[str] = Field(None, description="Optional target column for performance evaluation")
    thresholds: Optional[Dict[str, float]] = Field(None, description="Custom statistical thresholds")


@router.post("/run")
def run_model_monitoring(req: MonitoringRunRequest) -> Dict[str, Any]:
    """Execute statistical data drift, schema consistency, and performance degradation evaluation."""
    if not req.model_id:
        raise HTTPException(status_code=400, detail="Missing required 'model_id' parameter.")
    if not req.current_dataset or len(req.current_dataset) == 0:
        raise HTTPException(status_code=400, detail="Current dataset records are required for drift evaluation.")

    try:
        curr_df = pd.DataFrame(req.current_dataset)
        ref_df = pd.DataFrame(req.reference_dataset) if req.reference_dataset else None

        agent_result = _monitor_agent.run({
            "model_id": req.model_id,
            "current_data": curr_df,
            "reference_data": ref_df,
            "feature_columns": req.feature_columns,
            "target_column": req.target_column,
            "thresholds": req.thresholds or {},
        })

        if agent_result.status == "error":
            raise HTTPException(status_code=400, detail=agent_result.message or "Monitoring execution failed.")

        output_data = agent_result.result or agent_result.data or {}
        output_data["run_id"] = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(_monitoring_history) + 1}"
        output_data["model_id"] = req.model_id
        output_data["executed_at"] = datetime.now(timezone.utc).isoformat()

        # Save to local history
        _monitoring_history.insert(0, output_data)

        return json.loads(json.dumps(output_data, default=_json_default))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model monitoring failed: {str(exc)}") from exc


@router.get("/history")
def get_monitoring_history(model_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve historical monitoring runs, optionally filtered by model_id."""
    if model_id:
        filtered = [run for run in _monitoring_history if run.get("model_id") == model_id]
        return json.loads(json.dumps(filtered, default=_json_default))
    return json.loads(json.dumps(_monitoring_history, default=_json_default))


@router.get("/overview")
def get_monitoring_overview() -> Dict[str, Any]:
    """Retrieve high-level summary of model health, drift alerts, and total monitored models."""
    models = _registry.list_models()
    total_models = len(models)
    
    # Calculate health statuses from registered models & latest monitoring runs
    healthy_count = 0
    warning_count = 0
    critical_count = 0

    latest_runs_by_model: Dict[str, Dict[str, Any]] = {}
    for run in _monitoring_history:
        mid = run.get("model_id")
        if mid and mid not in latest_runs_by_model:
            latest_runs_by_model[mid] = run

    for m in models:
        mid = m.get("model_id")
        latest = latest_runs_by_model.get(mid)
        if latest:
            sev = str(latest.get("overall_severity", "NONE")).upper()
            if sev in ("HIGH", "CRITICAL"):
                critical_count += 1
            elif sev in ("MEDIUM", "LOW", "WARNING"):
                warning_count += 1
            else:
                healthy_count += 1
        else:
            # Default healthy if model status is active
            if m.get("status") == "active":
                healthy_count += 1
            else:
                healthy_count += 1

    last_run_timestamp = _monitoring_history[0].get("executed_at") if _monitoring_history else None

    return {
        "total_models": total_models,
        "monitored_models": len(latest_runs_by_model) if latest_runs_by_model else total_models,
        "healthy_models": healthy_count,
        "warning_models": warning_count,
        "critical_models": critical_count,
        "total_runs": len(_monitoring_history),
        "last_run_timestamp": last_run_timestamp,
    }
