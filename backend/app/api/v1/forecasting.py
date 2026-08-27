from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.app.api.v1.context import link_uploaded_dataset, resolve_context
from backend.app.config import UPLOAD_DIR
from backend.app.forecasting import Forecaster
from backend.app.services.dataset_service import DatasetService
from agent.autonomous_forecaster_agent import AutonomousForecasterAgent
from agent.forecasting_schemas import ForecastRequest, WhatIfRequest
from agent.json_utils import sanitize_for_json

router = APIRouter(prefix="/forecast", tags=["forecasting"])
_forecaster_agent = AutonomousForecasterAgent()


class ForecastRunRequest(BaseModel):
    dataset: Optional[List[Dict[str, Any]]] = None
    target_column: Optional[str] = None
    time_column: Optional[str] = None
    forecast_horizon: int = 6
    confidence_level: float = 0.80


class WhatIfRunRequest(BaseModel):
    dataset: Optional[List[Dict[str, Any]]] = None
    target: str
    scenario_name: str = "Custom Scenario"
    changed_variables: Dict[str, Any] = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)
    scenarios: Optional[Dict[str, Dict[str, Any]]] = None


@router.post("")
def forecast_dataset(
    file: UploadFile = File(...),
    horizon: int = Form(3),
    target: str | None = Form(None),
    date_column: str | None = Form(None),
    workspace_id: str | None = Form(None),
    project_id: str | None = Form(None),
) -> dict[str, Any]:
    service = DatasetService(upload_folder=str(UPLOAD_DIR))
    try:
        resolved_workspace_id, resolved_project_id = resolve_context(workspace_id, project_id)
        uploaded = service.upload_dataset(file)
        dataframe = service._read_dataframe(str(UPLOAD_DIR / uploaded["dataset"]["name"]))
        result = Forecaster().forecast(dataframe, horizon=horizon, target=target, date_column=date_column)
        linked_dataset = link_uploaded_dataset(uploaded, resolved_workspace_id, resolved_project_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecasting failed: {exc}") from exc

    payload = {
        "status": "success",
        **result.to_dict(),
        "workspace_id": resolved_workspace_id,
        "project_id": resolved_project_id,
        "workspace_dataset_id": linked_dataset["id"] if linked_dataset else None,
    }
    return sanitize_for_json(payload)


@router.post("/run")
def run_autonomous_forecast(req: ForecastRunRequest) -> dict[str, Any]:
    """Execute autonomous time-series forecasting with probabilistic uncertainty intervals."""
    if not req.dataset or len(req.dataset) == 0:
        raise HTTPException(status_code=400, detail="Dataset records are required for forecasting.")

    try:
        df = pd.DataFrame(req.dataset)
        fc_req = ForecastRequest(
            dataset=df,
            target_column=req.target_column,
            time_column=req.time_column,
            forecast_horizon=req.forecast_horizon,
            confidence_level=req.confidence_level,
        )
        res = _forecaster_agent.forecast(fc_req)
        return sanitize_for_json(res.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Autonomous forecasting failed: {str(exc)}") from exc


@router.post("/whatif")
def run_whatif_scenario(req: WhatIfRunRequest) -> dict[str, Any]:
    """Execute counterfactual What-If scenario simulations against dataset baseline."""
    if not req.dataset or len(req.dataset) == 0:
        raise HTTPException(status_code=400, detail="Dataset records are required for What-If scenario modeling.")

    try:
        df = pd.DataFrame(req.dataset)

        # Multi-scenario comparison
        if req.scenarios:
            comp_res = _forecaster_agent.compare_scenarios(
                df=df,
                target=req.target,
                scenarios_spec=req.scenarios,
            )
            return sanitize_for_json(comp_res.to_dict())

        # Single scenario simulation
        whatif_req = WhatIfRequest(
            dataset=df,
            target=req.target,
            scenario_name=req.scenario_name,
            changed_variables=req.changed_variables,
            assumptions=req.assumptions,
        )
        scen_res = _forecaster_agent.scenario(whatif_req)
        return sanitize_for_json(scen_res.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"What-If scenario simulation failed: {str(exc)}") from exc

