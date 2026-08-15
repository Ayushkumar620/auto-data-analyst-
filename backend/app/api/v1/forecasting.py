from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.config import UPLOAD_DIR
from backend.app.forecasting import Forecaster
from backend.app.services.dataset_service import DatasetService

router = APIRouter(prefix="/forecast", tags=["forecasting"])


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


@router.post("")
def forecast_dataset(
    file: UploadFile = File(...),
    horizon: int = Form(3),
    target: str | None = Form(None),
    date_column: str | None = Form(None),
) -> dict[str, Any]:
    service = DatasetService(upload_folder=str(UPLOAD_DIR))
    try:
        uploaded = service.upload_dataset(file)
        dataframe = service._read_dataframe(str(UPLOAD_DIR / uploaded["dataset"]["name"]))
        result = Forecaster().forecast(dataframe, horizon=horizon, target=target, date_column=date_column)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecasting failed: {exc}") from exc

    payload = {"status": "success", **result.to_dict()}
    return json.loads(json.dumps(payload, default=_json_default))
