from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.config import UPLOAD_DIR
from backend.app.forecasting import Forecaster
from backend.app.services.dataset_service import DatasetService

router = APIRouter(prefix="/forecast", tags=["forecasting"])


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

    return {"status": "success", **result.to_dict()}
