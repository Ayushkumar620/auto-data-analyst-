from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.config import UPLOAD_DIR
from backend.app.eda.orchestrator import EDAOrchestrator
from backend.app.insights import InsightEngine
from backend.app.services.dataset_service import DatasetService

router = APIRouter(prefix="/insights", tags=["insights"])

_ANALYSES: dict[str, dict[str, Any]] = {}


@router.post("/generate")
def generate_insights(file: UploadFile = File(...)) -> dict[str, Any]:
    service = DatasetService(upload_folder=str(UPLOAD_DIR))
    try:
        uploaded = service.upload_dataset(file)
        dataset_path = UPLOAD_DIR / uploaded["dataset"]["name"]
        dataframe = service._read_dataframe(str(dataset_path))
        eda = EDAOrchestrator().analyze(dataframe)
        result = InsightEngine().generate(dataframe, eda)
        dataset_id = uploaded["dataset"]["id"]
        _ANALYSES[dataset_id] = {
            "dataframe": dataframe,
            "eda": eda,
            "dataset_name": uploaded["dataset"]["name"],
            "insights": result["insights"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {exc}") from exc

    return result
