from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder

from backend.app.config import UPLOAD_DIR
from backend.app.eda.orchestrator import EDAOrchestrator
from backend.app.services.dataset_service import DatasetService
from backend.app.visualization.charts import ChartFactory
from backend.app.visualization.serializers import figure_to_json

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/")
def list_datasets() -> dict[str, Any]:
    return {"datasets": []}


@router.post("/upload")
def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    service = DatasetService(upload_folder=str(UPLOAD_DIR))
    try:
        result = service.upload_dataset(file)
        dataframe = service._read_dataframe(str(UPLOAD_DIR / result["dataset"]["name"]))
        profile = service.profile_dataset(
            dataframe=dataframe,
            filename=result["dataset"]["name"],
            file_type=result["dataset"]["file_type"],
            file_size=result["metadata"]["file_size"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    metadata = result["metadata"]
    payload = {
        "status": "uploaded",
        "dataset": result["dataset"],
        "rows": result["dataset"]["rows"],
        "columns": result["dataset"]["columns"],
        "file_type": result["dataset"]["file_type"],
        "column_names": metadata["column_names"],
        "data_types": metadata["data_types"],
        "missing_values": metadata["missing_values"],
        "duplicate_rows": metadata["duplicate_rows"],
        "memory_usage": metadata["memory_usage"],
        "profile": profile["profile"],
        "column_analysis": profile["column_analysis"],
        "numeric_analysis": profile["numeric_analysis"],
        "categorical_analysis": profile["categorical_analysis"],
        "preview": profile["preview"],
        "recommendations": profile["recommendations"],
    }
    return payload


@router.post("/clean")
def clean_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    service = DatasetService(upload_folder=str(UPLOAD_DIR))
    try:
        result = service.clean_dataset(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cleaning failed: {exc}") from exc

    return result


@router.post("/eda")
def analyze_dataset_eda(file: UploadFile = File(...)) -> dict[str, Any]:
    service = DatasetService(upload_folder=str(UPLOAD_DIR))
    try:
        uploaded = service.upload_dataset(file)
        dataframe = service._read_dataframe(str(UPLOAD_DIR / uploaded["dataset"]["name"]))
        result = EDAOrchestrator().analyze(dataframe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"EDA failed: {exc}") from exc

    return jsonable_encoder(result)


@router.post("/chart")
def generate_chart(
    file: UploadFile = File(...),
    chart_type: str = Form(...),
    x: str | None = Form(None),
    y: str | None = Form(None),
    title: str | None = Form(None),
) -> dict[str, Any]:
    service = DatasetService(upload_folder=str(UPLOAD_DIR))
    try:
        uploaded = service.upload_dataset(file)
        dataframe = service._read_dataframe(str(UPLOAD_DIR / uploaded["dataset"]["name"]))
        figure = ChartFactory().create(dataframe, chart_type=chart_type, x=x, y=y, title=title)
        payload = figure_to_json(figure)
        payload["chart_type"] = chart_type
        if title is not None:
            payload["layout"]["title"] = title
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {exc}") from exc

    return payload
