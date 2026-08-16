from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.config import UPLOAD_DIR
from backend.app.eda.orchestrator import EDAOrchestrator
from backend.app.services.dataset_service import DatasetService
from backend.app.services.workspace_service import get_workspace_service
from backend.app.visualization.charts import ChartFactory
from backend.app.visualization.serializers import figure_to_json

router = APIRouter(prefix="/datasets", tags=["datasets"])


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


@router.get("/")
def list_datasets() -> dict[str, Any]:
    return {"datasets": []}


@router.post("/upload")
def upload_dataset(
    file: UploadFile = File(...),
    workspace_id: str | None = Form(None),
    project_id: str | None = Form(None),
) -> dict[str, Any]:
    service = DatasetService(upload_folder=str(UPLOAD_DIR))
    workspace_service = get_workspace_service()
    try:
        resolved_workspace_id = workspace_id
        if project_id:
            project = workspace_service.get_project(project_id)
            if project is None:
                raise ValueError("Project not found")
            if resolved_workspace_id and project["workspace_id"] != resolved_workspace_id:
                raise ValueError("Project does not belong to workspace")
            resolved_workspace_id = project["workspace_id"]

        if resolved_workspace_id and not workspace_service.has_workspace(resolved_workspace_id):
            raise ValueError("Workspace not found")

        result = service.upload_dataset(file)
        dataframe = service._read_dataframe(str(UPLOAD_DIR / result["dataset"]["name"]))
        profile = service.profile_dataset(
            dataframe=dataframe,
            filename=result["dataset"]["name"],
            file_type=result["dataset"]["file_type"],
            file_size=result["metadata"]["file_size"],
        )

        linked_dataset: dict[str, Any] | None = None
        if resolved_workspace_id and project_id:
            linked_dataset = workspace_service.create_dataset(
                project_id=project_id,
                workspace_id=resolved_workspace_id,
                name=result["dataset"]["name"],
                rows=result["dataset"]["rows"],
                columns=result["dataset"]["columns"],
                schema=result["metadata"]["column_names"],
                stats={
                    "missing_values": result["metadata"]["missing_values"],
                    "duplicate_rows": result["metadata"]["duplicate_rows"],
                    "memory_usage": result["metadata"]["memory_usage"],
                },
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
        "workspace_id": resolved_workspace_id,
        "project_id": project_id,
        "workspace_dataset_id": linked_dataset["id"] if linked_dataset else None,
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

    return json.loads(json.dumps(result, default=_json_default))


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
