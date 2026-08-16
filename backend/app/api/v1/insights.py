from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.api.v1.context import link_uploaded_dataset, resolve_context
from backend.app.config import UPLOAD_DIR
from backend.app.eda.orchestrator import EDAOrchestrator
from backend.app.insights import InsightEngine
from backend.app.services.dataset_service import DatasetService

router = APIRouter(prefix="/insights", tags=["insights"])

_ANALYSES: dict[str, dict[str, Any]] = {}


@router.post("/generate")
def generate_insights(
    file: UploadFile = File(...),
    workspace_id: str | None = Form(None),
    project_id: str | None = Form(None),
) -> dict[str, Any]:
    service = DatasetService(upload_folder=str(UPLOAD_DIR))
    try:
        resolved_workspace_id, resolved_project_id = resolve_context(workspace_id, project_id)
        uploaded = service.upload_dataset(file)
        dataset_path = UPLOAD_DIR / uploaded["dataset"]["name"]
        dataframe = service._read_dataframe(str(dataset_path))
        eda = EDAOrchestrator().analyze(dataframe)
        result = InsightEngine().generate(dataframe, eda)
        linked_dataset = link_uploaded_dataset(uploaded, resolved_workspace_id, resolved_project_id)
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

    return {
        **result,
        "workspace_id": resolved_workspace_id,
        "project_id": resolved_project_id,
        "workspace_dataset_id": linked_dataset["id"] if linked_dataset else None,
    }
