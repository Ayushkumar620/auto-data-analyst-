from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from backend.app.api.v1.context import link_uploaded_dataset, resolve_context
from backend.app.config import UPLOAD_DIR
from backend.app.eda.orchestrator import EDAOrchestrator
from backend.app.insights import InsightEngine
from backend.app.reports import ReportEngine
from backend.app.services.dataset_service import DatasetService

router = APIRouter(prefix="/reports", tags=["reports"])


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

_REPORTS: dict[str, dict[str, Any]] = {}


@router.post("/generate")
def generate_report(
    file: UploadFile = File(...),
    output_format: str = Form("pdf"),
    workspace_id: str | None = Form(None),
    project_id: str | None = Form(None),
) -> dict[str, Any]:
    service = DatasetService(upload_folder=str(UPLOAD_DIR))
    try:
        resolved_workspace_id, resolved_project_id = resolve_context(workspace_id, project_id)
        uploaded = service.upload_dataset(file)
        dataframe = service._read_dataframe(str(UPLOAD_DIR / uploaded["dataset"]["name"]))
        eda = EDAOrchestrator().analyze(dataframe)
        insights = InsightEngine().generate(dataframe, eda)
        linked_dataset = link_uploaded_dataset(uploaded, resolved_workspace_id, resolved_project_id)
        analysis = {
            "dataframe": dataframe,
            "eda": eda,
            "insights": insights["insights"],
            "dataset_name": uploaded["dataset"]["name"],
            "forecast": {},
        }
        report, content, content_type = ReportEngine().generate(uploaded["dataset"]["id"], analysis, output_format)
        _REPORTS[report.report_id] = {
            "content": content,
            "content_type": content_type,
            "format": output_format,
            "title": report.title,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}") from exc

    payload = {
        "status": "completed",
        "report_id": report.report_id,
        "download_url": f"/api/v1/reports/{report.report_id}",
        "report": report.to_dict(),
        "workspace_id": resolved_workspace_id,
        "project_id": resolved_project_id,
        "workspace_dataset_id": linked_dataset["id"] if linked_dataset else None,
    }
    return json.loads(json.dumps(payload, default=_json_default))


@router.get("/{report_id}")
def download_report(report_id: str) -> Response:
    stored = _REPORTS.get(report_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Report not found.")

    extension = {"pdf": "pdf", "excel": "xlsx", "powerpoint": "pptx"}[stored["format"]]
    headers = {"Content-Disposition": f'attachment; filename="{report_id}.{extension}"'}
    return Response(content=stored["content"], media_type=stored["content_type"], headers=headers)
