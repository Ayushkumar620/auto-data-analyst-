from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from backend.app.config import UPLOAD_DIR
from backend.app.eda.orchestrator import EDAOrchestrator
from backend.app.insights import InsightEngine
from backend.app.reports import ReportEngine
from backend.app.services.dataset_service import DatasetService

router = APIRouter(prefix="/reports", tags=["reports"])

_REPORTS: dict[str, dict[str, Any]] = {}


@router.post("/generate")
def generate_report(
    file: UploadFile = File(...),
    output_format: str = Form("pdf"),
) -> dict[str, Any]:
    service = DatasetService(upload_folder=str(UPLOAD_DIR))
    try:
        uploaded = service.upload_dataset(file)
        dataframe = service._read_dataframe(str(UPLOAD_DIR / uploaded["dataset"]["name"]))
        eda = EDAOrchestrator().analyze(dataframe)
        insights = InsightEngine().generate(dataframe, eda)
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

    return {
        "status": "completed",
        "report_id": report.report_id,
        "download_url": f"/api/v1/reports/{report.report_id}",
        "report": report.to_dict(),
    }


@router.get("/{report_id}")
def download_report(report_id: str) -> Response:
    stored = _REPORTS.get(report_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Report not found.")

    extension = {"pdf": "pdf", "excel": "xlsx", "powerpoint": "pptx"}[stored["format"]]
    headers = {"Content-Disposition": f'attachment; filename="{report_id}.{extension}"'}
    return Response(content=stored["content"], media_type=stored["content_type"], headers=headers)
