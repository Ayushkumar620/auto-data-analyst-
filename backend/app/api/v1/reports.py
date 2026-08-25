from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
import uuid
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

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
_JSON_REPORTS: dict[str, dict[str, Any]] = {}


class CreateReportRequest(BaseModel):
    title: str = Field(..., description="Report title")
    dataset_name: Optional[str] = "Dataset"
    report_type: str = "comprehensive"  # "comprehensive", "forecast", "model", "monitoring", "analysis"
    executive_summary: str = Field(..., description="Executive findings summary")
    dataset_overview: Dict[str, Any] = Field(default_factory=dict)
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    kpis: List[Dict[str, Any]] = Field(default_factory=list)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    insights: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    forecast: Dict[str, Any] = Field(default_factory=dict)
    model_results: Dict[str, Any] = Field(default_factory=dict)
    monitoring: Dict[str, Any] = Field(default_factory=dict)


@router.get("")
def list_reports() -> List[Dict[str, Any]]:
    """List all generated analytical reports."""
    report_list = []
    
    # Add structured JSON reports
    for rid, rep in _JSON_REPORTS.items():
        report_list.append({
            "report_id": rid,
            "title": rep.get("title", "Untitled Report"),
            "dataset_name": rep.get("dataset_name", "Dataset"),
            "report_type": rep.get("report_type", "comprehensive"),
            "created_at": rep.get("created_at", datetime.now(timezone.utc).isoformat()),
            "status": rep.get("status", "ready"),
            "executive_summary": rep.get("executive_summary", "")[:180] + ("..." if len(rep.get("executive_summary", "")) > 180 else ""),
            "kpi_count": len(rep.get("kpis", [])),
            "insight_count": len(rep.get("insights", [])),
            "recommendation_count": len(rep.get("recommendations", [])),
            "has_forecast": bool(rep.get("forecast")),
            "has_model": bool(rep.get("model_results")),
            "has_monitoring": bool(rep.get("monitoring")),
        })

    # Add any file-generated reports
    for rid, rep in _REPORTS.items():
        if rid not in _JSON_REPORTS:
            report_list.append({
                "report_id": rid,
                "title": rep.get("title", "Generated Report"),
                "dataset_name": "Uploaded Dataset",
                "report_type": "file_export",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "ready",
                "executive_summary": "Auto-generated analytical export package.",
                "kpi_count": 0,
                "insight_count": 0,
                "recommendation_count": 0,
                "has_forecast": False,
                "has_model": False,
                "has_monitoring": False,
            })

    return json.loads(json.dumps(report_list, default=_json_default))


@router.post("/create")
def create_report(req: CreateReportRequest) -> Dict[str, Any]:
    """Create a structured analytical report deliverable."""
    report_id = f"rep_{uuid.uuid4().hex[:8]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    report_payload = {
        "report_id": report_id,
        "title": req.title,
        "dataset_name": req.dataset_name,
        "report_type": req.report_type,
        "created_at": now_iso,
        "status": "ready",
        "executive_summary": req.executive_summary,
        "dataset_overview": req.dataset_overview,
        "data_quality": req.data_quality,
        "kpis": req.kpis,
        "charts": req.charts,
        "insights": req.insights,
        "evidence": req.evidence,
        "recommendations": req.recommendations,
        "forecast": req.forecast,
        "model_results": req.model_results,
        "monitoring": req.monitoring,
    }

    _JSON_REPORTS[report_id] = report_payload
    return json.loads(json.dumps(report_payload, default=_json_default))


@router.get("/detail/{report_id}")
def get_report_detail(report_id: str) -> Dict[str, Any]:
    """Get full structured content for a specific report."""
    rep = _JSON_REPORTS.get(report_id)
    if rep:
        return json.loads(json.dumps(rep, default=_json_default))

    # Check if exists in file-based engine
    stored = _REPORTS.get(report_id)
    if stored:
        return {
            "report_id": report_id,
            "title": stored.get("title", "Exported Report"),
            "dataset_name": "Uploaded Dataset",
            "report_type": "file_export",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ready",
            "executive_summary": "Auto-generated analytical deliverable package.",
            "kpis": [],
            "insights": [],
            "charts": [],
            "recommendations": [],
            "evidence": [],
            "download_url": f"/api/v1/reports/{report_id}",
        }

    raise HTTPException(status_code=404, detail="Report not found.")


@router.delete("/{report_id}")
def delete_report(report_id: str) -> Dict[str, Any]:
    """Delete a report."""
    deleted = False
    if report_id in _JSON_REPORTS:
        del _JSON_REPORTS[report_id]
        deleted = True
    if report_id in _REPORTS:
        del _REPORTS[report_id]
        deleted = True

    if not deleted:
        raise HTTPException(status_code=404, detail="Report not found.")

    return {"status": "deleted", "report_id": report_id}


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
        # Also store into _JSON_REPORTS for unified viewing
        _JSON_REPORTS[report.report_id] = {
            "report_id": report.report_id,
            "title": report.title,
            "dataset_name": uploaded["dataset"]["name"],
            "report_type": "eda_comprehensive",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ready",
            "executive_summary": report.executive_summary,
            "dataset_overview": report.dataset_overview,
            "data_quality": report.data_quality,
            "kpis": report.kpis,
            "charts": report.charts,
            "insights": report.insights,
            "recommendations": report.recommendations,
            "forecast": report.forecast,
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
        raise HTTPException(status_code=404, detail="Report download content not found.")

    extension = {"pdf": "pdf", "excel": "xlsx", "powerpoint": "pptx"}.get(stored["format"], "pdf")
    headers = {"Content-Disposition": f'attachment; filename="{report_id}.{extension}"'}
    return Response(content=stored["content"], media_type=stored["content_type"], headers=headers)


class ExecutivePDFRequest(BaseModel):
    title: str
    command: str
    explanation: str
    kpis: dict[str, Any] = {}
    evidence_list: list[dict[str, Any]] = []
    dataset_summary: dict[str, Any] = {}
    validation_summary: dict[str, Any] = {}
    duration_ms: float | None = None


@router.post("/executive-pdf")
def create_executive_pdf(req: ExecutivePDFRequest) -> Response:
    """Generate and download a high-res multi-page Executive PDF report."""
    from backend.app.core.presentation_builder import global_presentation_engine
    try:
        pdf_bytes = global_presentation_engine.build_pdf_report(
            title=req.title,
            command=req.command,
            explanation=req.explanation,
            kpis=req.kpis,
            evidence_list=req.evidence_list,
            dataset_summary=req.dataset_summary,
            validation_summary=req.validation_summary,
            duration_ms=req.duration_ms,
        )
        headers = {"Content-Disposition": f'attachment; filename="executive_report.pdf"'}
        return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate Executive PDF: {str(e)}")


@router.post("/executive-deck")
def create_executive_deck_schema(req: ExecutivePDFRequest) -> dict[str, Any]:
    """Generate structured Executive Slide Deck model."""
    from backend.app.core.presentation_builder import global_presentation_engine
    try:
        deck = global_presentation_engine.build_deck_structure(
            title=req.title,
            command=req.command,
            explanation=req.explanation,
            kpis=req.kpis,
            evidence_list=req.evidence_list,
        )
        return deck.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate Executive Deck: {str(e)}")
