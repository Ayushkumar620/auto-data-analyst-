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
        dataset_name = uploaded["dataset"]["name"]
        dataframe = service._read_dataframe(str(dataset_path))
        eda = EDAOrchestrator().analyze(dataframe)
        result = InsightEngine().synthesize(dataframe, dataset_name=dataset_name, eda_results=eda)
        linked_dataset = link_uploaded_dataset(uploaded, resolved_workspace_id, resolved_project_id)
        dataset_id = uploaded["dataset"]["id"]
        _ANALYSES[dataset_id] = {
            "dataframe": dataframe,
            "eda": eda,
            "dataset_name": dataset_name,
            "insights": result["insights"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {exc}") from exc

    return {
        **result,
        "dataset_name": dataset_name,
        "workspace_id": resolved_workspace_id,
        "project_id": resolved_project_id,
        "workspace_dataset_id": linked_dataset["id"] if linked_dataset else None,
    }


@router.post("/synthesize")
def synthesize_cross_agent_insights(payload: dict[str, Any]) -> dict[str, Any]:
    """Universal cross-agent insight synthesis endpoint."""
    import pandas as pd
    from agent.insight_synthesis_engine import InsightSynthesisEngine
    from agent.orchestrator import UniversalOrchestrator

    orchestration_result = payload.get("orchestration_result") or payload.get("result")
    raw_data = payload.get("dataset") or payload.get("data")
    command = payload.get("command") or payload.get("user_request") or ""
    limit = payload.get("limit")
    conf_threshold = payload.get("confidence_threshold", 0.0)
    categories = payload.get("categories")

    df = None
    if isinstance(raw_data, list) and len(raw_data) > 0 and isinstance(raw_data[0], dict):
        df = pd.DataFrame(raw_data)
    elif isinstance(raw_data, dict) and "records" in raw_data:
        df = pd.DataFrame(raw_data["records"])

    # If only dataset and command are provided without orchestration_result, run orchestrator first
    if not orchestration_result and df is not None:
        orch = UniversalOrchestrator()
        orch_res = orch.orchestrate(command or "profile and analyze this dataset", df)
        orchestration_result = orch_res.to_dict()

    engine = InsightSynthesisEngine()
    report = engine.synthesize(
        orchestration_result=orchestration_result or {},
        dataframe=df,
        command=command,
    )

    report_dict = report.to_dict()
    # Apply optional filtering
    if conf_threshold > 0:
        report_dict["key_insights"] = [i for i in report_dict["key_insights"] if i.get("confidence", 1.0) >= conf_threshold]
    if categories and isinstance(categories, list):
        report_dict["key_insights"] = [i for i in report_dict["key_insights"] if i.get("category") in categories]
    if limit and isinstance(limit, int) and limit > 0:
        report_dict["key_insights"] = report_dict["key_insights"][:limit]

    return {
        "status": "success",
        "result": report_dict,
        "executive_summary": report.executive_summary,
        "key_insights": report_dict["key_insights"],
        "contradictions": report_dict["contradictions"],
        "confidence": report.overall_confidence,
    }

