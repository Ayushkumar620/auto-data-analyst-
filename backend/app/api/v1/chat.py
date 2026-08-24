from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.chat.agent import ChatAgent
from backend.app.config import UPLOAD_DIR
from backend.app.services.dataset_service import DatasetService
from agent.command_orchestrator import AutonomousCommandOrchestrator

router = APIRouter(tags=["chat"])

_ANALYSES: dict[str, dict[str, Any]] = {}
orchestrator = AutonomousCommandOrchestrator()


@router.post("/chat")
def chat(
    file: UploadFile = File(...),
    message: str = Form(...),
    session_id: str = Form("default"),
) -> dict[str, Any]:
    service = DatasetService(upload_folder=str(UPLOAD_DIR))
    try:
        uploaded = service.upload_dataset(file)
        dataset_path = UPLOAD_DIR / uploaded["dataset"]["name"]
        dataframe = service._read_dataframe(str(dataset_path))
        dataset_id = uploaded["dataset"]["id"]
        _ANALYSES[dataset_id] = {"dataframe": dataframe, "dataset_name": uploaded["dataset"]["name"]}
        response = ChatAgent().respond(dataframe, message, {"dataset_id": dataset_id, "session_id": session_id})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    return response.to_dict()


@router.post("/chat/command")
def execute_command_endpoint(
    command: str = Form(...),
    file: UploadFile | None = File(None),
    dataset_id: str | None = Form(None),
) -> dict[str, Any]:
    """Autonomous command-driven entry point: accepts natural language command and runs full DAG."""
    service = DatasetService(upload_folder=str(UPLOAD_DIR))
    try:
        if file is not None:
            uploaded = service.upload_dataset(file)
            dataset_path = UPLOAD_DIR / uploaded["dataset"]["name"]
            dataframe = service._read_dataframe(str(dataset_path))
        elif dataset_id and dataset_id in _ANALYSES:
            dataframe = _ANALYSES[dataset_id]["dataframe"]
        else:
            # Fallback to first available dataset or sample
            if _ANALYSES:
                first_id = next(iter(_ANALYSES))
                dataframe = _ANALYSES[first_id]["dataframe"]
            else:
                raise HTTPException(status_code=400, detail="No dataset provided for command execution.")

        result = orchestrator.execute_command(command, dataframe)
        return result.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Command execution failed: {exc}") from exc
