from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.chat.agent import ChatAgent
from backend.app.config import UPLOAD_DIR
from backend.app.services.dataset_service import DatasetService

router = APIRouter(tags=["chat"])

_ANALYSES: dict[str, dict[str, Any]] = {}


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
