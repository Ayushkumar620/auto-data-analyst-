from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.app.chat.agent import ChatAgent
from backend.app.config import UPLOAD_DIR
from backend.app.services.dataset_service import DatasetService
from agent.command_orchestrator import AutonomousCommandOrchestrator
from agent.conversational_analyst import ConversationalAnalystAgent
from agent.json_utils import sanitize_for_json

router = APIRouter(tags=["chat"])

_ANALYSES: dict[str, dict[str, Any]] = {}
orchestrator = AutonomousCommandOrchestrator()
conversational_agent = ConversationalAnalystAgent()


class ChatSessionMessageRequest(BaseModel):
    message: str = Field(..., description="Natural language analytical message or question")
    session_id: str = Field("default_session", description="Conversation session identifier")
    dataset: Optional[List[Dict[str, Any]]] = Field(None, description="In-memory dataset records or preview")
    dataset_name: Optional[str] = Field("dataset", description="Active dataset name")


class ChatSessionMessageResponse(BaseModel):
    session_id: str
    message: str
    response: str
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    dataset_context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str


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


@router.post("/chat/session", response_model=ChatSessionMessageResponse)
def handle_conversational_session(req: ChatSessionMessageRequest) -> ChatSessionMessageResponse:
    """Multi-turn conversational session endpoint powered by ConversationalAnalystAgent."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Prepare DataFrame if provided
    df = None
    if req.dataset:
        try:
            df = pd.DataFrame(req.dataset)
        except Exception:
            df = None

    try:
        resp_text, evidence_objs, turn_meta = conversational_agent.chat(
            command=req.message,
            session_id=req.session_id,
            data=df,
        )

        # Convert Evidence objects to standard dicts
        evidence_dicts: List[Dict[str, Any]] = []
        for ev in evidence_objs:
            if hasattr(ev, "model_dump"):
                evidence_dicts.append(ev.model_dump())
            elif hasattr(ev, "dict"):
                evidence_dicts.append(ev.dict())
            elif isinstance(ev, dict):
                evidence_dicts.append(ev)
            else:
                evidence_dicts.append({"claim": str(ev)})

        session = conversational_agent.get_or_create_session(req.session_id)
        ds_ctx = None
        if session.dataset_context:
            ds_ctx = {
                "dataset_name": session.dataset_context.dataset_name,
                "row_count": session.dataset_context.row_count,
                "column_count": session.dataset_context.column_count,
                "numeric_columns": session.dataset_context.numeric_columns,
                "categorical_columns": session.dataset_context.categorical_columns,
            }

        return ChatSessionMessageResponse(
            session_id=req.session_id,
            message=req.message,
            response=resp_text,
            evidence=sanitize_for_json(evidence_dicts),
            dataset_context=sanitize_for_json(ds_ctx),
            metadata=sanitize_for_json(turn_meta),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Conversational analyst error: {str(exc)}") from exc


@router.get("/chat/session/{session_id}")
def get_conversational_session(session_id: str) -> dict[str, Any]:
    """Retrieve full turn history and metadata for a conversation session."""
    session = conversational_agent.get_or_create_session(session_id)
    return {
        "session_id": session.session_id,
        "turns_count": len(session.turns),
        "dataset_name": session.dataset_context.dataset_name if session.dataset_context else None,
        "turns": [
            {
                "user_message": t.user_message,
                "assistant_response": t.assistant_response,
                "intent": t.resolved_intent.value if hasattr(t.resolved_intent, "value") else str(t.resolved_intent),
                "timestamp": t.timestamp,
                "evidence_count": len(t.evidence),
            }
            for t in session.turns
        ],
    }
