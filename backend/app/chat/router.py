"""HTTP API for safe, evidence-backed dataset chat."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from flask import Blueprint, jsonify, request
from backend.app.api.insights import _analyses
from .agent import ChatAgent
from .schemas import ChatRequest

chat_bp = Blueprint("chat_bp", __name__, url_prefix="/api")
_sessions: dict[tuple[str, str], list[dict[str, Any]]] = {}

@chat_bp.route("/chat", methods=["POST"])
def chat() -> Any:
    payload = request.get_json(silent=True) or {}
    chat_request = ChatRequest.from_payload(payload)
    if not chat_request.dataset_id or not chat_request.message:
        return jsonify({"message": "'dataset_id' and 'message' are required.", "intent": "validation", "status": "error", "evidence": {}, "visualization": None, "suggested_questions": []}), 400
    analysis = _analyses.get(chat_request.dataset_id)
    if analysis is None:
        return jsonify({"message": f"Dataset '{chat_request.dataset_id}' was not found in this server session.", "intent": "dataset", "status": "not_found", "evidence": {}, "visualization": None, "suggested_questions": []}), 404
    key = (chat_request.dataset_id, chat_request.session_id); history = _sessions.setdefault(key, [])
    prior = history[-1].get("evidence", {}) if history else {}
    response = ChatAgent().respond(analysis["dataframe"], chat_request.message, prior)
    record = {"user_message": chat_request.message, "intent": response.intent, "tool_result": response.evidence, "ai_response": response.message, "timestamp": datetime.now(timezone.utc).isoformat()}
    history.append(record)
    analysis.setdefault("chat_history", []).append(record)
    return jsonify(response.to_dict())

@chat_bp.route("/chat/history/<dataset_id>", methods=["GET"])
def history(dataset_id: str) -> Any:
    session_id = request.args.get("session_id", "default")
    return jsonify({"dataset_id": dataset_id, "session_id": session_id, "history": _sessions.get((dataset_id, session_id), [])})
