"""Request and response shapes for the chat API."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class ChatRequest:
    dataset_id: str
    message: str
    session_id: str = "default"
    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ChatRequest":
        return cls(str(payload.get("dataset_id", "")).strip(), str(payload.get("message", "")).strip(), str(payload.get("session_id") or "default").strip())

@dataclass
class ChatResponse:
    message: str
    intent: str
    status: str
    evidence: dict[str, Any] = field(default_factory=dict)
    visualization: dict[str, Any] | None = None
    suggested_questions: list[str] = field(default_factory=list)
    command_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
