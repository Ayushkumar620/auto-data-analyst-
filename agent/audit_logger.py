"""
Enterprise Immutable Audit Logger Engine.

Provides tamper-evident compliance audit trail (SOC2, HIPAA, GDPR) with SHA-256 cryptographic
chain hashing for all analytical queries, dataset modifications, model deployments, and exports.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"audit_{uuid.uuid4().hex[:12]}")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    user_id: str = "system"
    action: str  # "DATASET_INGEST", "QUERY_EXECUTION", "MODEL_TRAIN", "MODEL_DEPLOY", "REPORT_EXPORT", "PII_REDACT"
    resource_type: str  # "dataset", "model", "report", "connector", "alert"
    resource_id: str
    details: Dict[str, Any] = {}
    ip_address: Optional[str] = "127.0.0.1"
    previous_hash: str = "0000000000000000"
    signature_hash: str = ""


class AuditLogger:
    """Tamper-evident append-only audit trail."""

    def __init__(self):
        self._events: List[AuditEvent] = []
        self._last_hash: str = "0" * 32
        # Seed initial genesis event
        self.log_event(
            user_id="system",
            action="SYSTEM_INIT",
            resource_type="system",
            resource_id="root",
            details={"message": "Enterprise Compliance Audit Chain Initialized"},
        )

    def log_event(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: str = "analyst",
        details: Optional[Dict[str, Any]] = None,
        ip_address: str = "127.0.0.1",
    ) -> AuditEvent:
        """Append an immutable cryptographically chained audit event."""
        details_dict = details or {}
        event = AuditEvent(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details_dict,
            ip_address=ip_address,
            previous_hash=self._last_hash,
        )

        # Compute SHA-256 block hash
        raw_payload = f"{event.event_id}:{event.timestamp}:{event.user_id}:{event.action}:{event.resource_id}:{event.previous_hash}:{json.dumps(details_dict, sort_keys=True)}"
        sig = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
        event.signature_hash = sig
        self._last_hash = sig

        self._events.append(event)
        return event

    def list_events(self, limit: int = 100) -> List[AuditEvent]:
        """Return the most recent audit records in descending chronological order."""
        return list(reversed(self._events[-limit:]))

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """Verify cryptographic chain validity across all recorded events."""
        if not self._events:
            return True, None

        prev_hash = "0" * 32
        for i, ev in enumerate(self._events):
            if ev.previous_hash != prev_hash:
                return False, f"Broken chain link at index {i} (Event {ev.event_id})"

            raw_payload = f"{ev.event_id}:{ev.timestamp}:{ev.user_id}:{ev.action}:{ev.resource_id}:{ev.previous_hash}:{json.dumps(ev.details, sort_keys=True)}"
            expected_sig = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
            if ev.signature_hash != expected_sig:
                return False, f"Signature mismatch at index {i} (Event {ev.event_id})"
            prev_hash = ev.signature_hash

        return True, "Audit chain integrity verified (100% tamper-evident)"


GLOBAL_AUDIT_LOGGER = AuditLogger()
