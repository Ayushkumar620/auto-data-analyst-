"""
FastAPI REST Router for Enterprise Governance, PII Redaction & Audit Logging.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
import pandas as pd
from pydantic import BaseModel

from agent.audit_logger import AuditEvent, GLOBAL_AUDIT_LOGGER
from agent.pii_guard import GLOBAL_PII_GUARD, PIIScanReport, RedactionStrategy

router = APIRouter(prefix="/governance", tags=["Enterprise Governance & Compliance"])


class ScanDataRequest(BaseModel):
    data: List[Dict[str, Any]]


class RedactDataRequest(BaseModel):
    data: List[Dict[str, Any]]
    redactions: Optional[Dict[str, RedactionStrategy]] = None


@router.post("/scan-pii", response_model=PIIScanReport)
def scan_dataset_pii(req: ScanDataRequest):
    """Scan submitted dataset sample for personally identifiable information."""
    if not req.data:
        raise HTTPException(status_code=400, detail="Data payload cannot be empty")
    df = pd.DataFrame(req.data)
    report = GLOBAL_PII_GUARD.scan_dataframe(df)

    # Log audit event
    GLOBAL_AUDIT_LOGGER.log_event(
        action="PII_SCAN",
        resource_type="dataset",
        resource_id="sample",
        details={"pii_columns_found": report.total_pii_columns},
    )
    return report


@router.post("/redact-dataset")
def redact_dataset(req: RedactDataRequest):
    """Apply masking / hashing / anonymization to dataset columns."""
    if not req.data:
        raise HTTPException(status_code=400, detail="Data payload cannot be empty")
    df = pd.DataFrame(req.data)
    df_sanitized = GLOBAL_PII_GUARD.redact_dataframe(df, redactions=req.redactions)

    # Log audit event
    GLOBAL_AUDIT_LOGGER.log_event(
        action="PII_REDACT",
        resource_type="dataset",
        resource_id="sample",
        details={"columns_redacted": list(req.redactions.keys()) if req.redactions else "auto"},
    )
    return {
        "success": True,
        "rows": df_sanitized.to_dict(orient="records"),
        "total_rows": len(df_sanitized),
    }


@router.get("/audit-logs", response_model=List[AuditEvent])
def get_audit_trail(limit: int = Query(50, ge=1, le=500)):
    """Retrieve immutable audit records in descending chronological order."""
    return GLOBAL_AUDIT_LOGGER.list_events(limit=limit)


@router.get("/audit-logs/verify")
def verify_audit_integrity():
    """Verify cryptographic SHA-256 chain integrity."""
    valid, message = GLOBAL_AUDIT_LOGGER.verify_integrity()
    return {"valid": valid, "status": message}
