"""
FastAPI REST Router for Automated Continuous Alerts (Slack, MS Teams, Email).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from agent.alert_engine import (
    AlertRuleConfig,
    DispatchedAlert,
    GLOBAL_ALERT_ENGINE,
)

router = APIRouter(prefix="/alerts", tags=["Enterprise Automated Alerting"])


class CreateAlertRuleRequest(BaseModel):
    name: str
    channel_type: str = "slack"
    webhook_url: Optional[str] = None
    email_recipient: Optional[str] = None
    trigger_on_drift: bool = True
    trigger_on_degradation: bool = True
    trigger_on_forecast_breach: bool = False
    enabled: bool = True


class TestAlertRequest(BaseModel):
    rule_id: Optional[str] = None
    title: str = "Test Observability Notification"
    message: str = "This is a verification test from the Auto Data Analyst Enterprise Alert Engine."
    severity: str = "INFO"


@router.get("/rules", response_model=List[AlertRuleConfig])
def list_alert_rules():
    """List all registered alerting channels and webhook rules."""
    return GLOBAL_ALERT_ENGINE.list_rules()


@router.post("/rules", response_model=AlertRuleConfig)
def save_alert_rule(req: CreateAlertRuleRequest):
    """Register or update an alert channel rule."""
    rule = AlertRuleConfig(
        name=req.name,
        channel_type=req.channel_type,
        webhook_url=req.webhook_url,
        email_recipient=req.email_recipient,
        trigger_on_drift=req.trigger_on_drift,
        trigger_on_degradation=req.trigger_on_degradation,
        trigger_on_forecast_breach=req.trigger_on_forecast_breach,
        enabled=req.enabled,
    )
    return GLOBAL_ALERT_ENGINE.save_rule(rule)


@router.delete("/rules/{rule_id}")
def delete_alert_rule(rule_id: str):
    """Delete an alert channel rule."""
    deleted = GLOBAL_ALERT_ENGINE.delete_rule(rule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    return {"success": True, "deleted_id": rule_id}


@router.post("/test", response_model=List[DispatchedAlert])
def dispatch_test_alert(req: TestAlertRequest):
    """Dispatch a test notification to verify webhook channel setup."""
    return GLOBAL_ALERT_ENGINE.dispatch_alert(
        title=req.title,
        message=req.message,
        severity=req.severity,
        rule_id=req.rule_id,
    )


@router.get("/history", response_model=List[DispatchedAlert])
def get_alert_dispatch_history(limit: int = Query(50, ge=1, le=200)):
    """Retrieve historical alert deliveries and statuses."""
    return GLOBAL_ALERT_ENGINE.list_history(limit=limit)
