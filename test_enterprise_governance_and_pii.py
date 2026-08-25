"""
Unit and Integration Tests for Enterprise PII Redaction & Immutable Audit Logging.
"""
import pandas as pd
import pytest
from agent.pii_guard import PIIGuardEngine, PIIType, RedactionStrategy
from agent.audit_logger import AuditLogger


def test_pii_detection():
    engine = PIIGuardEngine()
    df = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "user_email": ["alice@company.com", "bob@example.org", "charlie@enterprise.net"],
        "phone_num": ["+1 555-123-4567", "555-987-6543", "+44 20 7946 0958"],
        "credit_card": ["4532-1234-5678-9010", "5425-2345-6789-0123", "3782-8224-6310-0051"],
        "revenue": [100.0, 250.0, 300.0],
    })

    report = engine.scan_dataframe(df)
    assert report.has_pii is True
    assert report.total_pii_columns == 3

    detected_types = {d.column: d.pii_type for d in report.detections}
    assert detected_types["user_email"] == PIIType.EMAIL
    assert detected_types["phone_num"] == PIIType.PHONE
    assert detected_types["credit_card"] == PIIType.CREDIT_CARD


def test_pii_redaction_strategies():
    engine = PIIGuardEngine()
    df = pd.DataFrame({
        "email": ["john.doe@acme.com", "jane.smith@corp.org"],
        "cc": ["4532-1234-5678-9010", "5425-2345-6789-0123"],
    })

    # Test MASK
    df_masked = engine.redact_dataframe(df, {"email": RedactionStrategy.MASK, "cc": RedactionStrategy.MASK})
    assert df_masked["email"].iloc[0].startswith("j***@")
    assert df_masked["cc"].iloc[0].endswith("9010")

    # Test HASH
    df_hashed = engine.redact_dataframe(df, {"email": RedactionStrategy.HASH})
    assert len(df_hashed["email"].iloc[0]) == 16
    assert "@" not in df_hashed["email"].iloc[0]


def test_immutable_audit_logger_integrity():
    logger = AuditLogger()
    ev1 = logger.log_event(action="DATA_INGEST", resource_type="dataset", resource_id="ds_01", details={"rows": 100})
    ev2 = logger.log_event(action="MODEL_TRAIN", resource_type="model", resource_id="mod_01", details={"acc": 0.95})

    assert ev1.signature_hash != ""
    assert ev2.previous_hash == ev1.signature_hash

    valid, status = logger.verify_integrity()
    assert valid is True
    assert "100% tamper-evident" in status
