"""Tests for Enterprise RBAC, PII Data Masking, and Tamper-Evident Audit Ledger.

Verifies:
1. EnterpriseAccessControlEngine role permission matrix (ADMIN, ANALYST, VIEWER)
2. Enforcement of authorization checks and PermissionError exceptions
3. Automated and custom column-level PII data masking for non-admin users
4. Row-level security (RLS) filtering based on user attributes
5. Cryptographic SHA-256 hash chaining and audit ledger integrity verification
6. Detection of tampering / altered records in the audit trail
"""
import pandas as pd
import pytest

from backend.app.core.rbac_audit import (
    EnterpriseAccessControlEngine,
    UserRole,
    Permission,
    AuditRecord,
    global_access_control,
)


@pytest.fixture
def sensitive_customer_df():
    return pd.DataFrame({
        "customer_id": [101, 102, 103],
        "name": ["Alice Smith", "Bob Jones", "Charlie Brown"],
        "email": ["alice.smith@enterprise.com", "bob@smb.org", "charlie.b@gmail.com"],
        "phone": ["+1-555-0199", "+1-555-0288", "+1-555-0377"],
        "ssn": ["987-65-4321", "123-45-6789", "555-12-3456"],
        "region": ["North", "South", "North"],
        "revenue": [50000.0, 12000.0, 35000.0],
    })


def test_role_permissions_matrix():
    """Verify role permissions hierarchy."""
    rbac = EnterpriseAccessControlEngine()

    # Admin has all permissions
    assert rbac.has_permission(UserRole.ADMIN, Permission.EXECUTE_SANDBOX) is True
    assert rbac.has_permission(UserRole.ADMIN, Permission.MANAGE_USERS) is True

    # Analyst can execute analysis and train models, but cannot manage users
    assert rbac.has_permission(UserRole.ANALYST, Permission.EXECUTE_ANALYSIS) is True
    assert rbac.has_permission(UserRole.ANALYST, Permission.TRAIN_MODEL) is True
    assert rbac.has_permission(UserRole.ANALYST, Permission.MANAGE_USERS) is False

    # Viewer can only read and generate reports
    assert rbac.has_permission(UserRole.VIEWER, Permission.READ_DATASET) is True
    assert rbac.has_permission(UserRole.VIEWER, Permission.GENERATE_REPORT) is True
    assert rbac.has_permission(UserRole.VIEWER, Permission.TRAIN_MODEL) is False
    assert rbac.has_permission(UserRole.VIEWER, Permission.EXECUTE_SANDBOX) is False

    # Authorize raises PermissionError
    with pytest.raises(PermissionError):
        rbac.authorize(UserRole.VIEWER, Permission.EXECUTE_SANDBOX)


def test_pii_data_masking(sensitive_customer_df):
    """Verify PII fields (email, phone, ssn) are masked for non-admin roles."""
    rbac = EnterpriseAccessControlEngine()

    # Admin gets unmasked data
    admin_df = rbac.apply_data_masking(sensitive_customer_df, role=UserRole.ADMIN)
    assert admin_df["email"].iloc[0] == "alice.smith@enterprise.com"
    assert admin_df["ssn"].iloc[0] == "987-65-4321"

    # Analyst gets masked PII
    analyst_df = rbac.apply_data_masking(sensitive_customer_df, role=UserRole.ANALYST)
    assert "@enterprise.com" in analyst_df["email"].iloc[0]
    assert analyst_df["email"].iloc[0] != "alice.smith@enterprise.com"
    assert "***" in analyst_df["email"].iloc[0]
    assert analyst_df["ssn"].iloc[0].startswith("****")
    assert analyst_df["phone"].iloc[0].startswith("****")


def test_row_level_security(sensitive_customer_df):
    """Verify row filtering according to user tenancy and regional attributes."""
    rbac = EnterpriseAccessControlEngine()

    # User with North region attribute
    user_attrs = {"region": "North"}
    rls_df = rbac.apply_row_level_security(sensitive_customer_df, user_attributes=user_attrs, role=UserRole.ANALYST)

    assert len(rls_df) == 2
    assert set(rls_df["region"]) == {"North"}

    # Admin bypasses RLS
    admin_rls = rbac.apply_row_level_security(sensitive_customer_df, user_attributes=user_attrs, role=UserRole.ADMIN)
    assert len(admin_rls) == 3


def test_tamper_evident_audit_ledger_and_integrity_check():
    """Verify cryptographic SHA-256 hash chaining and tamper detection."""
    rbac = EnterpriseAccessControlEngine()

    # Log 3 events
    rec1 = rbac.log_event("user_1", UserRole.ANALYST, "EXECUTE_ANALYSIS", "dataset_001", metadata={"query": "sales"})
    rec2 = rbac.log_event("user_2", UserRole.ADMIN, "TRAIN_MODEL", "dataset_001", metadata={"algo": "XGBoost"})
    rec3 = rbac.log_event("user_1", UserRole.VIEWER, "GENERATE_REPORT", "report_001")

    assert rec1.prev_hash == "0" * 64
    assert rec2.prev_hash == rec1.record_hash
    assert rec3.prev_hash == rec2.record_hash

    # Integrity verification must pass
    is_valid, err = rbac.verify_ledger_integrity()
    assert is_valid is True
    assert err is None

    # Simulate malicious record tampering
    rbac._audit_trail[1].action = "TAMPERED_ACTION"

    # Integrity verification must catch the tampering
    is_valid_after_tamper, tamper_err = rbac.verify_ledger_integrity()
    assert is_valid_after_tamper is False
    assert "tamper" in str(tamper_err).lower() or "broken" in str(tamper_err).lower()

