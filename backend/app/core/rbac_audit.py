"""Enterprise Role-Based Access Control (RBAC), Data Masking, & Tamper-Evident Audit Logging.

Provides:
1. Granular RBAC permission hierarchy (ADMIN, ANALYST, VIEWER)
2. Column-level sensitive PII data masking (Emails, Phones, SSN, Credit Cards)
3. Row-level security (RLS) policy enforcement
4. Cryptographic tamper-evident SHA-256 chained audit ledger for regulatory compliance
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd


class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class Permission(str, Enum):
    READ_DATASET = "read_dataset"
    UPLOAD_DATASET = "upload_dataset"
    EXECUTE_ANALYSIS = "execute_analysis"
    TRAIN_MODEL = "train_model"
    EXECUTE_SQL = "execute_sql"
    EXECUTE_SANDBOX = "execute_sandbox"
    GENERATE_REPORT = "generate_report"
    MANAGE_WORKSPACE = "manage_workspace"
    MANAGE_USERS = "manage_users"


ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.ADMIN: {
        Permission.READ_DATASET,
        Permission.UPLOAD_DATASET,
        Permission.EXECUTE_ANALYSIS,
        Permission.TRAIN_MODEL,
        Permission.EXECUTE_SQL,
        Permission.EXECUTE_SANDBOX,
        Permission.GENERATE_REPORT,
        Permission.MANAGE_WORKSPACE,
        Permission.MANAGE_USERS,
    },
    UserRole.ANALYST: {
        Permission.READ_DATASET,
        Permission.UPLOAD_DATASET,
        Permission.EXECUTE_ANALYSIS,
        Permission.TRAIN_MODEL,
        Permission.EXECUTE_SQL,
        Permission.EXECUTE_SANDBOX,
        Permission.GENERATE_REPORT,
    },
    UserRole.VIEWER: {
        Permission.READ_DATASET,
        Permission.GENERATE_REPORT,
    },
}


@dataclass
class AuditRecord:
    """A single cryptographically chained audit ledger event."""
    event_id: str
    timestamp: str
    user_id: str
    role: str
    action: str
    resource_id: str
    status: str
    prev_hash: str
    record_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "role": self.role,
            "action": self.action,
            "resource_id": self.resource_id,
            "status": self.status,
            "prev_hash": self.prev_hash,
            "record_hash": self.record_hash,
            "metadata": self.metadata,
        }


class EnterpriseAccessControlEngine:
    """Enforces RBAC permissions, PII masking, RLS filtering, and cryptographic audit logging."""

    # PII Column Heuristics
    PII_COLUMN_PATTERNS = re.compile(
        r"(email|ssn|social_security|credit_card|card_num|phone|password|secret|salary)",
        re.I
    )

    def __init__(self):
        self._audit_trail: List[AuditRecord] = []
        self._last_hash: str = "0" * 64

    # ------------------------------------------------------------------
    # 1. RBAC Permissions
    # ------------------------------------------------------------------
    def has_permission(self, role: Union[str, UserRole], permission: Union[str, Permission]) -> bool:
        """Check whether a user role possesses a specific permission."""
        try:
            r_enum = UserRole(str(role).lower())
            p_enum = Permission(str(permission).lower())
        except ValueError:
            return False

        allowed = ROLE_PERMISSIONS.get(r_enum, set())
        return p_enum in allowed

    def authorize(self, role: Union[str, UserRole], permission: Union[str, Permission]) -> None:
        """Raise PermissionError if role lacks required permission."""
        if not self.has_permission(role, permission):
            raise PermissionError(f"Role '{role}' is unauthorized to perform action '{permission}'.")

    # ------------------------------------------------------------------
    # 2. Dynamic Column-Level PII Data Masking
    # ------------------------------------------------------------------
    def apply_data_masking(
        self,
        df: pd.DataFrame,
        role: Union[str, UserRole],
        custom_mask_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Mask sensitive PII fields (e.g. emails, phone numbers, salaries) for non-admin roles.
        """
        r_str = str(role).lower()
        if r_str == UserRole.ADMIN.value:
            # Admins view unmasked raw data
            return df.copy()

        masked_df = df.copy()
        target_cols = set(custom_mask_cols or [])

        # Auto-detect PII columns by header regex
        for col in masked_df.columns:
            if self.PII_COLUMN_PATTERNS.search(str(col)):
                target_cols.add(col)

        for col in target_cols:
            if col in masked_df.columns:
                masked_df[col] = masked_df[col].apply(self._mask_scalar_value)

        return masked_df

    def _mask_scalar_value(self, val: Any) -> Any:
        """Mask a single scalar value while preserving type structure."""
        if pd.isna(val) or val is None:
            return val

        val_str = str(val).strip()
        # Email masking: j***e@example.com
        if "@" in val_str:
            parts = val_str.split("@")
            user, domain = parts[0], parts[1]
            if len(user) <= 2:
                masked_user = "**"
            else:
                masked_user = user[0] + "***" + user[-1]
            return f"{masked_user}@{domain}"

        # Numeric / general string masking
        if len(val_str) > 4:
            return "****" + val_str[-4:]
        return "****"

    # ------------------------------------------------------------------
    # 3. Row-Level Security (RLS) Policies
    # ------------------------------------------------------------------
    def apply_row_level_security(
        self,
        df: pd.DataFrame,
        user_attributes: Dict[str, Any],
        role: Union[str, UserRole] = UserRole.ANALYST,
    ) -> pd.DataFrame:
        """
        Filter dataset rows according to user tenancy / team / region attributes.
        """
        if str(role).lower() == UserRole.ADMIN.value:
            return df.copy()

        filtered_df = df.copy()
        for attr_key, attr_val in user_attributes.items():
            # Check if column matches user attribute key (e.g. 'region', 'team_id')
            matched_col = next((c for c in filtered_df.columns if c.lower() == attr_key.lower()), None)
            if matched_col and attr_val is not None:
                if isinstance(attr_val, list):
                    filtered_df = filtered_df[filtered_df[matched_col].isin(attr_val)]
                else:
                    filtered_df = filtered_df[filtered_df[matched_col] == attr_val]

        return filtered_df

    # ------------------------------------------------------------------
    # 4. Cryptographic Tamper-Evident Audit Ledger
    # ------------------------------------------------------------------
    def log_event(
        self,
        user_id: str,
        role: Union[str, UserRole],
        action: str,
        resource_id: str,
        status: str = "SUCCESS",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """Log an immutable audit event chained with SHA-256 hash."""
        now_ts = datetime.utcnow().isoformat()
        event_id = f"evt_{len(self._audit_trail) + 1:06d}_{int(time.time()*1000)}"
        meta = metadata or {}

        # Construct payload string for hashing
        payload_dict = {
            "event_id": event_id,
            "timestamp": now_ts,
            "user_id": user_id,
            "role": str(role),
            "action": action,
            "resource_id": resource_id,
            "status": status,
            "prev_hash": self._last_hash,
            "metadata": meta,
        }
        payload_str = json.dumps(payload_dict, sort_keys=True)
        current_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

        record = AuditRecord(
            event_id=event_id,
            timestamp=now_ts,
            user_id=user_id,
            role=str(role),
            action=action,
            resource_id=resource_id,
            status=status,
            prev_hash=self._last_hash,
            record_hash=current_hash,
            metadata=meta,
        )

        self._audit_trail.append(record)
        self._last_hash = current_hash
        return record

    def verify_ledger_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Cryptographically verify that no records in the audit trail have been modified,
        inserted out of order, or tampered with.
        
        Returns:
            Tuple of (is_valid, error_description)
        """
        expected_prev = "0" * 64

        for idx, rec in enumerate(self._audit_trail):
            # Check prev_hash link
            if rec.prev_hash != expected_prev:
                return False, f"Hash chain broken at record index {idx} ({rec.event_id}): expected prev {expected_prev}, got {rec.prev_hash}"

            # Recompute record hash
            payload_dict = {
                "event_id": rec.event_id,
                "timestamp": rec.timestamp,
                "user_id": rec.user_id,
                "role": rec.role,
                "action": rec.action,
                "resource_id": rec.resource_id,
                "status": rec.status,
                "prev_hash": rec.prev_hash,
                "metadata": rec.metadata,
            }
            computed_hash = hashlib.sha256(json.dumps(payload_dict, sort_keys=True).encode("utf-8")).hexdigest()

            if computed_hash != rec.record_hash:
                return False, f"Tampered record detected at index {idx} ({rec.event_id}): hash mismatch!"

            expected_prev = rec.record_hash

        return True, None

    def get_audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return the most recent audit records."""
        return [r.to_dict() for r in self._audit_trail[-limit:]]


# Global singleton instance
global_access_control = EnterpriseAccessControlEngine()
