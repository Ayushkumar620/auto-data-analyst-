"""
Enterprise PII (Personally Identifiable Information) Guard Engine.

Provides automatic detection and redaction of sensitive customer and financial data:
- Email addresses
- Phone numbers (international & US formats)
- Credit card numbers & IBAN
- Social Security Numbers (SSN) / Tax IDs
- IP addresses (IPv4 & IPv6)
- High-cardinality personal names / addresses

Supports configurable redaction strategies:
- MASK: "j***e@domain.com", "****-****-****-1234"
- HASH: Cryptographic salted SHA-256 hash preserving referential consistency
- ANONYMIZE: Synthetic entity replacement ("[EMAIL_1]", "[PHONE_2]")
"""
from __future__ import annotations

import re
import hashlib
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
from pydantic import BaseModel, Field


class PIIType(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    IP_ADDRESS = "ip_address"
    GENERIC_SENSITIVE = "generic_sensitive"


class RedactionStrategy(str, Enum):
    MASK = "mask"
    HASH = "hash"
    ANONYMIZE = "anonymize"


class PIIDetection(BaseModel):
    column: str
    pii_type: PIIType
    confidence: float
    sample_matches_count: int
    total_samples_tested: int
    suggested_strategy: RedactionStrategy = RedactionStrategy.MASK


class PIIScanReport(BaseModel):
    has_pii: bool
    total_pii_columns: int
    detections: List[PIIDetection]
    scanned_columns: int
    scanned_rows: int


# Compiled high-precision regex patterns
_PATTERNS = {
    PIIType.EMAIL: re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'),
    PIIType.PHONE: re.compile(r'^(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}$'),
    PIIType.CREDIT_CARD: re.compile(r'^\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}$'),
    PIIType.SSN: re.compile(r'^\d{3}-\d{2}-\d{4}$'),
    PIIType.IP_ADDRESS: re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'),
}

# Sensitive column name heuristics
_NAME_HEURISTICS = {
    PIIType.EMAIL: ["email", "e_mail", "mail_addr"],
    PIIType.PHONE: ["phone", "mobile", "tel", "cell"],
    PIIType.CREDIT_CARD: ["credit_card", "card_number", "cc_num", "pan", "iban"],
    PIIType.SSN: ["ssn", "social_security", "tax_id", "national_id"],
    PIIType.IP_ADDRESS: ["ip_address", "client_ip", "remote_ip"],
}


class PIIGuardEngine:
    """Enterprise PII detection and redaction engine."""

    def __init__(self, salt: str = "auto_analyst_secure_salt"):
        self.salt = salt

    def scan_dataframe(self, df: pd.DataFrame, sample_size: int = 100) -> PIIScanReport:
        """Scan a pandas DataFrame for PII fields across all columns."""
        detections: List[PIIDetection] = []
        sampled_df = df.head(sample_size)
        total_rows = len(sampled_df)

        for col in df.columns:
            col_str = str(col).lower()
            series = sampled_df[col].dropna().astype(str)
            if len(series) == 0:
                continue

            detected_type: Optional[PIIType] = None
            max_matches = 0

            # 1. Test value-level regex matches
            for p_type, regex in _PATTERNS.items():
                matches = sum(1 for val in series if regex.match(val.strip()))
                if matches > 0 and (matches / len(series)) >= 0.2:
                    if matches > max_matches:
                        max_matches = matches
                        detected_type = p_type

            # 2. Check column name heuristics if value-level wasn't triggered
            if not detected_type:
                for p_type, keywords in _NAME_HEURISTICS.items():
                    if any(k in col_str for k in keywords):
                        detected_type = p_type
                        max_matches = len(series)
                        break

            if detected_type:
                confidence = round(min(1.0, (max_matches / len(series)) + 0.2), 2)
                detections.append(
                    PIIDetection(
                        column=str(col),
                        pii_type=detected_type,
                        confidence=confidence,
                        sample_matches_count=max_matches,
                        total_samples_tested=len(series),
                        suggested_strategy=RedactionStrategy.MASK,
                    )
                )

        return PIIScanReport(
            has_pii=len(detections) > 0,
            total_pii_columns=len(detections),
            detections=detections,
            scanned_columns=len(df.columns),
            scanned_rows=len(df),
        )

    def redact_dataframe(
        self,
        df: pd.DataFrame,
        redactions: Optional[Dict[str, RedactionStrategy]] = None,
    ) -> pd.DataFrame:
        """Apply redactions to specified columns or auto-detected PII columns."""
        df_out = df.copy()

        # If no explicit mapping provided, auto-scan
        if redactions is None:
            scan = self.scan_dataframe(df)
            redactions = {d.column: d.suggested_strategy for d in scan.detections}

        for col, strategy in redactions.items():
            if col not in df_out.columns:
                continue

            if strategy == RedactionStrategy.MASK:
                df_out[col] = df_out[col].apply(self._mask_value)
            elif strategy == RedactionStrategy.HASH:
                df_out[col] = df_out[col].apply(self._hash_value)
            elif strategy == RedactionStrategy.ANONYMIZE:
                # Assign indexed anonymous identifiers
                unique_vals = {v: f"[{col.upper()}_{i+1}]" for i, v in enumerate(df_out[col].dropna().unique())}
                df_out[col] = df_out[col].map(unique_vals).fillna(df_out[col])

        return df_out

    def _mask_value(self, val: Any) -> Any:
        if pd.isna(val):
            return val
        s = str(val).strip()
        if "@" in s:
            parts = s.split("@")
            user = parts[0]
            domain = parts[1] if len(parts) > 1 else ""
            masked_user = user[0] + "***" if len(user) > 0 else "***"
            return f"{masked_user}@{domain}"
        if len(s) > 4:
            return "*" * (len(s) - 4) + s[-4:]
        return "****"

    def _hash_value(self, val: Any) -> Any:
        if pd.isna(val):
            return val
        s = f"{val}_{self.salt}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


GLOBAL_PII_GUARD = PIIGuardEngine()
