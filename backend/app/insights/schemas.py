"""JSON-friendly contracts returned by the insight engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal

InsightType = Literal["key_finding", "trend", "anomaly", "risk", "opportunity"]
Severity = Literal["info", "warning", "critical"]
Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class Insight:
    type: InsightType
    title: str
    description: str
    severity: Severity = "info"
    confidence: Confidence = "high"
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendation: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
