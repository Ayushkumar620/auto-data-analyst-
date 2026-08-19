"""Evidence System for the Universal Data Intelligence Engine.

Every important claim produced by the system must carry:

    claim          - the statement being made
    evidence       - the measured data that supports the claim
    calculation    - how the numbers were derived (formula / source)
    confidence     - 0..1 numeric confidence
    interpretation - plain-language meaning of the claim
    evidence_type  - FACT | OBSERVATION | CORRELATION | INFERENCE | RECOMMENDATION

The engine never claims causation from correlation alone: a CORRELATION
statement must never be upgraded to a causal FACT by any downstream consumer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

EVIDENCE_TYPES = ("FACT", "OBSERVATION", "CORRELATION", "INFERENCE", "RECOMMENDATION")


@dataclass(frozen=True)
class Evidence:
    claim: str
    evidence: str
    calculation: str
    confidence: float
    interpretation: str
    evidence_type: str = "OBSERVATION"
    source: str = "engine"
    data_points: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evidence_type not in EVIDENCE_TYPES:
            raise ValueError(
                f"evidence_type must be one of {EVIDENCE_TYPES}, got {self.evidence_type!r}"
            )
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_correlation(self) -> bool:
        return self.evidence_type == "CORRELATION"


def as_fact(claim: str, evidence: str, calculation: str, confidence: float,
            interpretation: str, source: str = "engine",
            data_points: dict[str, Any] | None = None) -> Evidence:
    """Build a FACT evidence item."""
    return Evidence(claim=claim, evidence=evidence, calculation=calculation,
                    confidence=confidence, interpretation=interpretation,
                    evidence_type="FACT", source=source, data_points=data_points or {})


def as_observation(claim: str, evidence: str, calculation: str, confidence: float,
                   interpretation: str, source: str = "engine",
                   data_points: dict[str, Any] | None = None) -> Evidence:
    """Build an OBSERVATION evidence item (directly measured pattern)."""
    return Evidence(claim=claim, evidence=evidence, calculation=calculation,
                    confidence=confidence, interpretation=interpretation,
                    evidence_type="OBSERVATION", source=source,
                    data_points=data_points or {})


def as_correlation(claim: str, evidence: str, calculation: str, confidence: float,
                   interpretation: str, source: str = "engine",
                   data_points: dict[str, Any] | None = None) -> Evidence:
    """Build a CORRELATION evidence item.

    The interpretation must not assert causation. Callers that need a causal
    statement must be explicit that it is an INFERENCE with lower confidence.
    """
    if "cause" in interpretation.casefold() or "because of" in interpretation.casefold():
        raise ValueError("A CORRELATION evidence item must not claim causation.")
    return Evidence(claim=claim, evidence=evidence, calculation=calculation,
                    confidence=confidence, interpretation=interpretation,
                    evidence_type="CORRELATION", source=source,
                    data_points=data_points or {})


def as_inference(claim: str, evidence: str, calculation: str, confidence: float,
                 interpretation: str, source: str = "engine",
                 data_points: dict[str, Any] | None = None) -> Evidence:
    """Build an INFERENCE evidence item (a reasoned step beyond direct measurement)."""
    return Evidence(claim=claim, evidence=evidence, calculation=calculation,
                    confidence=confidence, interpretation=interpretation,
                    evidence_type="INFERENCE", source=source,
                    data_points=data_points or {})


def as_recommendation(claim: str, evidence: str, calculation: str, confidence: float,
                      interpretation: str, source: str = "engine",
                      data_points: dict[str, Any] | None = None) -> Evidence:
    """Build a RECOMMENDATION evidence item (always conditional)."""
    return Evidence(claim=claim, evidence=evidence, calculation=calculation,
                    confidence=confidence, interpretation=interpretation,
                    evidence_type="RECOMMENDATION", source=source,
                    data_points=data_points or {})
