from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class ForecastRequest:
    dataset_id: str
    target: str | None = None
    date_column: str | None = None
    horizon: int = 3
    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ForecastRequest":
        return cls(str(payload.get("dataset_id", "")).strip(), payload.get("target"), payload.get("date_column"), int(payload.get("horizon", 3)))

@dataclass
class ForecastResult:
    target: str
    date_column: str
    frequency: str
    horizon: int
    model: str
    metrics: dict[str, float | None]
    forecast: list[dict[str, Any]]
    historical_period: dict[str, str]
    limitations: list[str] = field(default_factory=list)
    visualization: dict[str, Any] | None = None
    def to_dict(self) -> dict[str, Any]: return asdict(self)
