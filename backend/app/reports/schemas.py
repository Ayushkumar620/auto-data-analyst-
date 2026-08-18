from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class Report:
    report_id: str; dataset_id: str; title: str; executive_summary: str
    dataset_overview: dict[str, Any]; data_quality: dict[str, Any]; kpis: list[dict[str, Any]]
    charts: list[dict[str, Any]] = field(default_factory=list); insights: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list); forecast: dict[str, Any] = field(default_factory=dict)
    methodology: dict[str, Any] = field(default_factory=dict); appendix: dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]: return asdict(self)
