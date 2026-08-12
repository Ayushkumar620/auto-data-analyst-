"""Coordinates fact extraction and deterministic insight generation."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from backend.app.eda.orchestrator import EDAOrchestrator

from .analyzer import FactAnalyzer
from .rules import InsightRules


class InsightEngine:
    def __init__(self) -> None:
        self.fact_analyzer = FactAnalyzer()
        self.rules = InsightRules()
        self.eda = EDAOrchestrator()

    def generate(self, dataframe: pd.DataFrame, eda_results: Dict[str, Any] | None = None) -> Dict[str, Any]:
        facts = self.fact_analyzer.extract(dataframe, eda_results or self.eda.analyze(dataframe))
        insights: List[Dict[str, Any]] = [insight.to_dict() for insight in self.rules.evaluate(facts)]
        if not insights:
            insights.append({"type": "key_finding", "title": "Limited Evidence", "severity": "info",
                             "confidence": "low", "evidence": {}, "recommendation": None,
                             "description": "The available data does not provide enough evidence for a strong business finding."})
        return {"facts": facts, "insights": insights}
