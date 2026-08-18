"""Coordinates fact extraction and insight generation.

Numerical facts are always computed deterministically with pandas. The optional
LLM layer only writes narrative explanations around those verified facts and
can never invent statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from backend.app.eda.orchestrator import EDAOrchestrator

from .analyzer import FactAnalyzer
from .interpreter import InsightInterpreter
from .rules import InsightRules
from .schemas import Insight


class InsightEngine:
    def __init__(self) -> None:
        self.fact_analyzer = FactAnalyzer()
        self.rules = InsightRules()
        self.interpreter = InsightInterpreter()
        self.eda = EDAOrchestrator()

    def generate(self, dataframe: pd.DataFrame, eda_results: Dict[str, Any] | None = None) -> Dict[str, Any]:
        eda = eda_results or self.eda.analyze(dataframe)
        facts = self.fact_analyzer.extract(dataframe, eda)
        insights: List[Insight] = self.rules.evaluate(facts)
        if not insights:
            insights.append(Insight(
                type="key_finding",
                title="Limited Evidence",
                severity="info",
                confidence="low",
                evidence={},
                recommendation=None,
                description="The available data does not provide enough evidence for a strong business finding.",
            ))
        enriched = self.interpreter.enrich("dataset", facts, insights)
        return {"facts": facts, "insights": [insight.to_dict() for insight in enriched]}

    def synthesize(self, dataframe: pd.DataFrame, dataset_name: str = "dataset",
                   eda_results: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Return a categorized, recommendation-complete insight bundle.

        Guarantees an entry for each supported category (key finding, trend,
        anomaly, risk, opportunity, recommendation) so the frontend and report
        layer can render a complete overview even when rules find no signal.
        """
        result = self.generate(dataframe, eda_results)
        facts = result["facts"]
        insights: List[Dict[str, Any]] = result["insights"]

        by_type: Dict[str, List[Dict[str, Any]]] = {
            "key_finding": [], "trend": [], "anomaly": [],
            "risk": [], "opportunity": [], "recommendation": [],
        }
        for item in insights:
            by_type.setdefault(item["type"], []).append(item)

        recommendations = self.rules.recommendations(facts)
        if recommendations:
            by_type["recommendation"] = [
                {
                    "type": "recommendation",
                    "title": "Recommended Actions",
                    "description": " ".join(recommendations),
                    "severity": "info",
                    "confidence": "high",
                    "evidence": facts,
                    "recommendation": " | ".join(recommendations),
                    "source": "rule",
                }
            ]

        for category, label in (
            ("key_finding", "Key Findings"),
            ("trend", "Trends"),
            ("anomaly", "Anomalies"),
            ("risk", "Risks"),
            ("opportunity", "Opportunities"),
            ("recommendation", "Recommendations"),
        ):
            if not by_type[category]:
                by_type[category] = [Insight(
                    type=category,
                    title=f"No {label} Detected",
                    severity="info",
                    confidence="medium",
                    evidence=facts,
                    recommendation=None,
                    description=(f"No {label.lower()} were identified in the available evidence. "
                                 "The dataset may not contain enough information to support one."),
                ).to_dict()]

        return {"facts": facts, "insights": insights, "categories": by_type}
