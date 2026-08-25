"""
Insight Ranker and Deduplication Engine.

Ranks insights by business importance, magnitude, confidence, and intent relevance,
while eliminating redundant or duplicate findings.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from agent.autonomous_analysis_schemas import (
    Insight,
    InsightCategory,
    InsightSeverity,
)
from agent.intent import UserIntent


class InsightRanker:
    """
    Evaluates, ranks, deduplicates, and synthesizes structured analytical insights.
    """

    SEVERITY_WEIGHTS = {
        InsightSeverity.CRITICAL: 1.0,
        InsightSeverity.HIGH: 0.85,
        InsightSeverity.MEDIUM: 0.65,
        InsightSeverity.LOW: 0.50,
        InsightSeverity.INFORMATIONAL: 0.40,
    }

    def deduplicate(self, insights: List[Insight]) -> List[Insight]:
        """
        Merge overlapping or duplicate insights sharing identical columns and category.
        """
        unique_insights: List[Insight] = []
        seen_signatures: Set[str] = set()

        for ins in insights:
            cols_key = "_".join(sorted(ins.affected_columns))
            signature = f"{ins.category.value}:{cols_key}"

            if signature in seen_signatures:
                # If seen, check if current has higher importance/confidence to replace
                for i, existing in enumerate(unique_insights):
                    existing_cols_key = "_".join(sorted(existing.affected_columns))
                    existing_sig = f"{existing.category.value}:{existing_cols_key}"
                    if existing_sig == signature:
                        if (ins.importance * ins.confidence) > (existing.importance * existing.confidence):
                            unique_insights[i] = ins
                        break
                continue

            seen_signatures.add(signature)
            unique_insights.append(ins)

        return unique_insights

    def rank(
        self,
        insights: List[Insight],
        user_intent: Optional[UserIntent] = None,
        top_k: Optional[int] = None,
    ) -> List[Insight]:
        """
        Score and rank insights by multi-factor objective value.
        """
        deduped = self.deduplicate(insights)
        if not deduped:
            return []

        intent_metrics = set(user_intent.metrics) if user_intent and user_intent.metrics else set()
        intent_dims = set(user_intent.dimensions) if user_intent and user_intent.dimensions else set()

        def compute_score(ins: Insight) -> float:
            # Relevance bonus if affected columns match user query
            aff_cols = set(ins.affected_columns)
            relevance = 1.0 if (aff_cols & (intent_metrics | intent_dims)) else 0.60
            sev_w = self.SEVERITY_WEIGHTS.get(ins.severity, 0.40)
            importance = ins.importance
            confidence = ins.confidence

            return (0.35 * relevance) + (0.25 * importance) + (0.25 * sev_w) + (0.15 * confidence)

        ranked = sorted(deduped, key=compute_score, reverse=True)
        if top_k and top_k > 0:
            return ranked[:top_k]
        return ranked

    def extract_recommendations(self, insights: List[Insight]) -> List[str]:
        """Collect actionable recommendations from ranked insights."""
        recs = []
        for ins in insights:
            if ins.recommended_action and ins.recommended_action not in recs:
                recs.append(ins.recommended_action)
        return recs

    def extract_limitations(self, insights: List[Insight]) -> List[str]:
        """Collect limitations and caveats from ranked insights."""
        limits = []
        for ins in insights:
            for lim in ins.limitations:
                if lim not in limits:
                    limits.append(lim)
        return limits

