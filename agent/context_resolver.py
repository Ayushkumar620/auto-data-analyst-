"""
Context Resolution Engine for Conversational Data Analysis.

Disambiguates natural language queries, resolves anaphoric references
("it", "that", "those", "same dataset", "last year"), determines conversational intent,
and handles multi-candidate ambiguity by requesting user clarification.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from agent.conversational_schemas import (
    ConversationSession,
    ConversationalIntent,
    DatasetContext,
)


class ContextResolver:
    """
    Resolves conversational pronouns, temporal markers, and domain entities
    against the active ConversationSession state.
    """

    INTENT_KEYWORDS = [
        (re.compile(r"\b(report|generate report|briefing|executive summary|presentation)\b", re.I), ConversationalIntent.GENERATE_REPORT),
        (re.compile(r"\b(why|driver|cause|reason|investigate|root cause|what caused)\b", re.I), ConversationalIntent.INVESTIGATE),
        (re.compile(r"\b(compare|versus|vs|difference between|compared to)\b", re.I), ConversationalIntent.COMPARE),
        (re.compile(r"\b(forecast|future|next month|next quarter|next year)\b", re.I), ConversationalIntent.FORECAST),
        (re.compile(r"\b(predict|classification|regression|train model|build model)\b", re.I), ConversationalIntent.PREDICT),
        (re.compile(r"\b(anomaly|anomalies|outlier|outliers|unusual|spike)\b", re.I), ConversationalIntent.DETECT_ANOMALY),
        (re.compile(r"\b(drift|monitor|degradation|decay|data drift)\b", re.I), ConversationalIntent.MONITOR),
        (re.compile(r"\b(top \d+|bottom \d+|rank|largest|highest|lowest)\b", re.I), ConversationalIntent.DRILL_DOWN),
        (re.compile(r"\b(filter|show only|where|just)\b", re.I), ConversationalIntent.FILTER),
        (re.compile(r"\b(summarize|overview|tell me everything|quick summary)\b", re.I), ConversationalIntent.SUMMARIZE),
        (re.compile(r"\b(recommend|action|what should (we|i) do|strategy)\b", re.I), ConversationalIntent.RECOMMEND),
        (re.compile(r"\b(explain|details|breakdown|expand)\b", re.I), ConversationalIntent.EXPLAIN),
        (re.compile(r"\b(analyze|explore|inspect|study)\b", re.I), ConversationalIntent.ANALYZE),
    ]

    def resolve(
        self,
        command: str,
        session: ConversationSession,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[str, ConversationalIntent, List[str], bool, Optional[str]]:
        """
        Disambiguate command against session context.

        Returns:
            (resolved_command, intent, referenced_entities, needs_clarification, clarification_prompt)
        """
        cmd_clean = command.strip()
        resolved_cmd = cmd_clean
        entities: List[str] = []
        needs_clarification = False
        clarification_prompt: Optional[str] = None

        # 1. Detect Intent
        intent = ConversationalIntent.ANALYZE
        for pattern, mapped_intent in self.INTENT_KEYWORDS:
            if pattern.search(cmd_clean):
                intent = mapped_intent
                break

        # 2. Check for Ambiguity across Competing Metrics
        # e.g., if previous insights featured multiple metrics and user says "Why did it increase?"
        pronoun_match = re.search(r"\b(it|that|this)\b", cmd_clean, re.I)
        if pronoun_match and not session.active_metric:
            # Check if previous insights discuss multiple distinct metrics
            mentioned_metrics = set()
            for ins in session.previous_insights:
                for col in ins.affected_columns:
                    if session.dataset_context and col in session.dataset_context.numeric_columns:
                        mentioned_metrics.add(col)

            if len(mentioned_metrics) > 1:
                metrics_list = sorted(list(mentioned_metrics))
                return (
                    cmd_clean,
                    ConversationalIntent.CLARIFICATION,
                    [],
                    True,
                    f"Do you mean {' or '.join(metrics_list)}?",
                )

        # 3. Pronoun Resolution: "it" -> active_metric / active_target
        if session.dataset_context:
            primary_metric = session.dataset_context.primary_metric or (
                session.dataset_context.numeric_columns[0] if session.dataset_context.numeric_columns else None
            )
            target_metric = primary_metric

            if target_metric:
                if re.search(r"\bwhy did it\b", resolved_cmd, re.I):
                    resolved_cmd = re.sub(r"\bwhy did it\b", f"why did {target_metric}", resolved_cmd, flags=re.I)
                    entities.append(target_metric)
                elif re.search(r"\b(predict|forecast) it\b", resolved_cmd, re.I):
                    resolved_cmd = re.sub(r"\b(predict|forecast) it\b", f"\\1 {target_metric}", resolved_cmd, flags=re.I)
                    entities.append(target_metric)
                elif re.search(r"\bcompare (it|that)\b", resolved_cmd, re.I):
                    resolved_cmd = re.sub(r"\bcompare (it|that)\b", f"compare {target_metric}", resolved_cmd, flags=re.I)
                    entities.append(target_metric)
                elif re.match(r"^why\??$", resolved_cmd, re.I):
                    resolved_cmd = f"why did {target_metric} change?"
                    entities.append(target_metric)

        # 4. Dimension & Entity Resolution: "Show me North" -> "filter by region == 'North'"
        if session.dataset_context and session.dataset_context.categorical_columns:
            for cat_col in session.dataset_context.categorical_columns:
                # If command matches a category value mentioned earlier
                for ins in session.previous_insights:
                    for seg in ins.affected_segments:
                        if re.search(rf"\b{re.escape(seg)}\b", resolved_cmd, re.I):
                            entities.append(seg)
                            if re.match(rf"^(show me |breakdown for )?{re.escape(seg)}\??$", resolved_cmd, re.I):
                                resolved_cmd = f"breakdown performance for {cat_col} == '{seg}'"
                                intent = ConversationalIntent.DRILL_DOWN

        # 5. Temporal Marker Resolution: "last year" -> "compare with previous period"
        if re.search(r"\b(last year|previous year|prior period|previous quarter)\b", resolved_cmd, re.I):
            entities.append("temporal_comparison")
            if intent == ConversationalIntent.ANALYZE:
                intent = ConversationalIntent.COMPARE

        # 6. Model Context Resolution: "how reliable is the model?" -> "evaluate model <active_model>"
        if session.active_model and re.search(r"\b(the model|it|this model)\b", resolved_cmd, re.I):
            entities.append(session.active_model)
            if re.search(r"\breliab|accura|perform|evaluat\b", resolved_cmd, re.I):
                resolved_cmd = f"evaluate performance of model {session.active_model}"
                intent = ConversationalIntent.MONITOR

        # 7. Report Context Resolution
        if intent == ConversationalIntent.GENERATE_REPORT:
            entities.append("report_generation")

        return resolved_cmd, intent, entities, needs_clarification, clarification_prompt
