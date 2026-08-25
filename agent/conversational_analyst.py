"""
Master Conversational Analyst Agent.

Orchestrates:
User Command
      ↓
Session Management & Context Isolation
      ↓
ContextResolver (Pronoun & Entity Resolution)
      ↓
Active Dataset / Analysis Context Tracking
      ↓
Autonomous Analysis & Model Engines
      ↓
Evidence-First Natural Language Response
      ↓
Memory Retention & Traceable ConversationTurn Logging
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from agent.autonomous_analysis_schemas import (
    AnalysisDepth,
    AutonomousAnalysisRequest,
    AutonomousAnalysisResult,
    Insight,
)
from agent.autonomous_analyst_agent import AutonomousAnalystAgent
from agent.base import BaseAgent
from agent.context_resolver import ContextResolver
from agent.conversational_schemas import (
    ConversationSession,
    ConversationSummary,
    ConversationTurn,
    ConversationalIntent,
    DatasetContext,
    GeneratedReport,
    ReportType,
)
from agent.evidence_report_generator import EvidenceReportGenerator
from agent.intent import UserIntent
from agent.schemas import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)


class ConversationalAnalystAgent(BaseAgent):
    """
    Stateful Conversational Senior Data Analyst Agent maintaining multi-turn context,
    resolving ambiguous references, generating evidence-backed responses, and creating reports.
    """
    name = "Conversational Analyst Agent"
    role = "lead_conversational_analyst"
    description = "Maintains stateful conversational analytical context, resolves anaphoric queries, and generates evidence-backed reports."

    def __init__(self, data: Optional[Any] = None, max_turns_limit: int = 20):
        super().__init__(data=data)
        self.max_turns_limit = max_turns_limit
        self.resolver = ContextResolver()
        self.analyst_agent = AutonomousAnalystAgent()
        self.report_generator = EvidenceReportGenerator()
        self._sessions: Dict[str, ConversationSession] = {}

    def get_or_create_session(self, session_id: str) -> ConversationSession:
        """Retrieve or initialize an isolated conversation session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(session_id=session_id)
        session = self._sessions[session_id]
        session.update_timestamp()
        return session

    def _update_dataset_context(self, session: ConversationSession, df: pd.DataFrame, dataset_name: str = "dataset"):
        """Extract and persist dataset metadata into session context."""
        session.active_dataset = df
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(include=["object", "category", "string", "str"]).columns)
        date_cols = []
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_cols.append(c)
            elif any(t in c.lower() for t in ("date", "time", "year", "month")):
                try:
                    pd.to_datetime(df[c].dropna().head(5))
                    date_cols.append(c)
                except Exception:
                    pass

        session.dataset_context = DatasetContext(
            dataset_id=f"ds_{abs(hash(dataset_name)) % 100000:05d}",
            dataset_name=dataset_name,
            row_count=len(df),
            column_count=len(df.columns),
            numeric_columns=num_cols,
            categorical_columns=cat_cols,
            date_columns=date_cols,
            primary_metric=num_cols[0] if num_cols else None,
            primary_dimension=cat_cols[0] if cat_cols else None,
        )

    def summarize_session(self, session: ConversationSession) -> ConversationSummary:
        """Create condensed summary of conversation history."""
        ds_name = session.dataset_context.dataset_name if session.dataset_context else None
        findings = [ins.summary for ins in session.previous_insights[:5]]
        metrics = {}
        if session.dataset_context:
            metrics["rows"] = session.dataset_context.row_count
            metrics["columns"] = session.dataset_context.column_count

        return ConversationSummary(
            session_id=session.session_id,
            active_dataset=ds_name,
            important_findings=findings,
            important_metrics=metrics,
            current_model=session.active_model,
        )

    def chat(
        self,
        command: str,
        session_id: str = "default_session",
        data: Optional[Any] = None,
    ) -> Tuple[str, List[Evidence], Dict[str, Any]]:
        """
        Process a single natural language conversational turn.

        Returns:
            (response_text, evidence_list, turn_metadata)
        """
        session = self.get_or_create_session(session_id)

        # 1. Update Dataset Context if provided
        if data is not None:
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, (dict, list)):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame()
            if not df.empty:
                self._update_dataset_context(session, df)
        else:
            df = session.active_dataset

        # 2. Context & Reference Resolution
        resolved_cmd, intent, entities, needs_clarification, prompt = self.resolver.resolve(
            command=command,
            session=session,
            df=df,
        )

        # 3. Handle Ambiguity / Clarification
        if needs_clarification and prompt:
            turn = ConversationTurn(
                session_id=session_id,
                user_message=command,
                resolved_intent=ConversationalIntent.CLARIFICATION,
                referenced_entities=entities,
                assistant_response=prompt,
            )
            session.turns.append(turn)
            return prompt, [], {"needs_clarification": True, "prompt": prompt}

        # 4. Handle Report Generation Request
        if intent == ConversationalIntent.GENERATE_REPORT:
            # Determine ReportType
            r_type = ReportType.ANALYST_REPORT
            if re.search(r"\bquick|brief|short\b", command, re.I):
                r_type = ReportType.QUICK_SUMMARY
            elif re.search(r"\bexecutive|business|strategic\b", command, re.I):
                r_type = ReportType.EXECUTIVE_REPORT
            elif re.search(r"\btechnical|data science|statistical\b", command, re.I):
                r_type = ReportType.TECHNICAL_REPORT

            report = self.report_generator.generate_report(session=session, report_type=r_type)
            turn = ConversationTurn(
                session_id=session_id,
                user_message=command,
                resolved_intent=intent,
                referenced_entities=entities,
                evidence=report.evidence,
                assistant_response=report.markdown_content,
                result={"report_id": report.report_id, "report_type": r_type.value},
            )
            session.turns.append(turn)
            return report.markdown_content, report.evidence, {"report": report.to_dict()}

        # 5. Handle Analytical Execution
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            resp = "No active dataset found in this session. Please provide or upload a dataset to begin analysis."
            return resp, [], {"error": "no_data"}

        # Create structured intent for autonomous analysis
        target_metric = session.dataset_context.primary_metric if session.dataset_context else None
        target_dim = session.dataset_context.primary_dimension if session.dataset_context else None

        user_intent = UserIntent(
            intent_type=intent.value,
            objective=resolved_cmd,
            metrics=[target_metric] if target_metric else [],
            dimensions=[target_dim] if target_dim else [],
            original_command=command,
        )

        analysis_req = AutonomousAnalysisRequest(
            dataset=df,
            user_intent=user_intent,
            analysis_depth=AnalysisDepth.STANDARD,
        )

        analysis_res: AutonomousAnalysisResult = self.analyst_agent.analyze(analysis_req)

        # Merge insights into session
        for ins in analysis_res.insights:
            session.previous_insights.append(ins)

        # Cap memory insight collection
        if len(session.previous_insights) > 50:
            session.previous_insights = session.previous_insights[-50:]

        # Formulate response
        response_lines = [f"📊 **Analysis for: \"{command}\"**\n"]
        if analysis_res.insights:
            for idx, ins in enumerate(analysis_res.insights[:4], start=1):
                response_lines.append(f"{idx}. **{ins.title}:** {ins.summary}")
        else:
            response_lines.append(analysis_res.summary)

        if analysis_res.recommendations:
            response_lines.append(f"\n💡 **Recommendation:** {analysis_res.recommendations[0]}")

        final_response = "\n".join(response_lines)

        # 6. Log Turn
        turn = ConversationTurn(
            session_id=session_id,
            user_message=command,
            resolved_intent=intent,
            referenced_entities=entities,
            evidence=analysis_res.evidence,
            assistant_response=final_response,
            result={"key_metrics": analysis_res.key_metrics, "status": analysis_res.status},
        )
        session.turns.append(turn)

        # 7. Memory Retention Limits
        if len(session.turns) > self.max_turns_limit:
            session.turns = session.turns[-self.max_turns_limit:]

        return final_response, analysis_res.evidence, {
            "resolved_command": resolved_cmd,
            "intent": intent.value,
            "session_id": session_id,
            "turn_id": turn.turn_id,
        }

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """Standardized BaseAgent interface for conversational execution."""
        self._start()
        command = task.get("command") or task.get("user_message") or task.get("query") or "Analyze dataset"
        session_id = task.get("session_id", "default_session")
        data = task.get("data", self.data)

        try:
            response_text, evidence, meta = self.chat(
                command=command,
                session_id=session_id,
                data=data,
            )
            return self._finish(
                result={"response": response_text, **meta},
                evidence=evidence,
                confidence=0.95,
                metadata={"session_id": session_id, "turn_count": len(self.get_or_create_session(session_id).turns)},
            )
        except Exception as exc:
            return self._error(f"Conversational analyst encountered an error: {str(exc)}", category=ErrorCategory.COMPUTATION)
