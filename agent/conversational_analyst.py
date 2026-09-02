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
from agent.autonomous_forecaster_agent import AutonomousForecasterAgent
from agent.forecasting_schemas import ForecastRequest, WhatIfRequest
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
        self.forecaster_agent = AutonomousForecasterAgent()
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
        dataset: Optional[Any] = None,
        df: Optional[Any] = None,
    ) -> Tuple[str, List[Evidence], Dict[str, Any]]:
        """
        Process a single natural language conversational turn.

        Returns:
            (response_text, evidence_list, turn_metadata)
        """
        session = self.get_or_create_session(session_id)

        # 1. Update Dataset Context if provided
        active_input_data = data if data is not None else (dataset if dataset is not None else df)
        if active_input_data is not None:
            if isinstance(active_input_data, pd.DataFrame):
                df = active_input_data
            elif isinstance(active_input_data, (dict, list)):
                df = pd.DataFrame(active_input_data)
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
            if re.search(r"\b(executive|business|strategic)\b", command, re.I):
                r_type = ReportType.EXECUTIVE_REPORT
            elif re.search(r"\b(technical|technically|data science|statistical)\b", command, re.I):
                r_type = ReportType.TECHNICAL_REPORT
            elif re.search(r"\b(quick|short|\bbrief\b)\b", command, re.I):
                r_type = ReportType.QUICK_SUMMARY

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

        target_metric = session.dataset_context.primary_metric if session.dataset_context else None
        target_dim = session.dataset_context.primary_dimension if session.dataset_context else None

        explicit_target = None
        if isinstance(entities, dict):
            explicit_target = entities.get("metric") or (entities.get("metrics")[0] if entities.get("metrics") else None)
        chosen_target = explicit_target or target_metric

        # 5a. Handle Forecasting & What-If Queries
        is_what_if = bool(re.search(r"\b(what if|what happens|scenario|best (and|&) worst|increases? by|decreases? by|drops? by|grows? by)\b", command, re.I))
        if intent == ConversationalIntent.FORECAST or is_what_if:
            fc_agent_res = self.forecaster_agent.run({"data": df, "command": command, "target": chosen_target})
            fc_data = fc_agent_res.data
            
            # Format Markdown Response
            if is_what_if:
                if "scenarios" in fc_data:
                    scen_lines = [f"🔮 **What-If Multi-Scenario Analysis for '{target_metric}':**\n", f"Baseline Value: **{fc_data.get('baseline_value', 0):,.2f}**\n", "| Scenario | Projected Value | Absolute Delta | Percentage Delta |", "| :--- | :---: | :---: | :---: |"]
                    for sc in fc_data.get("ranked_scenarios", []):
                        scen_lines.append(f"| **{sc.get('scenario_name')}** | {sc.get('scenario_value'):,.2f} | {sc.get('absolute_difference'):+,.2f} | **{sc.get('percentage_difference'):+.1f}%** |")
                    scen_lines.append("\n> [!NOTE]\n> *Projections represent associational model co-movements and do not constitute proven causal interventions.*")
                    final_response = "\n".join(scen_lines)
                else:
                    final_response = (
                        f"🔮 **What-If Simulation Result:**\n"
                        f"- Target Metric: **{fc_data.get('target_metric')}**\n"
                        f"- Baseline Value: **{fc_data.get('baseline_value', 0):,.2f}**\n"
                        f"- Simulated Value: **{fc_data.get('scenario_value', 0):,.2f}**\n"
                        f"- Estimated Impact: **{fc_data.get('absolute_difference', 0):+,.2f} ({fc_data.get('percentage_difference', 0):+.1f}%)**\n\n"
                        f"> [!NOTE]\n"
                        f"> *Simulation reflects predictive associations and does not guarantee causal intervention outcome.*"
                    )
            else:
                if fc_data.get("status") == "NOT_SUPPORTED":
                    final_response = f"⚠️ **Forecasting Unavailable:** {fc_data.get('warnings', ['Dataset is not suitable for time-series forecasting.'])[0]}"
                else:
                    preds = fc_data.get("predictions", [])
                    fc_lines = [
                        f"📈 **Time-Series Forecast for '{fc_data.get('target')}' ({fc_data.get('forecast_horizon')} Periods Ahead):**\n",
                        f"- **Selected Model:** `{fc_data.get('model_name')}` (Validation MAE: {fc_data.get('validation_metrics', {}).get('MAE', 0):.2f})",
                        f"- **Prediction Interval:** {int(fc_data.get('confidence_level', 0.8)*100)}% probabilistic bounds\n",
                        "| Timestamp | Forecast | Lower Bound | Upper Bound |",
                        "| :--- | :---: | :---: | :---: |",
                    ]
                    for p in preds:
                        fc_lines.append(f"| `{p.get('timestamp')}` | **{p.get('prediction'):,.2f}** | {p.get('lower_bound'):,.2f} | {p.get('upper_bound'):,.2f} |")
                    fc_lines.append("\n> [!TIP]\n> *Forecasts are probabilistic projections with widening intervals over extended horizons.*")
                    final_response = "\n".join(fc_lines)

            turn = ConversationTurn(
                session_id=session_id,
                user_message=command,
                resolved_intent=intent,
                referenced_entities=entities,
                evidence=fc_agent_res.evidence,
                assistant_response=final_response,
                result=fc_data,
            )
            session.turns.append(turn)
            return final_response, fc_agent_res.evidence, {
                "result": fc_data,
                "intent": intent.value,
                "resolved_command": resolved_cmd,
                "session_id": session_id,
                "turn_id": turn.turn_id,
            }

        # 5b. Handle Statistical Relationship & Correlation Queries
        is_rel_query = intent in (ConversationalIntent.RELATIONSHIPS, ConversationalIntent.CORRELATION) or any(w in command.lower() for w in ("correlation", "correlations", "relationship", "relationships", "pearson", "spearman", "effect size", "fdr", "outlier sensitivity", "subgroup"))
        if is_rel_query:
            from agent.statistical_analysis_agent import StatisticalAnalysisAgent
            stat_agent_res = StatisticalAnalysisAgent().run({"data": df})
            if stat_agent_res.is_success:
                stat_data = stat_agent_res.data
                top_rels = stat_data.get("top_relationships", []) or stat_data.get("relationships", [])
                lines = [f"📊 **Statistical Relationship Analysis for: \"{command}\"**\n"]
                if top_rels:
                    lines.append("| Relationship | Pearson r | Spearman ρ | Raw p-value | FDR-adjusted p-value | Effect Strength | Valid N | Outlier Sensitivity |")
                    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
                    for r in top_rels[:8]:
                        fx = r.get("feature_x") or r.get("variable_x")
                        fy = r.get("feature_y") or r.get("variable_y")
                        p_info = r.get("pearson", {}) or {}
                        s_info = r.get("spearman", {}) or {}
                        p_r = p_info.get("r", r.get("statistic", 0.0))
                        p_val = p_info.get("p_value", r.get("p_value", 0.0))
                        rho_val = s_info.get("rho", 0.0)
                        adj_p = r.get("adjusted_p_value", p_val)
                        strength = r.get("strength", "moderate").replace("_", " ").title()
                        outlier_sens = "Yes ⚠️" if r.get("outlier_sensitivity") else "No"
                        v_n = r.get("valid_rows", len(df))
                        lines.append(
                            f"| **{fx}** ↔ **{fy}** | {p_r:.3f} | {rho_val:.3f} | {p_val:.4g} | {adj_p:.4g} | {strength} | {v_n} | {outlier_sens} |"
                        )

                    lines.append("\n**Key Statistical Observations**:")
                    for i, r in enumerate(top_rels[:4], 1):
                        fx = r.get("feature_x")
                        fy = r.get("feature_y")
                        p_r = r.get("pearson", {}).get("r", r.get("statistic", 0.0))
                        rho_val = r.get("spearman", {}).get("rho", 0.0)
                        p_val = r.get("p_value", 0.0)
                        adj_p = r.get("adjusted_p_value", p_val)
                        strength = r.get("strength", "moderate")
                        lines.append(
                            f"{i}. **{fx}** ↔ **{fy}**: {strength.replace('_', ' ').title()} association (Pearson r = {p_r:.3f}, Spearman ρ = {rho_val:.3f}, raw p = {p_val:.4g}, FDR-adjusted p = {adj_p:.4g}, valid N = {r.get('valid_rows')})."
                        )

                lines.append("\n> [!NOTE]\n> *All reported p-values are adjusted using Benjamini-Hochberg FDR control. Correlations describe mathematical associations without establishing causal mechanisms.*")
                final_response = "\n".join(lines)

                turn = ConversationTurn(
                    session_id=session_id,
                    user_message=command,
                    resolved_intent=intent,
                    referenced_entities=entities,
                    evidence=stat_agent_res.evidence,
                    assistant_response=final_response,
                    result={
                        "relationships": stat_data.get("relationships", []),
                        "top_relationships": top_rels,
                        "correlation_matrix": stat_data.get("correlation_matrix", {}),
                        "subgroup_analysis": stat_data.get("subgroup_analysis", {}),
                        "status": "completed",
                    },
                )
                session.turns.append(turn)
                return final_response, stat_agent_res.evidence, {
                    "result": stat_data,
                    "intent": intent.value,
                    "resolved_command": resolved_cmd,
                    "session_id": session_id,
                    "turn_id": turn.turn_id,
                }

        # 5c. Handle Filtering & Filtered Aggregations
        # 5c. Handle Filtering, Grouping & Analytical Plans
        from agent.filter_engine import FilterEngine
        if intent == ConversationalIntent.FILTER or FilterEngine.has_filter_intent(command):
        _conv_plan = FilterEngine.parse_query_plan(command, dataframe=df)
        if intent == ConversationalIntent.FILTER or FilterEngine.has_filter_intent(command) or _conv_plan.filter is not None or bool(_conv_plan.group_by and (len(_conv_plan.group_by) > 1 or _conv_plan.ranking or _conv_plan.secondary_analysis)):
            filter_res = FilterEngine.execute(df, command)
            final_response = filter_res.markdown_response

            from agent.schemas import ClaimType, Evidence
            evidence_objs = [
                Evidence(
                    source="FilterEngine",
                    method="vectorized_filtering",
                    claim_type=ClaimType.FACT,
                    confidence=1.0,
                    raw_value={
                        "filter": filter_res.filter_description,
                        "matching_rows": filter_res.matching_rows,
                        "total_rows": filter_res.total_rows,
                        "aggregations": filter_res.aggregations,
                    },
                )
            ]

            turn = ConversationTurn(
                session_id=session_id,
                user_message=command,
                resolved_intent=intent,
                referenced_entities=entities,
                evidence=evidence_objs,
                assistant_response=final_response,
                result={
                    "filter": filter_res.filter_description,
                    "matching_rows": filter_res.matching_rows,
                    "total_rows": filter_res.total_rows,
                    "aggregations": filter_res.aggregations,
                    "filtered_data": filter_res.filtered_df.head(10).to_dict(orient="records"),
                    "status": "completed",
                },
            )
            session.turns.append(turn)
            return final_response, evidence_objs, {
                "result": {
                    "filter": filter_res.filter_description,
                    "matching_rows": filter_res.matching_rows,
                    "total_rows": filter_res.total_rows,
                    "aggregations": filter_res.aggregations,
                    "filtered_data": filter_res.filtered_df.head(10).to_dict(orient="records"),
                },
                "intent": intent.value if hasattr(intent, "value") else str(intent),
                "resolved_command": resolved_cmd,
                "session_id": session_id,
                "turn_id": turn.turn_id,
            }

        # Create structured intent for autonomous analysis
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
