"""
Master Autonomous Data Analyst Agent.

Orchestrates:
Dataset + UserIntent
      ↓
AnalysisDiscoveryAgent
      ↓
AutonomousAnalysisEngine (Execution)
      ↓
InsightRanker (Deduplication & Ranking)
      ↓
AutonomousAnalysisResult
      ↓
Standardized AgentResult with Evidence
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from agent.analysis_discovery_agent import AnalysisDiscoveryAgent
from agent.autonomous_analysis_engine import AutonomousAnalysisEngine
from agent.autonomous_analysis_schemas import (
    AnalysisCandidate,
    AnalysisDepth,
    AutonomousAnalysisRequest,
    AutonomousAnalysisResult,
    Insight,
)
from agent.base import BaseAgent
from agent.dataset_knowledge import DatasetKnowledge
from agent.insight_ranker import InsightRanker
from agent.intent import UserIntent
from agent.schemas import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)


class AutonomousAnalystAgent(BaseAgent):
    """
    Autonomous Senior Data Analyst Agent capable of end-to-end data exploration,
    hypothesis testing, pattern validation, and executive insight synthesis.
    """
    name = "Autonomous Analyst Agent"
    role = "senior_lead_data_analyst"
    description = "Autonomously discovers dataset patterns, calculates statistical evidence, ranks insights, and delivers executive answers."

    def __init__(self, data: Optional[Any] = None):
        super().__init__(data=data)
        self.discovery_agent = AnalysisDiscoveryAgent()
        self.analysis_engine = AutonomousAnalysisEngine()
        self.ranker = InsightRanker()

    def analyze(self, request: AutonomousAnalysisRequest) -> AutonomousAnalysisResult:
        """Execute full autonomous analysis pipeline on request payload."""
        start_t = time.perf_counter()

        # 1. Prepare DataFrame
        raw_data = request.dataset
        if isinstance(raw_data, dict):
            df = pd.DataFrame(raw_data)
        elif isinstance(raw_data, list):
            df = pd.DataFrame(raw_data)
        elif isinstance(raw_data, pd.DataFrame):
            df = raw_data.copy()
        else:
            return AutonomousAnalysisResult(
                status="failed",
                summary="Unsupported dataset format provided.",
                warnings=[f"Cannot convert {type(raw_data)} to DataFrame."],
                confidence=0.0,
            )

        if df.empty:
            return AutonomousAnalysisResult(
                status="failed",
                summary="Dataset is empty.",
                warnings=["The provided dataset contains 0 records."],
                confidence=0.0,
            )

        # 2. Discover and prioritize candidate analyses
        candidates = self.discovery_agent.discover_analyses(
            df=df,
            user_intent=request.user_intent,
            depth=request.analysis_depth,
            max_steps=request.max_analysis_steps,
        )

        all_insights: List[Insight] = []
        all_evidence: List[Evidence] = []
        key_metrics: Dict[str, Any] = {}
        performed: List[str] = []
        skipped: List[str] = []
        warnings: List[str] = []

        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
        date_cols = []
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_cols.append(c)
            elif "date" in c.lower() or "time" in c.lower() or "year" in c.lower() or "month" in c.lower():
                try:
                    pd.to_datetime(df[c].dropna().head(10))
                    date_cols.append(c)
                except Exception:
                    pass

        # 3. Execute prioritized analyses with failure isolation
        for cand in candidates:
            a_type = cand.analysis_type
            try:
                if a_type == "data_quality":
                    m, ins = self.analysis_engine.analyze_data_quality(df)
                    key_metrics["data_quality"] = m
                    all_insights.extend(ins)
                    performed.append(a_type)

                elif a_type == "descriptive_statistics":
                    m, ins = self.analysis_engine.analyze_descriptive_stats(df, num_cols)
                    key_metrics["descriptive_statistics"] = m
                    all_insights.extend(ins)
                    performed.append(a_type)

                elif a_type == "trend_analysis":
                    if date_cols and num_cols:
                        d_col = date_cols[0]
                        m_col = cand.required_inputs[1] if len(cand.required_inputs) > 1 and cand.required_inputs[1] in num_cols else num_cols[0]
                        m, ins = self.analysis_engine.analyze_trends(df, d_col, m_col)
                        if ins:
                            key_metrics["trends"] = m
                            all_insights.extend(ins)
                            performed.append(a_type)
                        else:
                            skipped.append(f"{a_type} (insufficient temporal depth)")
                    else:
                        skipped.append(f"{a_type} (no date columns)")

                elif a_type in ("segmentation", "business_driver_investigation"):
                    if cat_cols and num_cols:
                        dim_col = cand.required_inputs[0] if cand.required_inputs and cand.required_inputs[0] in cat_cols else cat_cols[0]
                        m_col = cand.required_inputs[1] if len(cand.required_inputs) > 1 and cand.required_inputs[1] in num_cols else num_cols[0]
                        m, ins = self.analysis_engine.analyze_segmentation(df, dim_col, m_col)
                        if ins:
                            key_metrics["segmentation"] = m
                            all_insights.extend(ins)
                            performed.append(a_type)
                        else:
                            skipped.append(f"{a_type} (empty segments)")
                    else:
                        skipped.append(f"{a_type} (no categorical dimensions)")

                elif a_type == "correlation_analysis":
                    if len(num_cols) >= 2:
                        m, ins = self.analysis_engine.analyze_correlations(df, num_cols)
                        key_metrics["correlations"] = m
                        all_insights.extend(ins)
                        performed.append(a_type)
                    else:
                        skipped.append(f"{a_type} (requires >= 2 numeric columns)")

                elif a_type == "anomaly_detection":
                    if num_cols:
                        m_col = cand.required_inputs[0] if cand.required_inputs and cand.required_inputs[0] in num_cols else num_cols[0]
                        m, ins = self.analysis_engine.analyze_anomalies(df, m_col)
                        if ins:
                            key_metrics["anomalies"] = m
                            all_insights.extend(ins)
                            performed.append(a_type)
                        else:
                            performed.append(f"{a_type} (no anomalies)")
                    else:
                        skipped.append(f"{a_type} (no numeric columns)")

                elif a_type == "concentration_analysis":
                    if (cat_cols or len(df.columns) > 1) and num_cols:
                        dim_col = cat_cols[0] if cat_cols else df.columns[0]
                        m_col = num_cols[0]
                        m, ins = self.analysis_engine.analyze_concentration(df, dim_col, m_col)
                        if ins:
                            key_metrics["concentration"] = m
                            all_insights.extend(ins)
                            performed.append(a_type)
                        else:
                            skipped.append(f"{a_type} (low cardinality)")
                    else:
                        skipped.append(f"{a_type} (no dimensions)")

            except Exception as exc:
                warnings.append(f"Analysis '{a_type}' encountered error: {str(exc)}")
                skipped.append(a_type)

        # 4. Rank and deduplicate insights
        top_k = 4 if request.analysis_depth == AnalysisDepth.QUICK else (8 if request.analysis_depth == AnalysisDepth.STANDARD else 15)
        ranked_insights = self.ranker.rank(all_insights, user_intent=request.user_intent, top_k=top_k)

        for ins in ranked_insights:
            all_evidence.append(ins.evidence)

        recommendations = self.ranker.extract_recommendations(ranked_insights)
        limitations = self.ranker.extract_limitations(ranked_insights)

        exec_time = time.perf_counter() - start_t

        # 5. Formulate concise executive summary
        summary_sentences = [
            f"Autonomous analysis evaluated {len(performed)} analytical dimensions across {len(df):,} records and {len(df.columns)} features."
        ]
        if ranked_insights:
            summary_sentences.append(f"Key finding: {ranked_insights[0].summary}")
        if len(ranked_insights) > 1:
            summary_sentences.append(f"Secondary finding: {ranked_insights[1].summary}")

        overall_status = "success" if performed and not warnings else ("partial" if performed and warnings else "failed")

        return AutonomousAnalysisResult(
            status=overall_status,
            summary=" ".join(summary_sentences),
            insights=ranked_insights,
            key_metrics=key_metrics,
            analyses_performed=performed,
            analyses_skipped=skipped,
            warnings=warnings,
            limitations=limitations,
            recommendations=recommendations,
            evidence=all_evidence,
            confidence=0.94 if performed else 0.50,
            execution_time=exec_time,
        )

    def run(self, task: Dict[str, Any]) -> AgentResult:
        """Standardized BaseAgent execution interface."""
        self._start()
        data = task.get("data", self.data)
        if data is None:
            return self._error("No data provided for autonomous analysis.", category=ErrorCategory.INPUT_VALIDATION)

        intent_obj = task.get("user_intent")
        if isinstance(intent_obj, dict):
            intent_obj = UserIntent(**intent_obj)

        depth_str = task.get("analysis_depth", "standard")
        try:
            depth = AnalysisDepth(depth_str)
        except Exception:
            depth = AnalysisDepth.STANDARD

        req = AutonomousAnalysisRequest(
            dataset=data,
            user_intent=intent_obj,
            analysis_depth=depth,
            business_objective=task.get("business_objective"),
            max_analysis_steps=int(task.get("max_analysis_steps", 10)),
            confidence_threshold=float(task.get("confidence_threshold", 0.60)),
        )

        result = self.analyze(req)

        if result.status == "failed":
            return self._error(
                message=result.summary,
                category=ErrorCategory.COMPUTATION,
                details={"warnings": result.warnings},
            )

        if result.status == "partial":
            return self._partial(
                result=result.to_dict(),
                message=result.summary,
                warnings=result.warnings,
                evidence=result.evidence,
                confidence=result.confidence,
            )

        return self._finish(
            result=result.to_dict(),
            evidence=result.evidence,
            confidence=result.confidence,
            metadata={
                "analyses_performed": result.analyses_performed,
                "insight_count": len(result.insights),
            },
        )
