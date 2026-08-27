"""
Universal Insight Synthesis & Decision Engine.

Single source of truth for cross-agent analytical insight synthesis, multi-agent
agreement/disagreement reasoning, contradiction detection, duplicate suppression,
evidence-grounded narrative generation, and causality protection.

Synthesizes results across:
- EDA & Data Quality
- Statistical Relationships & Hypothesis Testing
- Anomaly Detection
- Clustering & Customer Segmentation
- Time Series Forecasting
- Supervised Prediction (Regression & Classification)
- Data Transformations
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from agent.agent_result import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)
from agent.canonical_data_layer import CanonicalDataLayer, SemanticProfile


class InsightCategory(str, Enum):
    TREND = "trend"
    DISTRIBUTION = "distribution"
    RELATIONSHIP = "relationship"
    ANOMALY = "anomaly"
    SEGMENT = "segment"
    FORECAST = "forecast"
    PREDICTIVE_PERFORMANCE = "predictive_performance"
    DATA_QUALITY = "data_quality"
    LIMITATION = "limitation"
    CROSS_ANALYSIS = "cross_analysis"


class SynthesizedInsight(BaseModel):
    """Evidence-backed analytical insight."""
    insight_id: str = Field(default_factory=lambda: f"ins_{uuid.uuid4().hex[:8]}")
    category: str = InsightCategory.DISTRIBUTION.value
    title: str
    statement: str
    evidence_refs: List[Evidence] = Field(default_factory=list)
    supporting_metrics: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    importance: float = Field(default=0.50, ge=0.0, le=1.0)
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "category": self.category,
            "title": self.title,
            "statement": self.statement,
            "evidence_refs": [e.to_dict() if isinstance(e, Evidence) else e for e in self.evidence_refs],
            "supporting_metrics": self.supporting_metrics,
            "confidence": round(float(self.confidence), 4),
            "importance": round(float(self.importance), 4),
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "provenance": self.provenance,
        }


class Contradiction(BaseModel):
    """Structured representation of conflicting analytical results."""
    contradiction_id: str = Field(default_factory=lambda: f"contra_{uuid.uuid4().hex[:8]}")
    involved_insights: List[str] = Field(default_factory=list)
    conflicting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    explanation: str
    confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    resolution: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "involved_insights": self.involved_insights,
            "conflicting_evidence": self.conflicting_evidence,
            "explanation": self.explanation,
            "confidence": round(float(self.confidence), 4),
            "resolution": self.resolution,
        }


class SynthesisReport(BaseModel):
    """Complete analytical synthesis report integrating multi-agent outputs."""
    executive_summary: str
    key_insights: List[SynthesizedInsight] = Field(default_factory=list)
    important_findings: List[str] = Field(default_factory=list)
    data_quality_findings: List[SynthesizedInsight] = Field(default_factory=list)
    model_findings: List[SynthesizedInsight] = Field(default_factory=list)
    forecast_findings: List[SynthesizedInsight] = Field(default_factory=list)
    anomalies: List[SynthesizedInsight] = Field(default_factory=list)
    segments: List[SynthesizedInsight] = Field(default_factory=list)
    relationships: List[SynthesizedInsight] = Field(default_factory=list)
    cross_analysis_findings: List[SynthesizedInsight] = Field(default_factory=list)
    contradictions: List[Contradiction] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    recommended_next_questions: List[str] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    evidence: List[Evidence] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executive_summary": self.executive_summary,
            "key_insights": [i.to_dict() for i in self.key_insights],
            "important_findings": self.important_findings,
            "data_quality_findings": [i.to_dict() for i in self.data_quality_findings],
            "model_findings": [i.to_dict() for i in self.model_findings],
            "forecast_findings": [i.to_dict() for i in self.forecast_findings],
            "anomalies": [i.to_dict() for i in self.anomalies],
            "segments": [i.to_dict() for i in self.segments],
            "relationships": [i.to_dict() for i in self.relationships],
            "cross_analysis_findings": [i.to_dict() for i in self.cross_analysis_findings],
            "contradictions": [c.to_dict() for c in self.contradictions],
            "limitations": self.limitations,
            "recommended_next_questions": self.recommended_next_questions,
            "overall_confidence": round(float(self.overall_confidence), 4),
            "evidence": [e.to_dict() if isinstance(e, Evidence) else e for e in self.evidence],
            "metadata": self.metadata,
        }


class InsightSynthesisEngine:
    """
    Authoritative cross-agent insight synthesis engine.
    Converts multi-agent execution outputs into a structured, evidence-grounded narrative.
    """

    # Forbidden causal tokens and their safe observational substitutions
    CAUSAL_REPLACEMENTS = [
        (r"\bcauses\b", "is associated with"),
        (r"\bcaused\b", "coincided with"),
        (r"\bdrives\b", "is strongly associated with"),
        (r"\bdriven by\b", "associated with"),
        (r"\bleads to\b", "is correlated with"),
        (r"\bresults in\b", "is characterized by"),
        (r"\bimpacts\b", "shows a statistical relationship with"),
        (r"\bimpacted by\b", "associated with"),
        (r"\bresponsible for\b", "exhibiting statistical dependency with"),
        (r"\bbecause of\b", "in connection with"),
    ]

    def __init__(self):
        pass

    # --------------------------------------------------------------------------
    # Main Synthesis Entrypoint
    # --------------------------------------------------------------------------

    def synthesize(
        self,
        orchestration_result: Union[AgentResult, Dict[str, Any]],
        dataframe: Optional[pd.DataFrame] = None,
        profile: Optional[SemanticProfile] = None,
        command: Optional[str] = None,
    ) -> SynthesisReport:
        """
        Synthesize multi-agent analytical outputs into a unified, evidence-backed narrative.
        """
        task_outputs = self._extract_task_outputs(orchestration_result)
        source_evidence = self._extract_evidence(orchestration_result)
        source_confidence = self._extract_confidence(orchestration_result)
        cmd_text = command or self._extract_command(orchestration_result)

        if dataframe is not None and profile is None:
            profile = CanonicalDataLayer.ingest(dataframe).profile

        raw_insights: List[SynthesizedInsight] = []

        # 1. Synthesize EDA & Data Quality
        eda_insights = self._synthesize_eda_and_data_quality(task_outputs.get("eda"), profile, dataframe)
        raw_insights.extend(eda_insights)

        # 2. Synthesize Statistical Relationships & Hypothesis Tests
        stat_insights = self._synthesize_relationships(task_outputs.get("statistical_analysis"), task_outputs.get("hypothesis_testing"))
        raw_insights.extend(stat_insights)

        # 3. Synthesize Anomaly Detection
        anom_insights = self._synthesize_anomalies(task_outputs.get("anomaly_detection"))
        raw_insights.extend(anom_insights)

        # 4. Synthesize Clustering & Segmentation
        clust_insights = self._synthesize_clustering(task_outputs.get("clustering"))
        raw_insights.extend(clust_insights)

        # 5. Synthesize Time Series Forecasting
        fc_insights = self._synthesize_forecasting(task_outputs.get("forecasting"))
        raw_insights.extend(fc_insights)

        # 6. Synthesize Supervised Prediction
        pred_insights = self._synthesize_prediction(task_outputs.get("prediction"))
        raw_insights.extend(pred_insights)

        # 7. Synthesize Data Quality Gate if available
        dq_insights = self._synthesize_data_quality_gate(task_outputs.get("data_quality_gate"))
        raw_insights.extend(dq_insights)

        # 8. Duplicate Insight Suppression & Merging
        unique_insights = self._suppress_duplicates(raw_insights)

        # 9. Cross-Agent Reasoning & Agreement Insights
        cross_insights = self._synthesize_cross_agent(unique_insights, task_outputs)
        unique_insights.extend(cross_insights)

        # 10. Contradiction Detection
        contradictions = self._detect_contradictions(unique_insights, task_outputs)

        # 11. Prioritize and Rank Insights by Importance
        ranked_insights = self._calculate_importance_and_rank(unique_insights)

        # 12. Partition Insights by Category
        dq_findings = [i for i in ranked_insights if i.category == InsightCategory.DATA_QUALITY.value]
        mod_findings = [i for i in ranked_insights if i.category == InsightCategory.PREDICTIVE_PERFORMANCE.value]
        fc_findings = [i for i in ranked_insights if i.category == InsightCategory.FORECAST.value]
        anom_findings = [i for i in ranked_insights if i.category == InsightCategory.ANOMALY.value]
        seg_findings = [i for i in ranked_insights if i.category == InsightCategory.SEGMENT.value]
        rel_findings = [i for i in ranked_insights if i.category == InsightCategory.RELATIONSHIP.value]
        cross_findings = [i for i in ranked_insights if i.category == InsightCategory.CROSS_ANALYSIS.value]

        key_insights = ranked_insights[:6]
        important_findings = [i.statement for i in key_insights]

        # 13. Limitations & Disclaimers
        limitations = [
            "All relationships and patterns reported are observational and do not establish causal mechanisms.",
            "Model forecasts and predictions are statistical projections subject to uncertainty intervals.",
        ]
        if any(i.supporting_metrics.get("is_sampled", False) for i in ranked_insights):
            limitations.append("Insights were derived from representative bounded sampling of the massive dataset.")

        # 14. Recommended Follow-up Questions
        next_questions = self._generate_next_questions(ranked_insights, contradictions, dq_findings)

        # 15. Executive Summary
        exec_summary = self._generate_executive_summary(
            ranked_insights=ranked_insights,
            contradictions=contradictions,
            command=cmd_text,
            task_outputs=task_outputs,
        )

        # 16. Calculate Principled Overall Confidence
        if ranked_insights:
            conf_sum = sum(i.confidence * i.importance for i in ranked_insights)
            weight_sum = sum(i.importance for i in ranked_insights)
            base_conf = conf_sum / max(1e-6, weight_sum)
            # Penalty for unresolved contradictions
            contra_penalty = 0.85 if contradictions else 1.0
            overall_conf = round(max(0.0, min(1.0, base_conf * contra_penalty)), 4)
        else:
            overall_conf = 0.50

        # Combine all Evidence records
        all_evidence = list(source_evidence)
        for i in ranked_insights:
            for e in i.evidence_refs:
                if e not in all_evidence:
                    all_evidence.append(e)

        return SynthesisReport(
            executive_summary=exec_summary,
            key_insights=key_insights,
            important_findings=important_findings,
            data_quality_findings=dq_findings,
            model_findings=mod_findings,
            forecast_findings=fc_findings,
            anomalies=anom_findings,
            segments=seg_findings,
            relationships=rel_findings,
            cross_analysis_findings=cross_findings,
            contradictions=contradictions,
            limitations=limitations,
            recommended_next_questions=next_questions,
            overall_confidence=overall_conf,
            evidence=all_evidence,
            metadata={
                "synthesized_at": datetime.now().isoformat(),
                "total_insights_generated": len(ranked_insights),
                "contradictions_detected": len(contradictions),
                "analyzed_tasks": list(task_outputs.keys()),
            },
        )

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        """Safely coerce any value to a finite float."""
        try:
            if val is None:
                return default
            if isinstance(val, (int, float)):
                if math.isnan(val) or math.isinf(val):
                    return default
                return float(val)
            if isinstance(val, str):
                f = float(val)
                return default if (math.isnan(f) or math.isinf(f)) else f
        except (ValueError, TypeError):
            pass
        return default

    # --------------------------------------------------------------------------
    # Specialized Domain Synthesizers
    # --------------------------------------------------------------------------

    def _synthesize_eda_and_data_quality(
        self,
        eda_data: Optional[Dict[str, Any]],
        profile: Optional[SemanticProfile],
        df: Optional[pd.DataFrame],
    ) -> List[SynthesizedInsight]:
        """Synthesize observational data quality and distribution insights."""
        insights: List[SynthesizedInsight] = []

        # Handle direct DataFrame / SemanticProfile if eda_data is absent
        if (not eda_data or not isinstance(eda_data, dict)) and df is not None:
            n_rows = len(df)
            n_cols = len(df.columns)
            if n_rows > 0:
                null_counts = {c: int(df[c].isnull().sum()) for c in df.columns}
                total_nulls = sum(null_counts.values())
                qs = round(1.0 - (total_nulls / max(1, n_rows * n_cols)), 4)
                qs_pct = round(qs * 100, 1)

                ev = Evidence(
                    operation="data_quality.profile",
                    result={"rows": n_rows, "columns": n_cols, "quality_score": qs},
                    confidence=0.95,
                    claim_type=ClaimType.OBSERVATION,
                )
                insights.append(
                    SynthesizedInsight(
                        category=InsightCategory.DATA_QUALITY.value,
                        title="Dataset Structure & Data Quality Health",
                        statement=f"Dataset contains {n_rows:,} observations across {n_cols} attributes with an aggregate data quality index of {qs_pct}%.",
                        evidence_refs=[ev],
                        supporting_metrics={"rows": n_rows, "columns": n_cols, "quality_score": qs},
                        confidence=0.95,
                        importance=0.75,
                        provenance={"agent": "CanonicalDataLayer"},
                    )
                )

                # Null attributes
                null_cols = [c for c, cnt in null_counts.items() if cnt > 0]
                if null_cols:
                    ev_null = Evidence(
                        operation="data_quality.missingness",
                        columns=null_cols,
                        result={"null_counts": null_counts},
                        confidence=0.95,
                        claim_type=ClaimType.OBSERVATION,
                    )
                    insights.append(
                        SynthesizedInsight(
                            category=InsightCategory.DATA_QUALITY.value,
                            title="Missing Value Distribution",
                            statement=f"Identified missing observations in attributes: {', '.join(null_cols[:3])}. Non-destructive pairwise computation preserves observation rows.",
                            evidence_refs=[ev_null],
                            supporting_metrics={"null_columns": null_cols},
                            confidence=0.92,
                            importance=0.70,
                            provenance={"agent": "CanonicalDataLayer"},
                        )
                    )
            return insights

        if not eda_data or not isinstance(eda_data, dict):
            return insights

        summary = eda_data.get("summary", {})
        statistics = eda_data.get("statistics", {})
        dq = eda_data.get("data_quality", {})

        n_rows = summary.get("row_count") or summary.get("original_rows") or (len(df) if df is not None else 0)
        n_cols = summary.get("column_count") or summary.get("original_columns") or (len(df.columns) if df is not None else 0)

        # 1. Dataset Scale & Completeness
        if n_rows > 0:
            overall_qs = dq.get("overall_score") or dq.get("quality_score", 1.0)
            qs_pct = round(self._safe_float(overall_qs, 1.0) * 100, 1)
            ev = Evidence(
                operation="eda.data_quality.assessment",
                calculation=f"Quality score: {overall_qs}",
                result={"row_count": n_rows, "column_count": n_cols, "quality_score": overall_qs},
                confidence=0.95,
                claim_type=ClaimType.OBSERVATION,
            )
            insights.append(
                SynthesizedInsight(
                    category=InsightCategory.DATA_QUALITY.value,
                    title="Dataset Structure & Data Quality Health",
                    statement=f"Dataset contains {n_rows:,} observations across {n_cols} attributes with an aggregate data quality index of {qs_pct}%.",
                    evidence_refs=[ev],
                    supporting_metrics={"rows": n_rows, "columns": n_cols, "quality_score": overall_qs},
                    confidence=0.95,
                    importance=0.75,
                    provenance={"agent": "EDAAgent", "section": "summary"},
                )
            )

        # 2. Missingness Findings
        miss_analysis = eda_data.get("missing_analysis", {})
        high_miss_cols = miss_analysis.get("high_missing_columns", [])
        if high_miss_cols:
            col_names = [c["column"] if isinstance(c, dict) else str(c) for c in high_miss_cols[:3]]
            ev = Evidence(
                operation="eda.missing.analysis",
                columns=col_names,
                result={"high_missing_columns": high_miss_cols},
                confidence=0.95,
                claim_type=ClaimType.OBSERVATION,
            )
            insights.append(
                SynthesizedInsight(
                    category=InsightCategory.DATA_QUALITY.value,
                    title="Elevated Missingness in Selected Attributes",
                    statement=f"Significant missing values identified in attributes: {', '.join(col_names)}. Pairwise analysis preserves valid observations without destructive row deletion.",
                    evidence_refs=[ev],
                    supporting_metrics={"high_missing_columns": col_names},
                    confidence=0.92,
                    importance=0.70,
                    provenance={"agent": "EDAAgent", "section": "missing_analysis"},
                )
            )

        # 3. Numeric Distribution Highlights
        num_stats = statistics.get("numeric", {})
        for col_name, stats_dict in list(num_stats.items())[:3]:
            if isinstance(stats_dict, dict) and "mean" in stats_dict and "std" in stats_dict:
                v_mean = round(self._safe_float(stats_dict["mean"]), 2)
                v_median = round(self._safe_float(stats_dict.get("median", stats_dict["mean"])), 2)
                v_shape = stats_dict.get("distribution_shape", "symmetric")
                outlier_c = int(self._safe_float(stats_dict.get("outlier_count", 0)))

                stmt = f"Attribute '{col_name}' displays a {v_shape} distribution with median {v_median} and mean {v_mean}."
                if outlier_c > 0:
                    stmt += f" Detected {outlier_c} statistical outlier values beyond 1.5x IQR boundaries."

                ev = Evidence(
                    operation="eda.numeric.distribution",
                    columns=[col_name],
                    result=stats_dict,
                    confidence=0.90,
                    claim_type=ClaimType.OBSERVATION,
                )
                insights.append(
                    SynthesizedInsight(
                        category=InsightCategory.DISTRIBUTION.value,
                        title=f"Distribution Profile: {col_name}",
                        statement=stmt,
                        evidence_refs=[ev],
                        supporting_metrics=stats_dict,
                        confidence=0.90,
                        importance=0.60 if outlier_c == 0 else 0.72,
                        provenance={"agent": "EDAAgent", "column": col_name},
                    )
                )

        return insights

    def _synthesize_relationships(
        self,
        stats_data: Optional[Dict[str, Any]],
        hyp_data: Optional[Dict[str, Any]],
    ) -> List[SynthesizedInsight]:
        """Synthesize statistical dependencies, correlations, and hypothesis tests."""
        insights: List[SynthesizedInsight] = []

        if stats_data and isinstance(stats_data, dict):
            ranked_rels = stats_data.get("ranked_relationships") or stats_data.get("relationships", [])
            for rel in ranked_rels[:6]:
                if isinstance(rel, dict):
                    f1 = rel.get("feature_x") or rel.get("feature_1") or rel.get("feature") or "Variable 1"
                    f2 = rel.get("feature_y") or rel.get("feature_2") or rel.get("target") or "Variable 2"
                    stat_val = self._safe_float(rel.get("statistic", rel.get("correlation", rel.get("coefficient", rel.get("r", 0.0)))))
                    p_val = self._safe_float(rel.get("p_value", 0.0))
                    adj_p = self._safe_float(rel.get("adjusted_p_value", p_val))
                    effect = self._safe_float(rel.get("effect_size", abs(stat_val)), abs(stat_val))
                    method = rel.get("primary_method", rel.get("method", "correlation"))
                    strength = rel.get("strength")
                    if not strength:
                        eff_val = abs(effect if effect is not None else stat_val)
                        strength = "very strong" if eff_val >= 0.70 else ("strong" if eff_val >= 0.50 else ("moderate" if eff_val >= 0.30 else "weak"))
                    outlier_sens = rel.get("outlier_sensitivity", False)

                    p_info = rel.get("pearson", {})
                    s_info = rel.get("spearman", {})
                    r_str = f"Pearson r={p_info.get('r'):.3f}" if p_info.get("r") is not None else f"statistic={stat_val:.3f}"
                    rho_str = f", Spearman rho={s_info.get('rho'):.3f}" if s_info.get("rho") is not None else ""

                    direction = "positive" if stat_val > 0 else "negative" if stat_val < 0 else "neutral"

                    statement = (
                        f"Features '{f1}' and '{f2}' exhibit a {strength} {direction} statistical relationship "
                        f"({r_str}{rho_str}, raw p={p_val:.4g}, FDR-adjusted p={adj_p:.4g})."
                    )
                    if outlier_sens:
                        statement += " Relationship exhibits sensitivity to extreme outlier observations (divergence between Pearson and Spearman)."

                    statement = self._sanitize_causality(statement)

                    ev = Evidence(
                        operation=f"statistical_analysis.{rel.get('pair_type', 'bivariate')}",
                        columns=[str(f1), str(f2)],
                        result=rel,
                        confidence=0.90,
                        claim_type=ClaimType.CORRELATION,
                    )

                    importance_score = min(1.0, 0.40 + 0.50 * float(effect))
                    insights.append(
                        SynthesizedInsight(
                            category=InsightCategory.RELATIONSHIP.value,
                            title=f"Statistical Association: {f1} & {f2}",
                            statement=statement,
                            evidence_refs=[ev],
                            supporting_metrics=rel,
                            confidence=0.90,
                            importance=importance_score,
                            limitations=["Observational correlation does not imply a causal mechanism."],
                            provenance={"agent": "StatisticalAnalysisAgent", "pair": f"{f1}_{f2}"},
                        )
                    )

            # Subgroup findings
            subgroup_data = stats_data.get("subgroup_analysis", {})
            weak_findings = subgroup_data.get("weak_global_strong_subgroup_findings", [])
            for wf in weak_findings[:4]:
                if isinstance(wf, dict):
                    fx = wf.get("feature_x", "Feature X")
                    fy = wf.get("feature_y", "Feature Y")
                    dim = wf.get("subgroup_dimension", "Segment")
                    val = wf.get("subgroup_value", "Group")
                    gr = wf.get("global_r", 0.0)
                    sr = wf.get("subgroup_r", 0.0)
                    sp = wf.get("subgroup_p_value", 0.0)
                    sn = wf.get("subgroup_valid_rows", 0)

                    stmt = (
                        f"Subgroup heterogeneity: Association between '{fx}' and '{fy}' is weak overall (r={gr:.3f}), "
                        f"but becomes {wf.get('subgroup_strength', 'strong')} within {dim} = '{val}' "
                        f"(subgroup r={sr:.3f}, p={sp:.4g}, n={sn})."
                    )
                    stmt = self._sanitize_causality(stmt)

                    ev = Evidence(
                        operation="statistical_analysis.subgroup_heterogeneity",
                        columns=[str(fx), str(fy), str(dim)],
                        result=wf,
                        confidence=0.88,
                        claim_type=ClaimType.OBSERVATION,
                    )

                    insights.append(
                        SynthesizedInsight(
                            category=InsightCategory.RELATIONSHIP.value,
                            title=f"Subgroup Heterogeneity: {fx} & {fy} in {dim}='{val}'",
                            statement=stmt,
                            evidence_refs=[ev],
                            supporting_metrics=wf,
                            confidence=0.88,
                            importance=0.82,
                            limitations=["Subgroup findings reflect observational cohort differences."],
                            provenance={"agent": "StatisticalAnalysisAgent", "subgroup": f"{dim}_{val}"},
                        )
                    )

        if hyp_data and isinstance(hyp_data, dict):
            hypotheses = hyp_data.get("hypotheses") or hyp_data.get("findings", [])
            for h in hypotheses[:3]:
                if isinstance(h, dict):
                    f = h.get("feature", "Feature")
                    grp = h.get("group", "Group")
                    test_name = h.get("test_name", "Statistical test")
                    p_val = h.get("p_value", 1.0)
                    rej = h.get("reject_null", False)
                    effect_size = h.get("effect_size", 0.0)

                    status_phrase = "statistically significant differences" if rej else "no statistically significant difference"
                    stmt = f"Group comparison across '{grp}' for metric '{f}' via {test_name} indicates {status_phrase} (p={p_val:.4g}, effect size={effect_size:.3g})."
                    stmt = self._sanitize_causality(stmt)

                    ev = Evidence(
                        operation="hypothesis_testing.group_comparison",
                        columns=[str(f), str(grp)],
                        result=h,
                        confidence=0.88,
                        claim_type=ClaimType.OBSERVATION,
                    )

                    insights.append(
                        SynthesizedInsight(
                            category=InsightCategory.RELATIONSHIP.value,
                            title=f"Hypothesis Test: {f} by {grp}",
                            statement=stmt,
                            evidence_refs=[ev],
                            supporting_metrics=h,
                            confidence=0.88,
                            importance=0.75 if rej else 0.55,
                            provenance={"agent": "HypothesisTestingAgent"},
                        )
                    )

        return insights

    def _synthesize_anomalies(self, anomaly_data: Optional[Dict[str, Any]]) -> List[SynthesizedInsight]:
        """Synthesize statistical outlier and anomaly findings."""
        insights: List[SynthesizedInsight] = []
        if not anomaly_data or not isinstance(anomaly_data, dict):
            return insights

        anom_count = anomaly_data.get("anomaly_count", 0)
        anom_rate = anomaly_data.get("anomaly_rate", 0.0)
        anom_rate_pct = round(float(anom_rate) * 100, 2)
        detector = anomaly_data.get("detector_used") or anomaly_data.get("method", "Statistical Anomaly Detector")
        features = anomaly_data.get("features_analyzed", [])

        if anom_count > 0:
            stmt = f"{detector} identified {anom_count} statistically unusual observation(s) ({anom_rate_pct}% anomaly rate) deviating significantly from multidimensional distribution norms."
        else:
            stmt = f"{detector} found no extreme statistical anomalies within standard confidence thresholds."

        stmt = self._sanitize_causality(stmt)
        ev = Evidence(
            operation="anomaly_detection.isolation_forest",
            columns=[str(c) for c in features],
            result=anomaly_data,
            confidence=0.88,
            claim_type=ClaimType.OBSERVATION,
        )

        insights.append(
            SynthesizedInsight(
                category=InsightCategory.ANOMALY.value,
                title="Statistical Anomaly & Outlier Distribution",
                statement=stmt,
                evidence_refs=[ev],
                supporting_metrics={"anomaly_count": anom_count, "anomaly_rate": anom_rate},
                confidence=0.88,
                importance=0.80 if anom_count > 0 else 0.45,
                limitations=["Anomalies denote mathematical statistical extremity and require domain validation before operational intervention."],
                provenance={"agent": "AnomalyDetectionAgent"},
            )
        )

        return insights

    def _synthesize_clustering(self, clustering_data: Optional[Dict[str, Any]]) -> List[SynthesizedInsight]:
        """Synthesize unsupervised clustering and customer segmentation insights."""
        insights: List[SynthesizedInsight] = []
        if not clustering_data or not isinstance(clustering_data, dict):
            return insights

        k = clustering_data.get("cluster_count") or clustering_data.get("n_clusters", 0)
        algo = clustering_data.get("algorithm", "Clustering Engine")
        sil_score = clustering_data.get("silhouette_score", 0.0)
        sizes = clustering_data.get("cluster_sizes", {})

        if k >= 2:
            size_desc = ", ".join(f"Cluster {c}: {sz} records" for c, sz in list(sizes.items())[:4])
            stmt = f"Unsupervised {algo} partitioned dataset into {k} natural groups ({size_desc}) with a silhouette cohesion score of {sil_score:.3f}."
            stmt = self._sanitize_causality(stmt)

            ev = Evidence(
                operation="clustering.unsupervised_segmentation",
                result=clustering_data,
                confidence=0.85,
                claim_type=ClaimType.OBSERVATION,
            )

            insights.append(
                SynthesizedInsight(
                    category=InsightCategory.SEGMENT.value,
                    title=f"Unsupervised Segmentation: {k} Natural Groups",
                    statement=stmt,
                    evidence_refs=[ev],
                    supporting_metrics=clustering_data,
                    confidence=0.85,
                    importance=0.72,
                    limitations=["Cluster definitions are geometric descriptive groupings and do not represent predetermined behavioral archetypes."],
                    provenance={"agent": "ClusteringAgent"},
                )
            )

        return insights

    def _synthesize_forecasting(self, fc_data: Optional[Dict[str, Any]]) -> List[SynthesizedInsight]:
        """Synthesize autonomous time series forecast projections."""
        insights: List[SynthesizedInsight] = []
        if not fc_data or not isinstance(fc_data, dict):
            return insights

        target = fc_data.get("target_column") or "Target Series"
        horizon = fc_data.get("horizon", 6)
        model_name = fc_data.get("model_selected") or fc_data.get("model_name", "Autonomous Forecaster")
        trend = fc_data.get("trend_direction", "neutral")
        fc_values = fc_data.get("forecast") or fc_data.get("predictions", [])

        if fc_values:
            first_val = float(fc_values[0]) if isinstance(fc_values[0], (int, float)) else 0.0
            last_val = float(fc_values[-1]) if isinstance(fc_values[-1], (int, float)) else 0.0
            pct_change = round(((last_val - first_val) / max(1e-6, abs(first_val))) * 100, 1) if first_val != 0 else 0.0
            dir_str = "an upward" if last_val >= first_val else "a downward"

            stmt = f"The selected {model_name} projects {dir_str} trajectory for '{target}' over the next {horizon} periods (projected shift of {pct_change}% across horizon bounds)."
            stmt = self._sanitize_causality(stmt)

            ev = Evidence(
                operation="forecasting.autonomous_projection",
                columns=[str(target)],
                result=fc_data,
                confidence=0.86,
                claim_type=ClaimType.INFERENCE,
            )

            insights.append(
                SynthesizedInsight(
                    category=InsightCategory.FORECAST.value,
                    title=f"Time Series Forecast: {target} ({horizon} Periods)",
                    statement=stmt,
                    evidence_refs=[ev],
                    supporting_metrics={"horizon": horizon, "model": model_name, "projected_change_pct": pct_change},
                    confidence=0.86,
                    importance=0.82,
                    limitations=["Forecasts represent statistical projections under historical stationarity assumptions, not guaranteed future outcomes."],
                    provenance={"agent": "ForecastAgent", "target": target},
                )
            )

        return insights

    def _synthesize_prediction(self, pred_data: Optional[Dict[str, Any]]) -> List[SynthesizedInsight]:
        """Synthesize supervised predictive modeling and feature attribution insights."""
        insights: List[SynthesizedInsight] = []
        if not pred_data or not isinstance(pred_data, dict):
            return insights

        model_name = pred_data.get("selected_model") or pred_data.get("best_model", "Supervised Model")
        metrics = pred_data.get("metrics") or pred_data.get("validation_metrics", {})
        target = pred_data.get("target_column") or "Target"
        task_type = pred_data.get("task_type", "supervised learning")

        metric_strs = []
        if "r2" in metrics:
            metric_strs.append(f"R² = {metrics['r2']:.3f}")
        if "accuracy" in metrics:
            metric_strs.append(f"Accuracy = {metrics['accuracy']:.1%}")
        if "f1" in metrics or "f1_score" in metrics:
            f1_val = metrics.get("f1", metrics.get("f1_score", 0.0))
            metric_strs.append(f"F1 = {f1_val:.3f}")

        perf_str = f" with performance ({', '.join(metric_strs)})" if metric_strs else ""
        stmt = f"{model_name} benchmarked for {task_type} of '{target}'{perf_str}."
        stmt = self._sanitize_causality(stmt)

        ev = Evidence(
            operation="prediction.model_evaluation",
            columns=[str(target)],
            result=pred_data,
            confidence=0.87,
            claim_type=ClaimType.INFERENCE,
        )

        insights.append(
            SynthesizedInsight(
                category=InsightCategory.PREDICTIVE_PERFORMANCE.value,
                title=f"Predictive Model Benchmark: {target}",
                statement=stmt,
                evidence_refs=[ev],
                supporting_metrics=metrics,
                confidence=0.87,
                importance=0.80,
                limitations=["Feature importance metrics describe model partition weights and do not prove causal influence."],
                provenance={"agent": "PredictionAgent", "target": target},
            )
        )

        return insights

    def _synthesize_data_quality_gate(self, dq_data: Optional[Dict[str, Any]]) -> List[SynthesizedInsight]:
        """Synthesize data quality gate readiness insights."""
        insights: List[SynthesizedInsight] = []
        if not dq_data or not isinstance(dq_data, dict):
            return insights

        status = dq_data.get("status", "READY")
        rec_actions = dq_data.get("recommended_actions", [])

        stmt = f"Data quality pre-analysis gate verified dataset status as '{status}'."
        if rec_actions:
            stmt += f" Recommended preparatory steps: {'; '.join(rec_actions[:2])}."

        ev = Evidence(
            operation="data_quality_gate.validation",
            result=dq_data,
            confidence=0.95,
            claim_type=ClaimType.OBSERVATION,
        )

        insights.append(
            SynthesizedInsight(
                category=InsightCategory.DATA_QUALITY.value,
                title="Data Quality Gate Pre-Analysis Audit",
                statement=stmt,
                evidence_refs=[ev],
                supporting_metrics=dq_data,
                confidence=0.95,
                importance=0.65,
                provenance={"agent": "DataQualityAgent"},
            )
        )

        return insights

    # --------------------------------------------------------------------------
    # Cross-Agent Reasoning & Contradiction Detection
    # --------------------------------------------------------------------------

    def _synthesize_cross_agent(
        self,
        insights: List[SynthesizedInsight],
        task_outputs: Dict[str, Any],
    ) -> List[SynthesizedInsight]:
        """Detect cross-agent agreement to form high-confidence synthesized insights."""
        cross_insights: List[SynthesizedInsight] = []

        # Check for Forecast + Correlation agreement
        fc_data = task_outputs.get("forecasting")
        stats_data = task_outputs.get("statistical_analysis")

        if fc_data and stats_data:
            fc_trend = fc_data.get("trend_direction", "")
            target = fc_data.get("target_column")
            if target:
                # Find correlations with time or positive associations
                rels = stats_data.get("relationships", [])
                pos_rels = [r for r in rels if (r.get("feature_1") == target or r.get("feature_2") == target) and r.get("correlation", 0.0) > 0.50]
                if pos_rels and "up" in fc_trend.lower():
                    stmt = f"Cross-analysis synthesis: Upward forecast projection for '{target}' is corroborated by strong positive empirical feature relationships in historical observations."
                    ev = Evidence(
                        operation="cross_agent.agreement",
                        columns=[str(target)],
                        result={"forecast_trend": fc_trend, "corroborating_relationships": pos_rels},
                        confidence=0.92,
                        claim_type=ClaimType.INFERENCE,
                    )
                    cross_insights.append(
                        SynthesizedInsight(
                            category=InsightCategory.CROSS_ANALYSIS.value,
                            title=f"Cross-Agent Trend Corroboration: {target}",
                            statement=stmt,
                            evidence_refs=[ev],
                            confidence=0.92,
                            importance=0.88,
                            provenance={"agents": ["ForecastAgent", "StatisticalAnalysisAgent"]},
                        )
                    )

        return cross_insights

    def _detect_contradictions(
        self,
        insights: List[SynthesizedInsight],
        task_outputs: Dict[str, Any],
    ) -> List[Contradiction]:
        """Detect analytical conflicts between tasks without forcing artificial agreement."""
        contradictions: List[Contradiction] = []
        if not isinstance(task_outputs, dict):
            return contradictions

        # Check: Forecast Trend vs Historical Trend / Anomaly Spikes
        fc_data = task_outputs.get("forecasting")
        anom_data = task_outputs.get("anomaly_detection")

        if fc_data and anom_data and isinstance(fc_data, dict) and isinstance(anom_data, dict):
            anom_c = int(self._safe_float(anom_data.get("anomaly_count", 0)))
            fc_conf = self._safe_float(fc_data.get("confidence", 0.8), 0.8)
            if anom_c > 10 and fc_conf > 0.85:
                contra = Contradiction(
                    involved_insights=[i.insight_id for i in insights if i.category in (InsightCategory.FORECAST.value, InsightCategory.ANOMALY.value)],
                    conflicting_evidence=[{"task": "forecasting", "confidence": fc_conf}, {"task": "anomaly_detection", "anomaly_count": anom_c}],
                    explanation=f"High forecasting confidence ({fc_conf:.2f}) coincides with an elevated anomaly count ({anom_c} outlier observations), which may affect interval coverage.",
                    confidence=0.75,
                    resolution="Maintain wide prediction intervals and monitor post-anomaly stationarity.",
                )
                contradictions.append(contra)

        return contradictions

    # --------------------------------------------------------------------------
    # Prioritization, Deduplication & Narrative Helpers
    # --------------------------------------------------------------------------

    def _suppress_duplicates(self, insights: List[SynthesizedInsight]) -> List[SynthesizedInsight]:
        """Merge identical or symmetric insights while preserving supporting evidence."""
        merged: List[SynthesizedInsight] = []
        seen_keys: Set[str] = set()

        for ins in insights:
            # Symmetrical relationship key
            if ins.category == InsightCategory.RELATIONSHIP.value and "pair" in ins.provenance:
                parts = sorted(str(ins.provenance["pair"]).split("_"))
                key = f"rel_{'_'.join(parts)}"
            else:
                # Key by category and normalized title
                key = f"{ins.category}_{re.sub(r'[^a-zA-Z0-9]', '', ins.title.lower())}"

            if key in seen_keys:
                # Merge into existing
                for existing in merged:
                    if existing.title == ins.title or (existing.category == ins.category and existing.category == InsightCategory.RELATIONSHIP.value):
                        existing.evidence_refs.extend(ins.evidence_refs)
                        existing.confidence = max(existing.confidence, ins.confidence)
                        break
            else:
                seen_keys.add(key)
                merged.append(ins)

        return merged

    def _calculate_importance_and_rank(self, insights: List[SynthesizedInsight]) -> List[SynthesizedInsight]:
        """Rank insights deterministically by importance score [0.0, 1.0]."""
        for ins in insights:
            # Deterministic importance formula
            cat_weight = {
                InsightCategory.CROSS_ANALYSIS.value: 0.90,
                InsightCategory.FORECAST.value: 0.85,
                InsightCategory.PREDICTIVE_PERFORMANCE.value: 0.82,
                InsightCategory.ANOMALY.value: 0.80,
                InsightCategory.RELATIONSHIP.value: 0.75,
                InsightCategory.SEGMENT.value: 0.72,
                InsightCategory.DATA_QUALITY.value: 0.70,
                InsightCategory.DISTRIBUTION.value: 0.60,
                InsightCategory.LIMITATION.value: 0.50,
            }.get(ins.category, 0.50)

            raw_imp = 0.50 * cat_weight + 0.50 * ins.confidence
            ins.importance = round(max(0.0, min(1.0, raw_imp)), 4)

        return sorted(insights, key=lambda x: (x.importance, x.confidence, x.title), reverse=True)

    def _sanitize_causality(self, text: str) -> str:
        """Sanitize causal phrasing into safe observational language."""
        sanitized = text
        for pattern, replacement in self.CAUSAL_REPLACEMENTS:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        return sanitized

    def _generate_next_questions(
        self,
        ranked_insights: List[SynthesizedInsight],
        contradictions: List[Contradiction],
        dq_findings: List[SynthesizedInsight],
    ) -> List[str]:
        """Generate evidence-backed follow-up analytical questions."""
        questions: List[str] = []

        for ins in ranked_insights:
            if ins.category == InsightCategory.ANOMALY.value and ins.supporting_metrics.get("anomaly_count", 0) > 0:
                questions.append("What specific contextual factors coincide with the detected outlier timestamps/records?")
            elif ins.category == InsightCategory.RELATIONSHIP.value:
                pair = ins.provenance.get("pair")
                if pair:
                    questions.append(f"Does the statistical association between {pair.replace('_', ' and ')} persist across subgroup segments?")
            elif ins.category == InsightCategory.SEGMENT.value:
                questions.append("How do key operational metrics differ across the discovered cluster segments?")
            elif ins.category == InsightCategory.FORECAST.value:
                target = ins.provenance.get("target", "the target")
                questions.append(f"What scenario adjustments would impact the projected trajectory for {target}?")

        if contradictions:
            questions.append("How do anomaly distributions affect the stability of forecast prediction intervals?")

        if not questions:
            questions.append("What additional feature variables could explain remaining variance in key metrics?")

        return list(dict.fromkeys(questions))[:4]

    def _generate_executive_summary(
        self,
        ranked_insights: List[SynthesizedInsight],
        contradictions: List[Contradiction],
        command: str,
        task_outputs: Dict[str, Any],
    ) -> str:
        """Generate high-level cohesive executive narrative summary."""
        parts: List[str] = []

        if command:
            parts.append(f"Comprehensive analytical synthesis for command: '{command}'.")

        n_tasks = len(task_outputs) if isinstance(task_outputs, dict) else 0
        parts.append(f"Synthesized evidence from {n_tasks} validated analytical task(s).")

        if ranked_insights:
            top_insight = ranked_insights[0]
            parts.append(f"Primary analytical finding: {top_insight.statement}")

        if contradictions:
            parts.append(f"Noted {len(contradictions)} analytical tension(s) requiring contextual monitoring.")

        return " ".join(parts)

    # --------------------------------------------------------------------------
    # Output Parsing Helpers
    # --------------------------------------------------------------------------

    def _extract_task_outputs(self, res: Union[AgentResult, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(res, AgentResult):
            if isinstance(res.data, dict) and "task_outputs" in res.data and isinstance(res.data["task_outputs"], dict):
                return res.data["task_outputs"]
            if isinstance(res.output, dict) and "task_outputs" in res.output and isinstance(res.output["task_outputs"], dict):
                return res.output["task_outputs"]
            if isinstance(res.data, dict) and "tasks" in res.data and isinstance(res.data["tasks"], dict):
                return res.data["tasks"]
            return res.data if isinstance(res.data, dict) else {}
        elif isinstance(res, dict):
            if "task_outputs" in res and isinstance(res["task_outputs"], dict):
                return res["task_outputs"]
            if "result" in res and isinstance(res["result"], dict) and "task_outputs" in res["result"] and isinstance(res["result"]["task_outputs"], dict):
                return res["result"]["task_outputs"]
            if "tasks" in res and isinstance(res["tasks"], dict):
                return res["tasks"]
            return res if isinstance(res, dict) else {}
        return {}

    def _extract_evidence(self, res: Union[AgentResult, Dict[str, Any]]) -> List[Evidence]:
        if isinstance(res, AgentResult):
            return res.evidence
        if isinstance(res, dict) and "evidence" in res:
            raw_ev = res["evidence"]
            if isinstance(raw_ev, list):
                return [Evidence(**e) if isinstance(e, dict) else e for e in raw_ev if isinstance(e, (dict, Evidence))]
        return []

    def _extract_confidence(self, res: Union[AgentResult, Dict[str, Any]]) -> float:
        if isinstance(res, AgentResult):
            return float(res.confidence)
        if isinstance(res, dict):
            return float(res.get("confidence", 0.85))
        return 0.85

    def _extract_command(self, res: Union[AgentResult, Dict[str, Any]]) -> str:
        if isinstance(res, AgentResult):
            if isinstance(res.data, dict) and "user_request" in res.data:
                return str(res.data["user_request"])
            return str(res.message or "")
        if isinstance(res, dict):
            return str(res.get("user_request", res.get("command", "")))
        return ""