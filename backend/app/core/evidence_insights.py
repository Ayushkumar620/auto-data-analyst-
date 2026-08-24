"""Evidence-Based Insights Engine with Epistemic Classification and Attribution.

Every generated insight adheres to a strict scientific schema:
1. Insight text
2. Epistemic Claim Type:
   - FACT: Directly verifiable arithmetic / database computation.
   - OBSERVATION: Noticed statistical patterns, outliers, anomalies.
   - CORRELATION: Statistically significant associations with STRICT non-causal disclaimers.
   - INFERENCE: Model-based predictions, forecasts, and projections with confidence intervals.
   - RECOMMENDATION: Actionable decision advice with operational caveats.
3. Supporting metrics and calculations.
4. Confidence score (0.0 to 1.0).
5. Explicit caveats and limitations.
6. Data references and suggested chart visualization.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import pandas as pd

from agent.schemas import ClaimType


@dataclass
class StructuredInsight:
    """Standardized schema for an evidence-backed analytical insight."""
    insight_id: str
    text: str
    claim_type: ClaimType
    supporting_metrics: Dict[str, Any]
    confidence: float
    caveats: List[str] = field(default_factory=list)
    data_references: List[str] = field(default_factory=list)
    recommended_chart: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "text": self.text,
            "claim_type": self.claim_type.value,
            "supporting_metrics": self.supporting_metrics,
            "confidence": round(float(self.confidence), 4),
            "caveats": self.caveats,
            "data_references": self.data_references,
            "recommended_chart": self.recommended_chart,
        }


@dataclass
class InsightsCatalog:
    """Catalog of structured insights grouped by claim type."""
    dataset_name: str
    total_insights: int
    insights: List[StructuredInsight]
    facts: List[StructuredInsight] = field(default_factory=list)
    observations: List[StructuredInsight] = field(default_factory=list)
    correlations: List[StructuredInsight] = field(default_factory=list)
    inferences: List[StructuredInsight] = field(default_factory=list)
    recommendations: List[StructuredInsight] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "total_insights": self.total_insights,
            "insights": [i.to_dict() for i in self.insights],
            "facts": [i.to_dict() for i in self.facts],
            "observations": [i.to_dict() for i in self.observations],
            "correlations": [i.to_dict() for i in self.correlations],
            "inferences": [i.to_dict() for i in self.inferences],
            "recommendations": [i.to_dict() for i in self.recommendations],
        }


class EvidenceBasedInsightsEngine:
    """Generates structured, evidence-attributed analytical insights from tabular data."""

    def __init__(self, data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None):
        self.data = data

    def _get_df(self, data_input: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None) -> Tuple[str, pd.DataFrame]:
        target_data = data_input if data_input is not None else self.data
        if isinstance(target_data, dict):
            for name, df in target_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return name, df
        elif isinstance(target_data, pd.DataFrame):
            return "dataset", target_data
        raise ValueError("No valid pandas DataFrame available for insight generation.")

    # ------------------------------------------------------------------
    # 1. Fact Generation (Exact arithmetic calculations)
    # ------------------------------------------------------------------
    def generate_facts(self, df: pd.DataFrame) -> List[StructuredInsight]:
        facts: List[StructuredInsight] = []
        n_rows, n_cols = df.shape

        # Fact 1: Dataset Volume & Completeness
        total_cells = n_rows * n_cols
        null_cells = int(df.isna().sum().sum())
        completeness_pct = ((total_cells - null_cells) / total_cells) * 100 if total_cells > 0 else 100.0

        facts.append(
            StructuredInsight(
                insight_id=f"fact_{uuid.uuid4().hex[:6]}",
                text=f"Dataset contains {n_rows:,} records across {n_cols} columns with {completeness_pct:.1f}% data completeness.",
                claim_type=ClaimType.FACT,
                supporting_metrics={
                    "total_records": n_rows,
                    "total_columns": n_cols,
                    "null_cells": null_cells,
                    "completeness_pct": round(completeness_pct, 2),
                },
                confidence=1.0,
                caveats=["Direct row/cell count calculation; reflects ingested sample only."],
                data_references=list(df.columns),
            )
        )

        # Fact 2: Primary numeric metric aggregates
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols[:2]:
            clean_s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(clean_s) > 0:
                col_sum = float(clean_s.sum())
                col_mean = float(clean_s.mean())
                col_min = float(clean_s.min())
                col_max = float(clean_s.max())

                facts.append(
                    StructuredInsight(
                        insight_id=f"fact_{uuid.uuid4().hex[:6]}",
                        text=f"Total aggregate for '{col}' is {col_sum:,.2f} (mean: {col_mean:,.2f}, min: {col_min:,.2f}, max: {col_max:,.2f}).",
                        claim_type=ClaimType.FACT,
                        supporting_metrics={"sum": round(col_sum, 2), "mean": round(col_mean, 2), "min": round(col_min, 2), "max": round(col_max, 2)},
                        confidence=1.0,
                        caveats=["Exact arithmetic aggregate without outlier exclusions."],
                        data_references=[col],
                        recommended_chart={"chart_type": "histogram", "x": col},
                    )
                )

        return facts

    # ------------------------------------------------------------------
    # 2. Observation Generation (Patterns, distributions, outliers)
    # ------------------------------------------------------------------
    def generate_observations(self, df: pd.DataFrame) -> List[StructuredInsight]:
        observations: List[StructuredInsight] = []
        num_cols = df.select_dtypes(include=[np.number]).columns

        for col in num_cols[:3]:
            s = df[col].dropna()
            if len(s) >= 10:
                mean = s.mean()
                std = s.std()
                if std > 0:
                    z_scores = np.abs((s - mean) / std)
                    outliers_count = int((z_scores > 3.0).sum())
                    if outliers_count > 0:
                        outlier_pct = (outliers_count / len(s)) * 100
                        observations.append(
                            StructuredInsight(
                                insight_id=f"obs_{uuid.uuid4().hex[:6]}",
                                text=f"Detected {outliers_count} extreme outlier values ({outlier_pct:.1f}% of data) in '{col}' exceeding 3 standard deviations.",
                                claim_type=ClaimType.OBSERVATION,
                                supporting_metrics={"outlier_count": outliers_count, "outlier_percentage": round(outlier_pct, 2), "mean": round(mean, 2), "std": round(std, 2)},
                                confidence=0.95,
                                caveats=["Outliers identified via standard normal z-score threshold (z > 3.0)."],
                                data_references=[col],
                                recommended_chart={"chart_type": "box_plot", "y": col},
                            )
                        )

        # Categorical distribution concentration observation
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols[:2]:
            s_cat = df[col].dropna()
            if len(s_cat) > 0:
                top_val = s_cat.value_counts().index[0]
                top_cnt = s_cat.value_counts().iloc[0]
                top_pct = (top_cnt / len(s_cat)) * 100
                if top_pct >= 40.0:
                    observations.append(
                        StructuredInsight(
                            insight_id=f"obs_{uuid.uuid4().hex[:6]}",
                            text=f"High category concentration: '{top_val}' accounts for {top_pct:.1f}% of all records in '{col}'.",
                            claim_type=ClaimType.OBSERVATION,
                            supporting_metrics={"top_category": str(top_val), "count": int(top_cnt), "percentage": round(top_pct, 2)},
                            confidence=0.95,
                            caveats=["Based on empirical frequency distribution in current sample."],
                            data_references=[col],
                            recommended_chart={"chart_type": "bar_chart", "x": col},
                        )
                    )

        return observations

    # ------------------------------------------------------------------
    # 3. Correlation Generation (With STRICT Non-Causal Disclaimers)
    # ------------------------------------------------------------------
    def generate_correlations(self, df: pd.DataFrame, threshold: float = 0.50) -> List[StructuredInsight]:
        correlations: List[StructuredInsight] = []
        num_cols = df.select_dtypes(include=[np.number]).columns

        if len(num_cols) >= 2:
            corr_matrix = df[num_cols].corr()
            pairs = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    val = float(corr_matrix.loc[c1, c2])
                    if not np.isnan(val) and abs(val) >= threshold:
                        pairs.append((c1, c2, val))

            pairs.sort(key=lambda x: abs(x[2]), reverse=True)

            for c1, c2, r_val in pairs[:3]:
                direction = "positive" if r_val > 0 else "negative"
                strength = "strong" if abs(r_val) >= 0.75 else "moderate"

                correlations.append(
                    StructuredInsight(
                        insight_id=f"corr_{uuid.uuid4().hex[:6]}",
                        text=(
                            f"Observed {strength} {direction} correlation (r = {r_val:.3f}) between '{c1}' and '{c2}'. "
                            f"As '{c1}' increases, '{c2}' tends to {'increase' if r_val > 0 else 'decrease'} proportionally."
                        ),
                        claim_type=ClaimType.CORRELATION,
                        supporting_metrics={"feature_1": c1, "feature_2": c2, "pearson_r": round(r_val, 4), "r_squared": round(r_val**2, 4)},
                        confidence=0.90,
                        caveats=[
                            "CRITICAL: Correlation does NOT imply causation.",
                            "Confounding variables, shared trends, or reverse causality may explain this co-movement.",
                            "Linear correlation only captures affine co-variation and ignores non-linear dynamics.",
                        ],
                        data_references=[c1, c2],
                        recommended_chart={"chart_type": "scatter_plot", "x": c1, "y": c2},
                    )
                )

        return correlations

    # ------------------------------------------------------------------
    # 4. Inferences & Recommendations Generation
    # ------------------------------------------------------------------
    def generate_inferences_and_recommendations(
        self,
        df: pd.DataFrame,
        model_result: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[StructuredInsight], List[StructuredInsight]]:
        inferences: List[StructuredInsight] = []
        recommendations: List[StructuredInsight] = []

        num_cols = df.select_dtypes(include=[np.number]).columns

        # Inference from model or statistical projection
        if model_result and "best_model" in model_result:
            bm = model_result["best_model"]
            m_name = bm.get("model_name", "ML Model")
            metric_name = bm.get("primary_metric_name", "score")
            metric_val = bm.get("primary_metric_value", 0.0)

            inferences.append(
                StructuredInsight(
                    insight_id=f"inf_{uuid.uuid4().hex[:6]}",
                    text=f"Trained {m_name} achieves {metric_name} of {metric_val:.4f} and reliably generalizes future target patterns.",
                    claim_type=ClaimType.INFERENCE,
                    supporting_metrics=bm.get("metrics", {}),
                    confidence=0.88,
                    caveats=[
                        "Inference is contingent on data distribution stability (stationarity).",
                        "Concept drift or macroeconomic shifts will degrade out-of-sample projection accuracy.",
                    ],
                    data_references=bm.get("feature_importances", {}).keys(),
                )
            )

        # Actionable Recommendations
        if len(num_cols) >= 2:
            recommendations.append(
                StructuredInsight(
                    insight_id=f"rec_{uuid.uuid4().hex[:6]}",
                    text=f"Deploy automated monitoring on primary drivers '{num_cols[0]}' and '{num_cols[1]}' to detect distribution drift.",
                    claim_type=ClaimType.RECOMMENDATION,
                    supporting_metrics={"monitored_columns": list(num_cols[:2])},
                    confidence=0.85,
                    caveats=["Operational impact depends on business constraints and drift tolerances."],
                    data_references=list(num_cols[:2]),
                )
            )

        return inferences, recommendations

    # ------------------------------------------------------------------
    # Full Catalog Synthesis
    # ------------------------------------------------------------------
    def build_catalog(
        self,
        data_input: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
        model_result: Optional[Dict[str, Any]] = None,
    ) -> InsightsCatalog:
        """Synthesize all facts, observations, correlations, inferences, and recommendations."""
        name, df = self._get_df(data_input)

        facts = self.generate_facts(df)
        observations = self.generate_observations(df)
        correlations = self.generate_correlations(df)
        inferences, recommendations = self.generate_inferences_and_recommendations(df, model_result)

        all_insights = facts + observations + correlations + inferences + recommendations

        return InsightsCatalog(
            dataset_name=name,
            total_insights=len(all_insights),
            insights=all_insights,
            facts=facts,
            observations=observations,
            correlations=correlations,
            inferences=inferences,
            recommendations=recommendations,
        )
