"""Intent Analyzer Engine for Natural Language Analytical Understanding.

Analyzes natural language queries and extracts:
- Primary and secondary analytical intents (EDA, Cleaning, ML, ANN, CNN, Forecasting, Visualization, Explanation, Report)
- Target variables and feature references
- Time horizons and temporal constraints
- Top-k drivers and ranking requirements
- Confidence scores and traceable reasoning
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import re

import pandas as pd

from backend.app.core.dataset_knowledge import DatasetKnowledge


class AnalyticalIntent(str, Enum):
    EDA = "eda"
    CLEANING = "cleaning"
    VISUALIZATION = "visualization"
    PREDICTION = "prediction"
    FORECASTING = "forecasting"
    DEEP_LEARNING = "deep_learning"
    CNN = "cnn"
    EXPLANATION = "explanation"
    ANOMALIES = "anomalies"
    REPORT = "report"
    UNKNOWN = "unknown"


@dataclass
class IntentClassificationResult:
    """Detailed intent breakdown for analytical task planning."""
    primary_intent: AnalyticalIntent
    secondary_intents: List[AnalyticalIntent] = field(default_factory=list)
    confidence: float = 0.9
    target_column: Optional[str] = None
    feature_columns: List[str] = field(default_factory=list)
    group_by: Optional[str] = None
    time_horizon: Optional[int] = None
    chart_type: Optional[str] = None
    top_k: Optional[int] = None
    needs_cleaning: bool = False
    needs_explanation: bool = False
    raw_query: str = ""
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_intent": self.primary_intent.value,
            "secondary_intents": [i.value for i in self.secondary_intents],
            "confidence": round(self.confidence, 3),
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "group_by": self.group_by,
            "time_horizon": self.time_horizon,
            "chart_type": self.chart_type,
            "top_k": self.top_k,
            "needs_cleaning": self.needs_cleaning,
            "needs_explanation": self.needs_explanation,
            "raw_query": self.raw_query,
            "reasoning": self.reasoning,
        }


class IntentAnalyzer:
    """Deterministic, contextual intent extractor with semantic knowledge integration."""

    INTENT_KEYWORDS = {
        AnalyticalIntent.CLEANING: (
            "clean", "preprocess", "impute", "missing", "outlier removal",
            "dedup", "duplicates", "null values", "format data", "sanitize",
        ),
        AnalyticalIntent.FORECASTING: (
            "forecast", "future", "predict next", "time series", "horizon",
            "next month", "next quarter", "next year", "projection", "trend over time",
        ),
        AnalyticalIntent.CNN: (
            "cnn", "convolutional", "image", "spatial", "spectrogram", "signal image",
            "pixel", "grid data", "image classification",
        ),
        AnalyticalIntent.DEEP_LEARNING: (
            "deep learning", "ann", "neural network", "mlp", "multi-layer perceptron",
            "deep model", "pytorch",
        ),
        AnalyticalIntent.PREDICTION: (
            "train", "predict", "classifier", "regression", "model", "churn",
            "supervised", "forecast target", "random forest", "logistic regression",
            "xgboost", "fit model",
        ),
        AnalyticalIntent.EXPLANATION: (
            "explain", "driver", "drivers", "feature importance", "why",
            "top factors", "influence", "key indicators", "shap", "coefficients",
        ),
        AnalyticalIntent.VISUALIZATION: (
            "chart", "plot", "graph", "visualize", "bar chart", "line chart",
            "scatter", "histogram", "heatmap", "distribution plot",
        ),
        AnalyticalIntent.ANOMALIES: (
            "anomaly", "anomalies", "outlier", "outliers", "unusual", "deviations",
        ),
        AnalyticalIntent.REPORT: (
            "report", "summary report", "executive summary", "overview",
            "full analysis", "brief", "presentation",
        ),
        AnalyticalIntent.EDA: (
            "eda", "describe", "summary", "stats", "statistics", "correlations",
            "explore", "profile", "distribution", "overview",
        ),
    }

    def analyze(
        self,
        query: str,
        knowledge: Optional[DatasetKnowledge] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> IntentClassificationResult:
        """Parse natural language query into a structured multi-intent profile."""
        q_norm = query.strip().lower()
        reasoning: List[str] = []
        matched_intents: List[AnalyticalIntent] = []

        # 1. Match all intents mentioned in the query
        for intent, kw_list in self.INTENT_KEYWORDS.items():
            hits = [kw for kw in kw_list if kw in q_norm or f" {kw} " in f" {q_norm} "]
            if hits:
                matched_intents.append(intent)
                reasoning.append(f"Matched {intent.value} via keywords: {', '.join(hits)}")

        # Distinguish prediction vs forecasting vs deep learning
        primary = AnalyticalIntent.EDA
        secondary: List[AnalyticalIntent] = []

        if AnalyticalIntent.CNN in matched_intents:
            primary = AnalyticalIntent.CNN
        elif AnalyticalIntent.DEEP_LEARNING in matched_intents:
            primary = AnalyticalIntent.DEEP_LEARNING
        elif AnalyticalIntent.FORECASTING in matched_intents:
            primary = AnalyticalIntent.FORECASTING
        elif AnalyticalIntent.PREDICTION in matched_intents:
            primary = AnalyticalIntent.PREDICTION
        elif AnalyticalIntent.CLEANING in matched_intents:
            primary = AnalyticalIntent.CLEANING
        elif AnalyticalIntent.EXPLANATION in matched_intents:
            primary = AnalyticalIntent.EXPLANATION
        elif AnalyticalIntent.VISUALIZATION in matched_intents:
            primary = AnalyticalIntent.VISUALIZATION
        elif AnalyticalIntent.ANOMALIES in matched_intents:
            primary = AnalyticalIntent.ANOMALIES
        elif AnalyticalIntent.REPORT in matched_intents:
            primary = AnalyticalIntent.REPORT
        elif matched_intents:
            primary = matched_intents[0]

        secondary = [i for i in matched_intents if i != primary]

        # 2. Extract Top-K requirement (e.g. "top 3 drivers", "top 5 factors")
        top_k = None
        top_k_match = re.search(r"top\s+(\d+)", q_norm)
        if top_k_match:
            top_k = int(top_k_match.group(1))
            reasoning.append(f"Detected top-{top_k} limit")

        # 3. Extract time horizon (e.g. "next 6 months", "next 3 periods")
        horizon = None
        horizon_match = re.search(r"next\s+(\d+)\s*(?:month|day|week|quarter|period|year)?", q_norm)
        if horizon_match:
            horizon = int(horizon_match.group(1))
            reasoning.append(f"Detected time horizon of {horizon} periods")

        # 4. Extract Chart Type
        chart_type = None
        for ctype in ("bar", "line", "scatter", "histogram", "heatmap", "box", "pie"):
            if ctype in q_norm or f"{ctype} chart" in q_norm:
                chart_type = ctype
                reasoning.append(f"Detected requested chart type: {ctype}")
                break

        # 5. Extract Target Column and Features using DatasetKnowledge / DataFrame
        target_col = None
        feature_cols: List[str] = []
        group_by_col = None

        columns: List[str] = []
        if knowledge is not None:
            columns = knowledge.columns
        elif dataframe is not None:
            columns = list(dataframe.columns)

        if columns:
            # Check for explicitly mentioned column names
            for col in columns:
                col_lower = col.lower()
                if col_lower in q_norm or col_lower.replace("_", " ") in q_norm:
                    if target_col is None and (
                        "predict" in q_norm or "target" in q_norm or "forecast" in q_norm
                    ):
                        target_col = col
                        reasoning.append(f"Found target column '{col}' in query")
                    else:
                        feature_cols.append(col)

            # Check semantic concepts if target not explicitly named (e.g. "predict churn")
            if target_col is None and knowledge is not None:
                for concept in ("churn", "revenue", "profit", "sales", "price", "salary"):
                    if concept in q_norm:
                        matched = knowledge.find_columns_by_concept(concept)
                        if matched:
                            target_col = matched[0]
                            reasoning.append(f"Mapped concept '{concept}' to column '{target_col}'")
                            break

            # If still none, fallback to primary metric for prediction/forecasting
            if target_col is None and primary in (
                AnalyticalIntent.PREDICTION,
                AnalyticalIntent.FORECASTING,
                AnalyticalIntent.DEEP_LEARNING,
            ):
                if knowledge is not None:
                    target_col = knowledge.get_primary_metric()
                    if target_col:
                        reasoning.append(f"Defaulted target to primary metric '{target_col}'")

        needs_cleaning = AnalyticalIntent.CLEANING in matched_intents or "clean" in q_norm
        needs_explanation = (
            AnalyticalIntent.EXPLANATION in matched_intents
            or "explain" in q_norm
            or "driver" in q_norm
            or top_k is not None
        )

        confidence = 0.95 if matched_intents else 0.70

        return IntentClassificationResult(
            primary_intent=primary,
            secondary_intents=secondary,
            confidence=confidence,
            target_column=target_col,
            feature_columns=feature_cols,
            group_by=group_by_col,
            time_horizon=horizon,
            chart_type=chart_type,
            top_k=top_k,
            needs_cleaning=needs_cleaning,
            needs_explanation=needs_explanation,
            raw_query=query,
            reasoning=reasoning,
        )
