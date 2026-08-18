"""
Core schemas for the Multi-Agent System.

These dataclasses define the structured contracts that every agent and the
Master Orchestrator use to exchange information:

- ``Artifact`` -- a typed, named container for data that flows between agents
- ``AgentInput`` -- what an agent receives when ``execute()`` is called
- ``AgentOutput`` -- what an agent returns after execution (status, artifacts, logs)
- ``Task`` / ``TaskGraph`` -- the decomposed, dependency-ordered plan
- ``OrchestrationResult`` -- the final assembled result from the orchestrator
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExecutionStatus(str, Enum):
    """Lifecycle states an agent or task can be in."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


class ArtifactType(str, Enum):
    """Every artifact produced or consumed by an agent is tagged with one of these types."""
    # --- Data ---
    DATAFRAME = "dataframe"
    CLEANED_DATAFRAME = "cleaned_dataframe"
    DATA_PROFILE = "data_profile"

    # --- Cleaning ---
    CLEANING_REPORT = "cleaning_report"

    # --- EDA ---
    EDA_SUMMARY = "eda_summary"
    EDA_STATISTICS = "eda_statistics"
    EDA_CORRELATIONS = "eda_correlations"
    EDA_ANOMALIES = "eda_anomalies"
    EDA_DISTRIBUTIONS = "eda_distributions"
    EDA_CATEGORICAL = "eda_categorical"
    EDA_TIME_SERIES = "eda_time_series"
    EDA_RECOMMENDATIONS = "eda_recommendations"

    # --- Visualization ---
    CHART_RECOMMENDATIONS = "chart_recommendations"
    CHARTS = "charts"

    # --- Insights ---
    FACTS = "facts"
    INSIGHTS = "insights"
    RECOMMENDATIONS = "recommendations"

    # --- Forecasting ---
    FORECAST_RESULT = "forecast_result"

    # --- Reports ---
    REPORT = "report"
    REPORT_PDF = "report_pdf"

    # --- Chat ---
    CHAT_RESPONSE = "chat_response"
    CHAT_HISTORY = "chat_history"
