"""
Tool & Agent Registry for Dynamic Multi-Agent Analytical Orchestration.

Maintains an explicit, type-safe registry of all analytical tools and agents:
- ToolDefinition: Metadata, capabilities, input/output schemas, and execution wrappers
- ToolRegistry: Capability matching, discovery, validation, and safe invocation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union
import inspect
import pandas as pd

from agent.schemas import AgentResult


@dataclass
class ToolDefinition:
    """Explicit capability contract for an analytical tool or agent."""
    name: str
    description: str
    capabilities: List[str] = field(default_factory=list)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    supported_data_types: List[str] = field(default_factory=lambda: ["tabular", "time_series", "dataframe"])
    dependencies: List[str] = field(default_factory=list)
    execution_fn: Optional[Callable[..., Any]] = None
    validation_requirements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "supported_data_types": self.supported_data_types,
            "dependencies": self.dependencies,
            "validation_requirements": self.validation_requirements,
        }


class ToolRegistry:
    """Central registry and capability broker for all analytical operations."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def register(self, tool: ToolDefinition) -> None:
        """Register a new tool definition."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Look up tool by name."""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if tool exists."""
        return name in self._tools

    def list_tools(self) -> List[ToolDefinition]:
        """Return list of all registered tool definitions."""
        return list(self._tools.values())

    def get_tools_by_capability(self, capability: str) -> List[ToolDefinition]:
        """Find tools supporting a specific capability (e.g. 'forecasting', 'cleaning')."""
        cap_norm = capability.lower().strip()
        matches: List[ToolDefinition] = []
        for tool in self._tools.values():
            if any(c.lower().strip() == cap_norm for c in tool.capabilities):
                matches.append(tool)
        return matches

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Safely invoke registered tool execution function."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' is not registered in ToolRegistry.")
        tool = self._tools[name]
        if not tool.execution_fn:
            raise NotImplementedError(f"Tool '{name}' has no registered execution function.")
        return tool.execution_fn(**kwargs)

    # ------------------------------------------------------------------
    # Default Tool Registration
    # ------------------------------------------------------------------
    def _register_default_tools(self) -> None:
        """Register all default analytical and data engineering tools."""
        from agent.agents import (
            AnalysisAgent,
            CleaningAgent,
            DataValidationAgent,
            ForecastAgent,
            InsightAgent,
            ModelRegistryAgent,
            ModelSelectionAgent,
            PredictionAgent,
            ReportAgent,
            VisualizationAgent,
        )

        # 1. Dataset Profiling & Validation
        self.register(
            ToolDefinition(
                name="dataset_profiling",
                description="Profiles dataset schema, data quality, nulls, duplicates, and column types.",
                capabilities=["dataset_profiling", "data_validation", "schema_profiling"],
                input_schema={"data": "pd.DataFrame"},
                output_schema={"profile": "dict", "quality_score": "float"},
                execution_fn=lambda data, **kw: DataValidationAgent().run({"data": data, **kw}),
                validation_requirements=["quality_score >= 0"],
            )
        )

        # 2. Data Cleaning & Sanitization
        self.register(
            ToolDefinition(
                name="data_cleaning",
                description="Performs imputation of missing values, outlier treatment, and deduplication.",
                capabilities=["data_cleaning", "duplicate_handling", "imputation", "sanitization"],
                input_schema={"data": "pd.DataFrame", "strategy": "str"},
                output_schema={"cleaned_data": "pd.DataFrame", "report": "dict"},
                execution_fn=lambda data, **kw: CleaningAgent().run({"data": data, **kw}),
                validation_requirements=["cleaned_data is not empty"],
            )
        )

        # 3. Exploratory Data Analysis & Statistics
        self.register(
            ToolDefinition(
                name="eda",
                description="Computes summary statistics, distributions, quartiles, and descriptive metrics.",
                capabilities=["eda", "statistics", "descriptive_stats", "summary"],
                input_schema={"data": "pd.DataFrame", "request": "str"},
                output_schema={"summary": "dict", "statistics": "dict"},
                execution_fn=lambda data, **kw: AnalysisAgent().run({"data": data, "request": kw.get("request", "summary")}),
                validation_requirements=["summary is not empty"],
            )
        )

        # 4. Aggregation & Grouping
        self.register(
            ToolDefinition(
                name="aggregation",
                description="Calculates group-by aggregations, sums, means, rankings, and metric breakdowns.",
                capabilities=["aggregation", "group_by", "ranking", "regional_analysis", "segmentation"],
                input_schema={"data": "pd.DataFrame", "metric": "str", "dimension": "str", "agg_func": "str"},
                output_schema={"aggregated_data": "dict", "ranking": "list"},
                execution_fn=lambda data, **kw: AnalysisAgent().run({"data": data, "request": "summary", **kw}),
                validation_requirements=["aggregated_data is not empty"],
            )
        )

        # 5. Correlation Analysis
        self.register(
            ToolDefinition(
                name="correlation_analysis",
                description="Computes Pearson and Spearman correlation matrices across numeric features.",
                capabilities=["correlation_analysis", "feature_relationships", "relationship_analysis"],
                input_schema={"data": "pd.DataFrame"},
                output_schema={"correlation_matrix": "dict"},
                execution_fn=lambda data, **kw: AnalysisAgent().run({"data": data, "request": "correlation", **kw}),
                validation_requirements=["correlation_matrix is not empty"],
            )
        )

        # 6. Anomaly Detection
        self.register(
            ToolDefinition(
                name="anomaly_detection",
                description="Detects statistical outliers and abnormal spikes using IQR and Isolation Forest.",
                capabilities=["anomaly_detection", "outliers", "spike_detection"],
                input_schema={"data": "pd.DataFrame", "column": "str"},
                output_schema={"anomalies": "list", "anomaly_count": "int"},
                execution_fn=lambda data, **kw: InsightAgent().run({"data": data, "type": "anomalies", **kw}),
                validation_requirements=["anomaly_count >= 0"],
            )
        )

        # 7. Root-Cause Explanation & Driver Extraction
        self.register(
            ToolDefinition(
                name="explanation",
                description="Identifies top influential driver features explaining changes or target metrics.",
                capabilities=["root_cause_analysis", "explanation", "feature_drivers", "trend_analysis"],
                input_schema={"data": "pd.DataFrame", "target": "str", "top_k": "int"},
                output_schema={"drivers": "list", "explanations": "list"},
                execution_fn=lambda data, **kw: InsightAgent().run({"data": data, "type": "smart", **kw}),
                validation_requirements=["drivers is not empty"],
            )
        )

        # 8. Time Series Forecasting
        self.register(
            ToolDefinition(
                name="forecasting",
                description="Generates future multi-step forecasts with confidence intervals using ARIMA/ETS.",
                capabilities=["forecasting", "time_series", "future_projection"],
                input_schema={"data": "pd.DataFrame", "target": "str", "periods": "int"},
                output_schema={"forecast": "list", "metrics": "dict"},
                execution_fn=lambda data, **kw: ForecastAgent().run({"data": data, **kw}),
                validation_requirements=["forecast length matches periods"],
            )
        )

        # 9. Predictive Modeling (Regression & Classification)
        self.register(
            ToolDefinition(
                name="prediction",
                description="Trains and validates supervised machine learning models (Random Forest, Gradient Boosting).",
                capabilities=["prediction", "classification", "regression", "model_training", "feature_engineering"],
                input_schema={"data": "pd.DataFrame", "target": "str", "features": "list"},
                output_schema={"model_metrics": "dict", "predictions": "list"},
                execution_fn=lambda data, **kw: PredictionAgent().run({"data": data, **kw}),
                validation_requirements=["model_metrics has score >= 0"],
            )
        )

        # 10. Data Visualization
        self.register(
            ToolDefinition(
                name="visualization",
                description="Generates interactive charts (bar, line, scatter, histogram, pie).",
                capabilities=["visualization", "chart", "plotting"],
                input_schema={"data": "pd.DataFrame", "chart_type": "str", "x": "str", "y": "str"},
                output_schema={"chart_spec": "dict", "chart_type": "str"},
                execution_fn=lambda data, **kw: VisualizationAgent().run({"data": data, **kw}),
                validation_requirements=["chart_spec is valid JSON"],
            )
        )

        # 11. Narrative & Executive Reporting
        self.register(
            ToolDefinition(
                name="reporting",
                description="Synthesizes all pipeline outputs into a structured executive markdown report.",
                capabilities=["reporting", "synthesis", "executive_summary"],
                input_schema={"agent_outputs": "list"},
                output_schema={"report_markdown": "str", "sections": "list"},
                execution_fn=lambda **kw: ReportAgent().run({"request": "pipeline", **kw}),
                validation_requirements=["report_markdown is not empty"],
            )
        )


# Global default singleton
DEFAULT_TOOL_REGISTRY = ToolRegistry()
