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

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Alias for get_tool."""
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
            ANNAgent,
            AnomalyDetectionAgent,
            AutonomousAnalystAgent,
            AutonomousForecasterAgent,
            CleaningAgent,
            ClusteringAgent,
            CNNAgent,
            ConversationalAnalystAgent,
            DataQualityAgent,
            DataValidationAgent,
            EDAAgent,
            ForecastAgent,
            HypothesisTestingAgent,
            InsightAgent,
            ModelMonitorAgent,
            ModelOrchestratorAgent,
            ModelRegistryAgent,
            ModelSelectionAgent,
            ModelTrainingAgent,
            PredictionAgent,
            RecommendationAgent,
            ReportAgent,
            StatisticalAnalysisAgent,
            TransformationAgent,
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

        # 3. Exploratory Data Analysis, Profiling & Data Quality
        self.register(
            ToolDefinition(
                name="eda",
                description="Comprehensive dataset profiling, schema inference, distributions, missing values, duplicates, and data quality intelligence.",
                capabilities=["eda", "statistics", "descriptive_stats", "summary", "data_profiling", "data_quality", "missing_analysis", "profile"],
                input_schema={"data": "pd.DataFrame", "columns": "list"},
                output_schema={"summary": "dict", "statistics": "dict", "data_quality": "dict", "findings": "list"},
                execution_fn=lambda data, **kw: EDAAgent().run({"data": data, **kw}),
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

        # 5. Statistical Relationship & Correlation Analysis
        self.register(
            ToolDefinition(
                name="statistical_analysis",
                description="Discovers, measures, tests, and ranks bivariate and multivariate relationships, associations, and correlations.",
                capabilities=["statistical_analysis", "correlation_analysis", "feature_relationships", "relationship_analysis", "dependency_analysis"],
                input_schema={"data": "pd.DataFrame", "features": "list", "target": "str"},
                output_schema={"relationships": "list", "ranked_relationships": "list", "correlation_matrix": "dict"},
                execution_fn=lambda data, **kw: StatisticalAnalysisAgent().run({"data": data, **kw}),
                validation_requirements=["relationships is not empty"],
            )
        )
        self.register(
            ToolDefinition(
                name="correlation_analysis",
                description="Computes Pearson and Spearman correlation matrices and feature dependencies.",
                capabilities=["correlation_analysis", "feature_relationships", "relationship_analysis"],
                input_schema={"data": "pd.DataFrame"},
                output_schema={"correlation_matrix": "dict"},
                execution_fn=lambda data, **kw: StatisticalAnalysisAgent().run({"data": data, **kw}),
                validation_requirements=["correlation_matrix is not empty"],
            )
        )

        # 5b. Hypothesis Testing & Statistical Significance
        self.register(
            ToolDefinition(
                name="hypothesis_testing",
                description="Performs data-driven statistical hypothesis tests (t-tests, ANOVA, Mann-Whitney, Kruskal-Wallis, Chi-Square, Fisher's Exact) with effect sizes and FDR corrections.",
                capabilities=["hypothesis_testing", "statistical_significance", "significance_testing", "t_test", "anova", "group_comparison", "diff_testing"],
                input_schema={"data": "pd.DataFrame", "feature": "str", "group": "str", "alpha": "float"},
                output_schema={"hypotheses": "list", "findings": "list", "summary": "dict"},
                execution_fn=lambda data, **kw: HypothesisTestingAgent().run({"data": data, **kw}),
                validation_requirements=["hypotheses is not empty"],
            )
        )

        # 5c. Data Transformation & Feature Engineering
        self.register(
            ToolDefinition(
                name="transformation",
                description="Transforms tabular data into model-ready numerical feature matrices with imputation, encoding, scaling, and feature engineering.",
                capabilities=["transformation", "feature_engineering", "preprocessing", "data_cleaning", "encoding", "scaling", "imputation"],
                input_schema={"data": "pd.DataFrame", "target": "str", "features": "list", "config": "dict"},
                output_schema={"transformation_plan": "dict", "state": "dict", "summary": "dict"},
                execution_fn=lambda data, **kw: TransformationAgent().run({"data": data, **kw}),
                validation_requirements=["transformation_plan is not empty"],
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
                execution_fn=lambda data, **kw: AnomalyDetectionAgent().run({"data": data, **kw}),
                validation_requirements=["anomaly_count >= 0"],
            )
        )

        # 6b. Clustering & Segmentation (Milestone 6, Task 2)
        self.register(
            ToolDefinition(
                name="clustering",
                description="Discovers natural clusters and customer segments using benchmarked unsupervised algorithms.",
                capabilities=["clustering", "segmentation", "group_discovery", "natural_groups"],
                input_schema={"data": "pd.DataFrame", "features": "list", "n_clusters": "int"},
                output_schema={"cluster_count": "int", "labels": "list", "cluster_sizes": "dict"},
                execution_fn=lambda data, **kw: ClusteringAgent().run({"data": data, **kw}),
                validation_requirements=["cluster_count >= 2"],
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

        # 9. Predictive Modeling (Legacy Baseline)
        self.register(
            ToolDefinition(
                name="prediction",
                description="Trains and validates supervised machine learning models (Random Forest, Gradient Boosting).",
                capabilities=["prediction", "classification", "regression", "feature_engineering"],
                input_schema={"data": "pd.DataFrame", "target": "str", "features": "list"},
                output_schema={"model_metrics": "dict", "predictions": "list"},
                execution_fn=lambda data, **kw: PredictionAgent().run({"data": data, **kw}),
                validation_requirements=["model_metrics has score >= 0"],
            )
        )

        # 10. Advanced Model Training & Evaluation Engine (Milestone 3)
        self.register(
            ToolDefinition(
                name="model_training",
                description="Trains, cross-validates, evaluates, and registers multi-algorithm candidate models without data leakage.",
                capabilities=["model_training", "model_evaluation", "model_benchmarking", "algorithm_comparison"],
                input_schema={"data": "pd.DataFrame", "target": "str", "candidates": "list", "metric": "str"},
                output_schema={"best_model": "dict", "ranking": "list", "candidates": "list"},
                execution_fn=lambda data, **kw: ModelTrainingAgent().run({"data": data, **kw}),
                validation_requirements=["best_model is not None"],
            )
        )

        # 11. Artificial Neural Network (ANN) Engine (Milestone 3, Task 3)
        self.register(
            ToolDefinition(
                name="ann_trainer",
                description="Trains, tunes, and evaluates deep Multi-Layer Perceptrons on tabular data with loss curve tracking and early stopping.",
                capabilities=["ann_training", "tabular_regression", "tabular_binary_classification", "tabular_multiclass_classification", "deep_learning"],
                input_schema={"data": "pd.DataFrame", "target": "str", "epochs": "int", "layers": "list"},
                output_schema={"best_model": "dict", "metrics": "dict", "loss_curve": "list"},
                execution_fn=lambda data, **kw: ANNAgent().run({"data": data, **kw}),
                validation_requirements=["metrics is not empty"],
            )
        )

        # 12. Convolutional Neural Network (CNN) Engine (Milestone 3, Task 4)
        self.register(
            ToolDefinition(
                name="cnn_trainer",
                description="Trains, tunes, and evaluates Convolutional Neural Networks on image, spatial grid, and signal spectrogram datasets.",
                capabilities=["cnn_training", "image_classification", "spatial_modeling", "computer_vision"],
                input_schema={"data": "Any", "target": "str", "epochs": "int", "spatial_shape": "tuple"},
                output_schema={"best_model": "dict", "metrics": "dict", "spatial_gain": "float"},
                execution_fn=lambda data, **kw: CNNAgent().run({"data": data, **kw}),
                validation_requirements=["metrics is not empty"],
            )
        )

        # 13. Unified Intelligent Model Orchestrator (Milestone 4, Task 1)
        self.register(
            ToolDefinition(
                name="model_orchestrator",
                description="Coordinates end-to-end model selection, capability validation, parallel/sequential cross-validation, and winner registration across Traditional ML, ANN, and CNN.",
                capabilities=["model_orchestration", "multi_model_training", "model_comparison", "prediction_routing", "model_registry_deployment"],
                input_schema={"data": "pd.DataFrame", "target": "str", "task_type": "str", "modality": "str", "candidates": "list"},
                output_schema={"best_model": "dict", "ranking": "list", "selection_reason": "str"},
                execution_fn=lambda data, **kw: ModelOrchestratorAgent().run({"data": data, **kw}),
                validation_requirements=["best_model is not empty"],
            )
        )

        # 14. Model Monitoring & Drift Detection (Milestone 4, Task 2)
        self.register(
            ToolDefinition(
                name="model_monitor",
                description="Performs statistical data drift detection (KS test, Chi-square, PSI), schema drift analysis, prediction shift monitoring, and performance degradation tracking.",
                capabilities=["data_drift_detection", "schema_drift_detection", "prediction_drift_detection", "performance_monitoring", "data_quality_monitoring", "model_monitoring"],
                input_schema={"model_id": "str", "current_data": "Any", "reference_data": "Any", "thresholds": "dict"},
                output_schema={"overall_severity": "str", "data_drift": "dict", "performance_drift": "dict", "recommendations": "list"},
                execution_fn=lambda **kw: ModelMonitorAgent().run(kw),
                validation_requirements=["overall_severity is not empty"],
            )
        )

        # 15. Autonomous Data Analysis & Insight Generation (Milestone 5, Task 1)
        self.register(
            ToolDefinition(
                name="autonomous_analyst",
                description="Performs autonomous exploratory data analysis, pattern discovery, trend tracking, segmentation, correlation, anomaly detection, concentration analysis, and ranked evidence-based insight generation.",
                capabilities=["autonomous_analysis", "insight_generation", "pattern_discovery", "trend_analysis", "segmentation", "correlation_analysis", "anomaly_detection", "concentration_analysis"],
                input_schema={"data": "pd.DataFrame", "user_intent": "UserIntent", "analysis_depth": "str"},
                output_schema={"summary": "str", "insights": "list", "key_metrics": "dict"},
                execution_fn=lambda data, **kw: AutonomousAnalystAgent().run({"data": data, **kw}),
                validation_requirements=["insights is not empty or summary is not empty"],
            )
        )

        # 16. Conversational Analyst & Multi-Turn Context (Milestone 5, Task 2)
        self.register(
            ToolDefinition(
                name="conversational_analyst",
                description="Executes multi-turn conversational data intelligence, anaphora/pronoun resolution, evidence-grounded responses, and structured report synthesis.",
                capabilities=["conversational_analysis", "context_resolution", "follow_up_analysis", "multi_turn_chat", "conversational_reporting"],
                input_schema={"command": "str", "session_id": "str", "data": "Any"},
                output_schema={"response": "str", "resolved_command": "str", "intent": "str"},
                execution_fn=lambda **kw: ConversationalAnalystAgent().run(kw),
                validation_requirements=["response is not empty"],
            )
        )

        # 17. Autonomous Forecasting Engine (Milestone 5, Task 3)
        self.register(
            ToolDefinition(
                name="forecast_engine",
                description="Autonomous time-series forecasting, chronological candidate benchmarking (Naive, Moving Average, Exponential Smoothing, ML), and prediction intervals.",
                capabilities=["forecasting", "time_series_forecasting", "forecast_validation", "uncertainty_estimation"],
                input_schema={"data": "pd.DataFrame", "target_column": "str", "time_column": "str", "horizon": "int"},
                output_schema={"predictions": "list", "model_name": "str", "validation_metrics": "dict"},
                execution_fn=lambda data, **kw: AutonomousForecasterAgent().run({"data": data, "mode": "forecast", **kw}),
                validation_requirements=["status != 'FAILED'"],
            )
        )

        # 18. What-If Scenario Engine (Milestone 5, Task 3)
        self.register(
            ToolDefinition(
                name="scenario_engine",
                description="Deterministic counterfactual What-If simulations, segment elasticity shocks, and optimistic/expected/pessimistic multi-scenario comparisons.",
                capabilities=["what_if_analysis", "scenario_comparison", "counterfactual_simulation", "sensitivity_analysis"],
                input_schema={"data": "pd.DataFrame", "target": "str", "changed_variables": "dict"},
                output_schema={"scenario_value": "float", "percentage_difference": "float"},
                execution_fn=lambda data, **kw: AutonomousForecasterAgent().run({"data": data, "mode": "scenario", **kw}),
                validation_requirements=["status != 'FAILED'"],
            )
        )

        # 19. Autonomous Decision & Recommendation Engine (Milestone 5, Task 4)
        self.register(
            ToolDefinition(
                name="decision_engine",
                description="Evidence-backed decision support: ranked recommendations, risk assessment, opportunity detection, action prioritization, expected impact, and audit trail.",
                capabilities=["decision_support", "recommendation_generation", "risk_assessment",
                              "opportunity_detection", "action_prioritization", "recommendation_engine"],
                input_schema={"data": "pd.DataFrame", "insights": "list", "forecasts": "list",
                              "scenarios": "list", "monitoring_results": "list",
                              "business_constraints": "list", "objective": "Any",
                              "user_intent": "str", "max_recommendations": "int"},
                output_schema={"status": "str", "executive_summary": "str",
                               "recommendations": "list", "risks": "list", "audit_trail": "list"},
                execution_fn=lambda **kw: RecommendationAgent().run(kw),
                validation_requirements=["status != 'failed'"],
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
