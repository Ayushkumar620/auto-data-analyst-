"""End-to-End Evaluation & Golden Benchmarks Runner for the Auto Data Analyst Agent System.

Validates the full multi-agent autonomous workflow across 5 standard analytical modalities:
1. Tabular Regression
2. Tabular Classification
3. Time Series Forecasting
4. Spatial / Image Pattern Recognition
5. Data Safety, Leakage & Integrity Auditing
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from agent.schemas import AgentResult, AgentStatus, ClaimType
from agent.dynamic_planner import DynamicTaskPlanner
from agent.intent import AnalyticalIntent, IntentAnalyzer
from backend.app.core.semantic import SemanticSchemaAgent
from backend.app.core.evidence_insights import EvidenceBasedInsightsEngine
from backend.app.ml.model_selection import MLModelComparisonEngine
from backend.app.ml.ann_engine import ANNEngine
from backend.app.ml.cnn_engine import CNNEngine
from backend.app.ml.validation_engine import DataModelValidator


class BenchmarkType(str, Enum):
    TABULAR_REGRESSION = "tabular_regression"
    TABULAR_CLASSIFICATION = "tabular_classification"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    SPATIAL_IMAGE_CLASSIFICATION = "spatial_image_classification"
    DATA_INTEGRITY_LEAKAGE = "data_integrity_leakage"


@dataclass
class BenchmarkResult:
    """Outcome of a single golden benchmark test case."""
    name: str
    benchmark_type: BenchmarkType
    passed: bool
    duration_ms: float
    intent_detected: str
    primary_metric: Dict[str, Any]
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "benchmark_type": self.benchmark_type.value,
            "passed": self.passed,
            "duration_ms": round(float(self.duration_ms), 2),
            "intent_detected": self.intent_detected,
            "primary_metric": self.primary_metric,
            "error_message": self.error_message,
        }


@dataclass
class EvaluationSummaryReport:
    """Summary report across all golden benchmark suites."""
    total_benchmarks: int
    passed_benchmarks: int
    failed_benchmarks: int
    pass_rate_pct: float
    total_duration_ms: float
    results: List[BenchmarkResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_benchmarks": self.total_benchmarks,
            "passed_benchmarks": self.passed_benchmarks,
            "failed_benchmarks": self.failed_benchmarks,
            "pass_rate_pct": round(float(self.pass_rate_pct), 2),
            "total_duration_ms": round(float(self.total_duration_ms), 2),
            "results": [r.to_dict() for r in self.results],
        }


class BenchmarkDatasetFactory:
    """Generates standardized synthetic datasets for reproducible benchmarking."""

    @staticmethod
    def create_regression_dataset(n: int = 120, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        sqft = np.random.uniform(800, 4500, n)
        bedrooms = np.random.choice([1, 2, 3, 4, 5], n)
        dist_downtown = np.random.uniform(1.0, 30.0, n)
        # Price formula with non-linear interaction
        price = 50000 + sqft * 180 + bedrooms * 15000 - dist_downtown * 1200 + np.random.normal(0, 5000, n)
        return pd.DataFrame({
            "sqft": sqft,
            "bedrooms": bedrooms,
            "distance_downtown": dist_downtown,
            "sale_price": price,
        })

    @staticmethod
    def create_classification_dataset(n: int = 120, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        usage = np.random.uniform(5, 100, n)
        support_calls = np.random.poisson(2, n)
        tenure_months = np.random.uniform(1, 48, n)
        # Churn probability
        logit = -2.0 + support_calls * 0.8 - usage * 0.03 - tenure_months * 0.05
        prob = 1 / (1 + np.exp(-logit))
        churn = (np.random.uniform(0, 1, n) < prob).astype(int)
        return pd.DataFrame({
            "monthly_usage_hours": usage,
            "support_calls": support_calls,
            "tenure_months": tenure_months,
            "churned": churn,
        })

    @staticmethod
    def create_timeseries_dataset(n: int = 60, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        trend = np.linspace(100, 300, n)
        seasonality = 30 * np.sin(2 * np.pi * np.arange(n) / 12)
        sales = trend + seasonality + np.random.normal(0, 5, n)
        return pd.DataFrame({"date": dates, "monthly_sales": sales})

    @staticmethod
    def create_spatial_dataset(n: int = 80, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        # 16 features representing 4x4 spatial patch
        data_dict = {}
        for i in range(16):
            data_dict[f"pixel_{i}"] = np.random.uniform(0, 255, n)
        # Class label depends on central 2x2 pixels
        central_avg = (data_dict["pixel_5"] + data_dict["pixel_6"] + data_dict["pixel_9"] + data_dict["pixel_10"]) / 4.0
        data_dict["pattern_class"] = (central_avg > 128.0).astype(int)
        return pd.DataFrame(data_dict)

    @staticmethod
    def create_leaking_dataset(n: int = 60, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        target = np.random.uniform(10, 100, n)
        return pd.DataFrame({
            "clean_x": np.random.normal(0, 1, n),
            "leaking_proxy": target * 1.00005,  # Data leakage
            "target": target,
        })


class BenchmarkRunner:
    """Executes end-to-end golden benchmarks across all analytical engines."""

    def __init__(self):
        self.semantic_agent = SemanticSchemaAgent()
        self.intent_analyzer = IntentAnalyzer()
        self.planner = DynamicTaskPlanner()
        self.validator = DataModelValidator()
        self.insights_engine = EvidenceBasedInsightsEngine()

    def run_tabular_regression_benchmark(self) -> BenchmarkResult:
        """Benchmark 1: Tabular Regression End-to-End."""
        start_t = time.time()
        df = BenchmarkDatasetFactory.create_regression_dataset()
        query = "Train a machine learning model to predict sale_price"

        try:
            analysis = self.intent_analyzer.analyze(query, df)
            plan = self.planner.create_plan(query, df)
            exec_res = self.planner.execute_plan(plan, df)

            # Assert model selection trained models and achieved high R^2
            ml_engine = MLModelComparisonEngine()
            comp_report = ml_engine.benchmark_models(df, target_column="sale_price")

            passed = (
                analysis.primary_intent == AnalyticalIntent.PREDICTION
                and comp_report.best_model.primary_metric_value >= 0.85
                and exec_res.get("status") == "completed"
            )

            return BenchmarkResult(
                name="Tabular Regression (Housing Prices)",
                benchmark_type=BenchmarkType.TABULAR_REGRESSION,
                passed=passed,
                duration_ms=(time.time() - start_t) * 1000,
                intent_detected=analysis.primary_intent.value,
                primary_metric={"r2_score": comp_report.best_model.primary_metric_value, "best_model": comp_report.best_model.model_name},
            )
        except Exception as e:
            return BenchmarkResult(
                name="Tabular Regression (Housing Prices)",
                benchmark_type=BenchmarkType.TABULAR_REGRESSION,
                passed=False,
                duration_ms=(time.time() - start_t) * 1000,
                intent_detected="error",
                primary_metric={},
                error_message=str(e),
            )

    def run_tabular_classification_benchmark(self) -> BenchmarkResult:
        """Benchmark 2: Tabular Classification End-to-End."""
        start_t = time.time()
        df = BenchmarkDatasetFactory.create_classification_dataset()
        query = "Predict customer churn probability and identify key drivers"

        try:
            analysis = self.intent_analyzer.analyze(query, df)
            ml_engine = MLModelComparisonEngine()
            comp_report = ml_engine.benchmark_models(df, target_column="churned")

            passed = (
                comp_report.problem_type.value == "binary_classification"
                and comp_report.best_model.primary_metric_value >= 0.70
                and len(comp_report.candidate_evaluations) >= 3
            )

            return BenchmarkResult(
                name="Tabular Classification (Customer Churn)",
                benchmark_type=BenchmarkType.TABULAR_CLASSIFICATION,
                passed=passed,
                duration_ms=(time.time() - start_t) * 1000,
                intent_detected=analysis.primary_intent.value,
                primary_metric={"f1_score": comp_report.best_model.primary_metric_value, "best_model": comp_report.best_model.model_name},
            )
        except Exception as e:
            return BenchmarkResult(
                name="Tabular Classification (Customer Churn)",
                benchmark_type=BenchmarkType.TABULAR_CLASSIFICATION,
                passed=False,
                duration_ms=(time.time() - start_t) * 1000,
                intent_detected="error",
                primary_metric={},
                error_message=str(e),
            )

    def run_timeseries_forecasting_benchmark(self) -> BenchmarkResult:
        """Benchmark 3: Time Series Forecasting End-to-End."""
        start_t = time.time()
        df = BenchmarkDatasetFactory.create_timeseries_dataset()
        query = "Forecast monthly_sales for the next 6 months"

        try:
            analysis = self.intent_analyzer.analyze(query, df)
            plan = self.planner.create_plan(query, df)
            exec_res = self.planner.execute_plan(plan, df)

            passed = (
                analysis.primary_intent == AnalyticalIntent.FORECASTING
                and exec_res.get("status") == "completed"
            )

            return BenchmarkResult(
                name="Time Series Forecasting (Monthly Sales)",
                benchmark_type=BenchmarkType.TIME_SERIES_FORECASTING,
                passed=passed,
                duration_ms=(time.time() - start_t) * 1000,
                intent_detected=analysis.primary_intent.value,
                primary_metric={"periods": 6, "status": exec_res.get("status")},
            )
        except Exception as e:
            return BenchmarkResult(
                name="Time Series Forecasting (Monthly Sales)",
                benchmark_type=BenchmarkType.TIME_SERIES_FORECASTING,
                passed=False,
                duration_ms=(time.time() - start_t) * 1000,
                intent_detected="error",
                primary_metric={},
                error_message=str(e),
            )

    def run_spatial_cnn_benchmark(self) -> BenchmarkResult:
        """Benchmark 4: Spatial / Image CNN End-to-End."""
        start_t = time.time()
        df = BenchmarkDatasetFactory.create_spatial_dataset()

        try:
            cnn = CNNEngine()
            res = cnn.train_and_evaluate(
                data=df,
                target="pattern_class",
                spatial_shape=(4, 4),
            )

            passed = (
                res.metrics.get("accuracy", 0.0) >= 0.70
                and len(res.loss_curve) > 0
                and res.comparison_with_flat_baseline is not None
            )

            return BenchmarkResult(
                name="Spatial / Image Pattern (4x4 Grid CNN)",
                benchmark_type=BenchmarkType.SPATIAL_IMAGE_CLASSIFICATION,
                passed=passed,
                duration_ms=(time.time() - start_t) * 1000,
                intent_detected="cnn_spatial",
                primary_metric={"accuracy": res.metrics.get("accuracy"), "epochs": len(res.loss_curve)},
            )
        except Exception as e:
            return BenchmarkResult(
                name="Spatial / Image Pattern (4x4 Grid CNN)",
                benchmark_type=BenchmarkType.SPATIAL_IMAGE_CLASSIFICATION,
                passed=False,
                duration_ms=(time.time() - start_t) * 1000,
                intent_detected="error",
                primary_metric={},
                error_message=str(e),
            )

    def run_data_integrity_leakage_benchmark(self) -> BenchmarkResult:
        """Benchmark 5: Data Safety & Leakage Detection End-to-End."""
        start_t = time.time()
        df = BenchmarkDatasetFactory.create_leaking_dataset()

        try:
            issues, diag = self.validator.check_data_leakage(df, target_column="target")
            passed = (
                len(issues) >= 1
                and "leaking_proxy" in diag.get("leaking_features", [])
            )

            return BenchmarkResult(
                name="Data Integrity & Target Leakage Audit",
                benchmark_type=BenchmarkType.DATA_INTEGRITY_LEAKAGE,
                passed=passed,
                duration_ms=(time.time() - start_t) * 1000,
                intent_detected="validation_audit",
                primary_metric={"leaking_features_detected": diag.get("leaking_features", [])},
            )
        except Exception as e:
            return BenchmarkResult(
                name="Data Integrity & Target Leakage Audit",
                benchmark_type=BenchmarkType.DATA_INTEGRITY_LEAKAGE,
                passed=False,
                duration_ms=(time.time() - start_t) * 1000,
                intent_detected="error",
                primary_metric={},
                error_message=str(e),
            )

    def run_all_benchmarks(self) -> EvaluationSummaryReport:
        """Execute all golden benchmark test suites and return a summary report."""
        start_all = time.time()
        results = [
            self.run_tabular_regression_benchmark(),
            self.run_tabular_classification_benchmark(),
            self.run_timeseries_forecasting_benchmark(),
            self.run_spatial_cnn_benchmark(),
            self.run_data_integrity_leakage_benchmark(),
        ]

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        pass_rate = (passed / total) * 100 if total > 0 else 0.0

        return EvaluationSummaryReport(
            total_benchmarks=total,
            passed_benchmarks=passed,
            failed_benchmarks=failed,
            pass_rate_pct=pass_rate,
            total_duration_ms=(time.time() - start_all) * 1000,
            results=results,
        )
