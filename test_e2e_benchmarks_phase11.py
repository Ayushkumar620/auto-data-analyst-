"""Comprehensive test suite for Phase 11: End-to-End Evaluation & Golden Benchmarks."""
import pytest
from fastapi.testclient import TestClient

from backend.app.evaluation.benchmark_runner import (
    BenchmarkDatasetFactory,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkType,
    EvaluationSummaryReport,
)
from backend.app.main import app


# ==============================================================================
# 1. Benchmark Dataset Generation Tests
# ==============================================================================

def test_benchmark_dataset_factory():
    """Verify synthetic dataset generation for all analytical modalities."""
    df_reg = BenchmarkDatasetFactory.create_regression_dataset(n=50)
    assert len(df_reg) == 50
    assert "sale_price" in df_reg.columns

    df_clf = BenchmarkDatasetFactory.create_classification_dataset(n=50)
    assert len(df_clf) == 50
    assert "churned" in df_clf.columns

    df_ts = BenchmarkDatasetFactory.create_timeseries_dataset(n=30)
    assert len(df_ts) == 30
    assert "monthly_sales" in df_ts.columns

    df_sp = BenchmarkDatasetFactory.create_spatial_dataset(n=40)
    assert len(df_sp) == 40
    assert "pattern_class" in df_sp.columns
    assert "pixel_0" in df_sp.columns

    df_leak = BenchmarkDatasetFactory.create_leaking_dataset(n=40)
    assert "leaking_proxy" in df_leak.columns


# ==============================================================================
# 2. Individual Golden Benchmark Scenario Tests
# ==============================================================================

def test_tabular_regression_benchmark():
    """Verify end-to-end regression benchmark execution."""
    runner = BenchmarkRunner()
    res = runner.run_tabular_regression_benchmark()

    assert isinstance(res, BenchmarkResult)
    assert res.benchmark_type == BenchmarkType.TABULAR_REGRESSION
    assert res.passed is True
    assert res.duration_ms > 0
    assert res.primary_metric.get("r2_score", 0.0) >= 0.85


def test_tabular_classification_benchmark():
    """Verify end-to-end classification benchmark execution."""
    runner = BenchmarkRunner()
    res = runner.run_tabular_classification_benchmark()

    assert isinstance(res, BenchmarkResult)
    assert res.benchmark_type == BenchmarkType.TABULAR_CLASSIFICATION
    assert res.passed is True
    assert res.primary_metric.get("f1_score", 0.0) >= 0.70


def test_timeseries_forecasting_benchmark():
    """Verify end-to-end time-series forecasting benchmark execution."""
    runner = BenchmarkRunner()
    res = runner.run_timeseries_forecasting_benchmark()

    assert isinstance(res, BenchmarkResult)
    assert res.benchmark_type == BenchmarkType.TIME_SERIES_FORECASTING
    assert res.passed is True
    assert res.primary_metric.get("status") == "completed"


def test_spatial_cnn_benchmark():
    """Verify end-to-end spatial / image CNN benchmark execution."""
    runner = BenchmarkRunner()
    res = runner.run_spatial_cnn_benchmark()

    assert isinstance(res, BenchmarkResult)
    assert res.benchmark_type == BenchmarkType.SPATIAL_IMAGE_CLASSIFICATION
    assert res.passed is True
    assert res.primary_metric.get("accuracy", 0.0) >= 0.70


def test_data_integrity_leakage_benchmark():
    """Verify end-to-end data integrity & leakage detection benchmark execution."""
    runner = BenchmarkRunner()
    res = runner.run_data_integrity_leakage_benchmark()

    assert isinstance(res, BenchmarkResult)
    assert res.benchmark_type == BenchmarkType.DATA_INTEGRITY_LEAKAGE
    assert res.passed is True
    assert "leaking_proxy" in res.primary_metric.get("leaking_features_detected", [])


# ==============================================================================
# 3. Full Benchmark Suite & Pass Rate Verification
# ==============================================================================

def test_run_all_benchmarks_full_suite():
    """Verify entire golden benchmark suite runs with 100% pass rate."""
    runner = BenchmarkRunner()
    report = runner.run_all_benchmarks()

    assert isinstance(report, EvaluationSummaryReport)
    assert report.total_benchmarks == 5
    assert report.passed_benchmarks == 5
    assert report.failed_benchmarks == 0
    assert report.pass_rate_pct == 100.0


# ==============================================================================
# 4. FastAPI Endpoints Integration
# ==============================================================================

def test_fastapi_evaluation_endpoints():
    """Verify FastAPI GET /api/v1/evaluation/benchmarks and POST /api/v1/evaluation/run."""
    client = TestClient(app)

    # 1. List benchmarks
    res_list = client.get("/api/v1/evaluation/benchmarks")
    assert res_list.status_code == 200
    benchmarks = res_list.json().get("benchmarks", [])
    assert len(benchmarks) == 5

    # 2. Run evaluation suite
    res_run = client.post("/api/v1/evaluation/run")
    assert res_run.status_code == 200
    report_data = res_run.json()
    assert report_data["total_benchmarks"] == 5
    assert report_data["pass_rate_pct"] == 100.0
