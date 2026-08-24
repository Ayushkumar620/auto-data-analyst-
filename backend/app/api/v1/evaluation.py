"""FastAPI Endpoints for Golden Benchmarks & End-to-End Evaluation."""
from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter

from backend.app.evaluation.benchmark_runner import BenchmarkRunner, EvaluationSummaryReport

router = APIRouter(prefix="/evaluation", tags=["Evaluation & Benchmarks"])
benchmark_runner = BenchmarkRunner()


@router.get("/benchmarks")
async def list_benchmarks() -> Dict[str, Any]:
    """List all available golden benchmark test suites."""
    return {
        "benchmarks": [
            {"id": "tabular_regression", "name": "Tabular Regression (Housing Prices)", "modality": "tabular"},
            {"id": "tabular_classification", "name": "Tabular Classification (Customer Churn)", "modality": "tabular"},
            {"id": "time_series_forecasting", "name": "Time Series Forecasting (Monthly Sales)", "modality": "timeseries"},
            {"id": "spatial_image_classification", "name": "Spatial / Image Pattern (4x4 Grid CNN)", "modality": "spatial"},
            {"id": "data_integrity_leakage", "name": "Data Integrity & Target Leakage Audit", "modality": "validation"},
        ]
    }


@router.post("/run")
async def run_evaluation_suite() -> Dict[str, Any]:
    """Run all end-to-end golden benchmarks and return evaluation report."""
    report: EvaluationSummaryReport = benchmark_runner.run_all_benchmarks()
    return report.to_dict()
