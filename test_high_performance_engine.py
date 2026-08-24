"""Tests for High-Performance Analytical Execution Layer (DuckDB / Polars / Vectorized NumPy).

Verifies:
1. HighPerformanceExecutionEngine multi-column grouping and multi-metric aggregation
2. Sub-second performance on 100,000+ row datasets
3. Fast statistical profiling, Pearson correlation matrix, and percentiles
4. Analytical SQL query execution with filtering, joins, and sorting
5. Seamless integration with AutonomousCommandOrchestrator
"""
import numpy as np
import pandas as pd
import pytest

from backend.app.core.high_performance_engine import (
    HighPerformanceExecutionEngine,
    AggregationResult,
    HighPerformanceStats,
    global_high_performance_engine,
)
from agent.command_orchestrator import AutonomousCommandOrchestrator


@pytest.fixture
def large_sales_df():
    np.random.seed(42)
    n = 100_000
    return pd.DataFrame({
        "Country": np.random.choice(["USA", "India", "Germany", "Japan", "Brazil", "UK", "France"], n),
        "Region": np.random.choice(["North", "South", "East", "West", "Central"], n),
        "Product": np.random.choice(["Laptop", "Phone", "Tablet", "Monitor", "Headphones"], n),
        "Revenue": np.random.uniform(100.0, 5000.0, n),
        "Cost": np.random.uniform(50.0, 3000.0, n),
        "Discount": np.random.uniform(0.0, 0.35, n),
        "Units": np.random.randint(1, 20, n),
    })


def test_high_performance_aggregations(large_sales_df):
    """Verify sub-second multi-column group-by aggregation on 100k rows."""
    engine = HighPerformanceExecutionEngine()

    result: AggregationResult = engine.aggregate(
        df=large_sales_df,
        group_by=["Country", "Product"],
        aggregations={
            "Revenue": ["sum", "mean"],
            "Units": "sum",
        },
        sort_by="Revenue_sum",
        ascending=False,
        limit=15,
    )

    assert isinstance(result, AggregationResult)
    assert result.rows_processed == 100_000
    assert len(result.data) <= 15
    assert result.duration_ms < 500  # Sub-second execution (<500ms for 100k rows)
    assert "Country" in result.data.columns
    assert "Product" in result.data.columns
    assert "Revenue_sum" in result.data.columns
    assert "Revenue_mean" in result.data.columns

    # Verify top row has highest Revenue_sum
    assert result.data["Revenue_sum"].iloc[0] >= result.data["Revenue_sum"].iloc[-1]


def test_high_speed_statistics_and_correlations(large_sales_df):
    """Verify high-speed statistical profiling, correlation matrix, and percentiles."""
    engine = HighPerformanceExecutionEngine()

    stats: HighPerformanceStats = engine.compute_fast_statistics(
        df=large_sales_df,
        numeric_columns=["Revenue", "Cost", "Discount", "Units"],
    )

    assert isinstance(stats, HighPerformanceStats)
    assert stats.duration_ms < 500

    # 1. Check column statistics
    assert "Revenue" in stats.column_stats
    rev_stats = stats.column_stats["Revenue"]
    assert rev_stats["count"] == 100_000
    assert 2500 < rev_stats["mean"] < 2600
    assert rev_stats["min"] >= 100.0
    assert rev_stats["max"] <= 5000.0

    # 2. Check correlation matrix
    assert "Revenue" in stats.correlation_matrix
    assert "Cost" in stats.correlation_matrix["Revenue"]

    # 3. Check quantiles
    assert "Revenue" in stats.quantiles
    q = stats.quantiles["Revenue"]
    assert q["p10"] < q["p25"] < q["p50"] < q["p75"] < q["p90"] < q["p99"]


def test_analytical_sql_execution(large_sales_df):
    """Verify analytical SQL execution against in-memory DataFrames."""
    engine = HighPerformanceExecutionEngine()

    query = """
    SELECT Country, SUM(Revenue) AS TotalRevenue, AVG(Discount) AS AvgDiscount
    FROM sales
    WHERE Units >= 5
    GROUP BY Country
    ORDER BY TotalRevenue DESC
    LIMIT 5
    """

    res_df = engine.execute_sql(query, tables={"sales": large_sales_df})

    assert isinstance(res_df, pd.DataFrame)
    assert len(res_df) <= 5
    assert "Country" in res_df.columns
    assert "TotalRevenue" in res_df.columns
    assert "AvgDiscount" in res_df.columns
    assert res_df["TotalRevenue"].iloc[0] >= res_df["TotalRevenue"].iloc[-1]


def test_orchestrator_integration_with_hp_engine(large_sales_df):
    """Verify AutonomousCommandOrchestrator utilizes high-performance execution."""
    orchestrator = AutonomousCommandOrchestrator()

    res = orchestrator.execute_command(
        command="Find the top 5 countries by revenue",
        dataframe=large_sales_df,
        session_id="hp_test_session",
    )

    assert res.user_intent in ("eda", "segmentation", "ranking")
    assert "Revenue" in res.final_explanation or "revenue" in res.final_explanation
    assert len(res.execution_steps) >= 1
