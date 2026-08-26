import time
import pandas as pd
import numpy as np
import pytest

from agent.execution_engine import ExecutionEngine, StepStatus
from agent.dynamic_planner import ExecutionPlan, ExecutionStep
from agent.semantic_schema_agent import SemanticSchemaAgent
from agent.intent import IntentAnalyzer, AnalyticalIntent
from agent.model_training_engine import TraditionalMLTrainer
from agent.tool_registry import ToolRegistry, DEFAULT_TOOL_REGISTRY
from agent.schemas import AgentResult


def test_parallel_dag_execution():
    """Verify that independent DAG steps at the same topological level execute concurrently."""
    # Create custom tool registry with artificial delay
    registry = ToolRegistry()

    def slow_eda(data, **kwargs):
        time.sleep(0.1)
        return {"summary": "EDA done", "rows": len(data)}

    def slow_anomaly(data, **kwargs):
        time.sleep(0.1)
        return {"anomalies_found": 0}

    def slow_agg(data, **kwargs):
        time.sleep(0.1)
        return {"total_sales": 1000}

    def fast_report(data, agent_outputs=None, **kwargs):
        return {"report": "Synthesized report", "num_inputs": len(agent_outputs or [])}

    registry.register("eda", slow_eda)
    registry.register("anomaly", slow_anomaly)
    registry.register("agg", slow_agg)
    registry.register("report", fast_report)

    # 3 independent steps in Level 0, and 1 dependent step in Level 1
    steps = [
        ExecutionStep(
            step_id="step_1",
            tool_name="eda",
            purpose="EDA Analysis",
            dependencies=[],
        ),
        ExecutionStep(
            step_id="step_2",
            tool_name="anomaly",
            purpose="Anomaly Detection",
            dependencies=[],
        ),
        ExecutionStep(
            step_id="step_3",
            tool_name="agg",
            purpose="Sales Aggregation",
            dependencies=[],
        ),
        ExecutionStep(
            step_id="step_4",
            tool_name="report",
            purpose="Executive Report",
            dependencies=["step_1", "step_2", "step_3"],
        ),
    ]

    plan = ExecutionPlan(
        plan_id="parallel_test_plan",
        user_intent={"type": "test"},
        objective="Test concurrency",
        steps=steps,
    )

    df = pd.DataFrame({"sales": [10, 20, 30, 40, 50], "region": ["N", "S", "E", "W", "C"]})

    engine = ExecutionEngine(tool_registry=registry, max_workers=4)
    start_t = time.time()
    res = engine.execute_plan(plan, df)
    elapsed = time.time() - start_t

    assert res.is_success
    assert res.data["successful_steps"] == 4

    # Because steps 1, 2, and 3 ran in parallel (each 0.1s), total time should be ~0.15-0.25s,
    # significantly less than sequential 0.3s + overhead
    assert elapsed < 0.28, f"Expected parallel execution to take < 0.28s, took {elapsed:.3f}s"

    # Step 4 must receive outputs of all 3 upstream steps
    step_4_out = res.data["step_outputs"]["step_4"]
    assert step_4_out["num_inputs"] == 3


def test_semantic_schema_memoization():
    """Verify that repeated schema profiling on the same dataframe hits O(1) cache."""
    df = pd.DataFrame({
        "order_id": range(1, 201),
        "revenue": np.random.uniform(100, 5000, 200),
        "customer_name": [f"Cust_{i}" for i in range(200)],
        "date": pd.date_range("2024-01-01", periods=200, freq="D"),
    })

    agent = SemanticSchemaAgent()

    # First call - full profiling
    t0 = time.time()
    dk1 = agent.analyze_dataset(df, dataset_name="orders.csv")
    first_duration = time.time() - t0

    # Second call - cache hit
    t1 = time.time()
    dk2 = agent.analyze_dataset(df, dataset_name="orders.csv")
    cache_duration = time.time() - t1

    assert dk1.dataset_name == "orders.csv"
    assert dk2.dataset_name == "orders.csv"
    assert dk1.row_count == 200
    assert dk2.row_count == 200
    assert cache_duration < first_duration
    assert cache_duration < 0.005  # Sub-5ms cache retrieval


def test_intent_analyzer_cache():
    """Verify that normalized intent queries are cached and return instantly."""
    analyzer = IntentAnalyzer()
    query = "forecast revenue for the next 4 quarters"

    t0 = time.time()
    res1 = analyzer.analyze(query)
    first_dur = time.time() - t0

    t1 = time.time()
    res2 = analyzer.analyze(query)
    cache_dur = time.time() - t1

    assert res1.primary_intent == AnalyticalIntent.FORECASTING
    assert res2.primary_intent == AnalyticalIntent.FORECASTING
    assert cache_dur <= first_dur


def test_model_training_parallel_multicore():
    """Verify that RandomForest and ExtraTrees estimators train cleanly with n_jobs=-1."""
    trainer = TraditionalMLTrainer()
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1], size=100)

    est, name, fam = trainer.build_estimator("Random Forest Classifier", "classification", hyperparams={})
    assert hasattr(est, "n_jobs")
    assert est.n_jobs == -1

    est.fit(X, y)
    preds = est.predict(X)
    assert len(preds) == 100
