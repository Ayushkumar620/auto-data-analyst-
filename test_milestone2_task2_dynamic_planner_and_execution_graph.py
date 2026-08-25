"""
Tests for Milestone 2, Task 2: Dynamic Task Planner and Execution Graph.

Verifies:
1. Simple one-step plan generation
2. Multi-step compound plan generation
3. Dependency resolution & topological execution order
4. Independent parallel step isolation
5. Missing tool validation & rejection
6. Invalid tool rejection in execution graph
7. Circular dependency & cycle detection
8. Failed step handling and downstream skipping
9. Retry policy execution on recoverable step failure
10. Partial success preservation when independent steps fail
11. Invalid AgentResult validation handling
12. LLM-generated invalid plan rejection
13. Deterministic fallback plan synthesis
14. DatasetKnowledge integration during planning
15. UserIntent integration during planning
"""
import pytest
from datetime import datetime
import pandas as pd

from agent.dataset_knowledge import DatasetKnowledge
from agent.dynamic_planner import (
    DynamicTaskPlanner,
    ExecutionGraph,
    ExecutionPlan,
    ExecutionStep,
)
from agent.execution_engine import ExecutionEngine, StepStatus
from agent.intent import CommandIntelligenceAgent, IntentType, UserIntent
from agent.schemas import AgentError, AgentResult, AgentStatus
from agent.tool_registry import DEFAULT_TOOL_REGISTRY, ToolDefinition, ToolRegistry


# ==============================================================================
# Deterministic Test Fixtures
# ==============================================================================

@pytest.fixture
def sample_sales_df():
    """Deterministic sample dataframe for plan execution."""
    return pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "country": ["India", "US", "India", "US", "India", "US", "India", "US", "India", "US"],
        "revenue": [100.0, 200.0, 150.0, 300.0, 120.0, 250.0, 180.0, 400.0, 220.0, 350.0],
        "profit": [20.0, 50.0, 30.0, 80.0, 25.0, 60.0, 40.0, 100.0, 50.0, 90.0],
    })


@pytest.fixture
def mock_dataset_knowledge():
    """Mock DatasetKnowledge for e-commerce sales."""
    return DatasetKnowledge(
        dataset_id="sales_2024",
        dataset_name="sales.csv",
        row_count=10,
        column_count=4,
        columns=["order_date", "country", "revenue", "profit"],
        categorical_columns=["country"],
        numerical_columns=["revenue", "profit"],
        confidence=0.95,
    )


# ==============================================================================
# 1-4. Plan Generation & Dependency Resolution Tests
# ==============================================================================

def test_simple_one_step_plan(sample_sales_df):
    """1. Test generating a clean, valid one-step / standard EDA plan."""
    planner = DynamicTaskPlanner()
    intent = UserIntent(
        intent_type=IntentType.DATASET_ANALYSIS,
        objective="Run summary statistics.",
        metrics=["revenue"],
        required_capabilities=["eda"],
    )
    plan = planner.create_execution_plan(intent, dataframe=sample_sales_df)

    assert isinstance(plan, ExecutionPlan)
    assert len(plan.steps) >= 2
    assert "dataset_profiling" in plan.required_tools
    assert plan.confidence >= 0.80


def test_multi_step_plan(sample_sales_df):
    """2. Test generating a compound multi-step execution plan."""
    planner = DynamicTaskPlanner()
    cmd = "Clean the dataset, remove duplicate customers, find the top 10 cities by revenue, and forecast next month's revenue."
    plan = planner.create_execution_plan(cmd, dataframe=sample_sales_df)

    assert isinstance(plan, ExecutionPlan)
    tool_names = [s.tool_name for s in plan.steps]
    assert "dataset_profiling" in tool_names
    assert "data_cleaning" in tool_names
    assert "aggregation" in tool_names
    assert "forecasting" in tool_names
    assert len(plan.steps) >= 4


def test_dependency_resolution():
    """3. Test topological ordering and dependency structure."""
    step1 = ExecutionStep(step_id="step_1", tool_name="dataset_profiling", dependencies=[])
    step2 = ExecutionStep(step_id="step_2", tool_name="data_cleaning", dependencies=["step_1"])
    step3 = ExecutionStep(step_id="step_3", tool_name="aggregation", dependencies=["step_2"])

    graph = ExecutionGraph([step1, step2, step3])
    assert graph.detect_cycles() is False
    order = graph.get_execution_order()
    assert order == [["step_1"], ["step_2"], ["step_3"]]


def test_independent_parallel_tasks():
    """4. Test that independent steps share the same topological execution level."""
    step1 = ExecutionStep(step_id="step_1", tool_name="dataset_profiling", dependencies=[])
    step2_a = ExecutionStep(step_id="step_2a", tool_name="aggregation", dependencies=["step_1"])
    step2_b = ExecutionStep(step_id="step_2b", tool_name="anomaly_detection", dependencies=["step_1"])
    step3 = ExecutionStep(step_id="step_3", tool_name="reporting", dependencies=["step_2a", "step_2b"])

    graph = ExecutionGraph([step1, step2_a, step2_b, step3])
    assert graph.detect_cycles() is False
    order = graph.get_execution_order()
    assert order[0] == ["step_1"]
    assert set(order[1]) == {"step_2a", "step_2b"}
    assert order[2] == ["step_3"]


# ==============================================================================
# 5-7. Tool Validation & Cycle Rejection Tests
# ==============================================================================

def test_missing_tool_validation():
    """5. Test that plan referencing an unregistered tool is rejected with error."""
    registry = ToolRegistry()
    plan = ExecutionPlan(
        plan_id="plan_test_invalid",
        user_intent=UserIntent(intent_type="unknown"),
        steps=[
            ExecutionStep(step_id="step_1", tool_name="non_existent_tool_xyz", dependencies=[]),
        ],
    )
    errors = ExecutionGraph.validate_plan(plan, registry)
    assert len(errors) > 0
    assert "unregistered tool 'non_existent_tool_xyz'" in errors[0]


def test_invalid_tool_rejection_in_planner(sample_sales_df):
    """6. Test that DynamicTaskPlanner rejects invalid tools during validation."""
    planner = DynamicTaskPlanner()
    with pytest.raises(ValueError, match="Plan validation failed"):
        plan = ExecutionPlan(
            plan_id="plan_bad",
            user_intent=UserIntent(intent_type="unknown"),
            steps=[
                ExecutionStep(step_id="step_1", tool_name="unsupported_quantum_ml", dependencies=[]),
            ],
        )
        planner.execute_plan(plan, sample_sales_df)


def test_circular_dependency_rejection():
    """7. Test detection and rejection of circular dependencies in execution graph."""
    step1 = ExecutionStep(step_id="step_1", tool_name="dataset_profiling", dependencies=["step_2"])
    step2 = ExecutionStep(step_id="step_2", tool_name="data_cleaning", dependencies=["step_1"])

    graph = ExecutionGraph([step1, step2])
    assert graph.detect_cycles() is True

    with pytest.raises(ValueError, match="Circular dependency detected"):
        graph.get_execution_order()


# ==============================================================================
# 8-11. Execution Engine, Retries, and Error Handling Tests
# ==============================================================================

def test_failed_step_skips_downstream(sample_sales_df):
    """8. Test that a failed step causes downstream dependents to be skipped."""
    engine = ExecutionEngine()

    # Step 1: Broken tool that fails
    # Step 2: Dependent tool that must be skipped
    step1 = ExecutionStep(
        step_id="step_1",
        tool_name="failing_tool",
        dependencies=[],
        retry_policy={"max_retries": 0},
    )
    step2 = ExecutionStep(
        step_id="step_2",
        tool_name="aggregation",
        dependencies=["step_1"],
    )

    plan = ExecutionPlan(
        plan_id="plan_fail",
        user_intent=UserIntent(intent_type="test"),
        steps=[step1, step2],
    )

    result = engine.execute_plan(plan, sample_sales_df)
    assert step1.status == StepStatus.FAILED.value
    assert step2.status == StepStatus.SKIPPED.value
    assert result.status == AgentStatus.ERROR


def test_retry_on_recoverable_error(sample_sales_df):
    """9. Test that recoverable step errors trigger configured retry policy."""
    registry = ToolRegistry()
    call_count = {"count": 0}

    def flaky_fn(data, **kw):
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise RuntimeError("Transient connection blip")
        return AgentResult.success(output={"status": "recovered"})

    registry.register(
        ToolDefinition(
            name="flaky_tool",
            description="Flaky test tool",
            execution_fn=flaky_fn,
        )
    )

    engine = ExecutionEngine(tool_registry=registry)
    step = ExecutionStep(
        step_id="step_1",
        tool_name="flaky_tool",
        dependencies=[],
        retry_policy={"max_retries": 2},
    )
    plan = ExecutionPlan(
        plan_id="plan_retry",
        user_intent=UserIntent(intent_type="test"),
        steps=[step],
    )

    result = engine.execute_plan(plan, sample_sales_df)
    assert call_count["count"] == 2
    assert step.status == StepStatus.SUCCESS.value
    assert result.is_success is True


def test_partial_success_preservation(sample_sales_df):
    """10. Test that successful independent steps are preserved when an unrelated step fails."""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="failing_independent_tool",
            description="Independent failure",
            execution_fn=lambda data, **kw: AgentResult.error(error="Task failed"),
        )
    )

    engine = ExecutionEngine(tool_registry=registry)
    step_good = ExecutionStep(
        step_id="step_1",
        tool_name="dataset_profiling",
        dependencies=[],
    )
    step_bad = ExecutionStep(
        step_id="step_2",
        tool_name="failing_independent_tool",
        dependencies=[],
        retry_policy={"max_retries": 0},
    )

    plan = ExecutionPlan(
        plan_id="plan_partial",
        user_intent=UserIntent(intent_type="test"),
        steps=[step_good, step_bad],
    )

    result = engine.execute_plan(plan, sample_sales_df)
    assert result.status == AgentStatus.PARTIAL
    assert step_good.status == StepStatus.SUCCESS.value
    assert step_bad.status == StepStatus.FAILED.value
    assert "step_1" in result.data["step_outputs"]


def test_invalid_agent_result_handling(sample_sales_df):
    """11. Test that returning invalid / non-AgentResult objects is gracefully sanitized."""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="raw_dict_tool",
            description="Returns raw dict instead of AgentResult",
            execution_fn=lambda data, **kw: {"raw_key": "raw_val"},
        )
    )

    engine = ExecutionEngine(tool_registry=registry)
    step = ExecutionStep(step_id="step_1", tool_name="raw_dict_tool", dependencies=[])
    plan = ExecutionPlan(
        plan_id="plan_raw",
        user_intent=UserIntent(intent_type="test"),
        steps=[step],
    )

    result = engine.execute_plan(plan, sample_sales_df)
    assert result.is_success is True
    assert step.result is not None
    assert isinstance(step.result, AgentResult)


# ==============================================================================
# 12-15. LLM Validation, Fallbacks, and Context Integration Tests
# ==============================================================================

def test_llm_generated_invalid_plan_rejection():
    """12. Test that an invalid plan proposed by an LLM is caught and rejected."""
    registry = ToolRegistry()
    invalid_llm_plan = ExecutionPlan(
        plan_id="plan_hallucinated",
        user_intent=UserIntent(intent_type="deep_hack"),
        steps=[
            ExecutionStep(step_id="step_1", tool_name="arbitrary_python_eval_tool", dependencies=[]),
            ExecutionStep(step_id="step_2", tool_name="dataset_profiling", dependencies=["step_99"]),  # impossible dep
        ],
    )

    errors = ExecutionGraph.validate_plan(invalid_llm_plan, registry)
    assert len(errors) >= 2
    assert any("unregistered tool" in e for e in errors)
    assert any("impossible dependency 'step_99'" in e for e in errors)


def test_deterministic_fallback(sample_sales_df):
    """13. Test deterministic planning fallback when no LLM is configured."""
    planner = DynamicTaskPlanner(llm_provider=None)
    plan = planner.create_execution_plan("What is the average revenue by country?", dataframe=sample_sales_df)

    assert isinstance(plan, ExecutionPlan)
    assert len(plan.steps) >= 2
    assert "aggregation" in plan.required_tools


def test_dataset_knowledge_integration(sample_sales_df, mock_dataset_knowledge):
    """14. Test that DatasetKnowledge is used to select appropriate metrics and tools."""
    planner = DynamicTaskPlanner()
    plan = planner.create_execution_plan(
        "Why did profit decrease last quarter?",
        dataframe=sample_sales_df,
        knowledge=mock_dataset_knowledge,
    )

    assert isinstance(plan, ExecutionPlan)
    tool_names = [s.tool_name for s in plan.steps]
    assert "anomaly_detection" in tool_names
    assert "explanation" in tool_names


def test_user_intent_integration(sample_sales_df):
    """15. Test that pre-parsed UserIntent generates matching capability DAG."""
    planner = DynamicTaskPlanner()
    intent = UserIntent(
        intent_type=IntentType.FORECASTING,
        objective="Forecast next 6 periods of revenue.",
        metrics=["revenue"],
        required_capabilities=["forecasting"],
        confidence=0.96,
    )

    plan = planner.create_execution_plan(intent, dataframe=sample_sales_df)
    assert isinstance(plan, ExecutionPlan)
    assert "forecasting" in plan.required_tools
    assert plan.confidence == pytest.approx(0.96, 0.01)
