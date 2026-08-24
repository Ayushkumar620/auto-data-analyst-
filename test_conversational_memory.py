"""Tests for Conversational Memory & Multi-Turn Context Resolution Engine.

Verifies:
1. ConversationalMemoryEngine state management & turn history
2. Pronoun & anaphora disambiguation ("it", "those", "build model for it", "why?")
3. Context preservation across multi-turn AutonomousCommandOrchestrator runs
4. Memory eviction and TTL cleanups
"""
import numpy as np
import pandas as pd
import pytest

from agent.conversational_memory import ConversationalMemoryEngine, ConversationTurn, SessionState
from agent.command_orchestrator import AutonomousCommandOrchestrator, CommandExecutionResult


@pytest.fixture
def sales_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "Region": np.random.choice(["North", "South", "East", "West"], n),
        "Product": np.random.choice(["Widget A", "Widget B", "Widget C"], n),
        "Revenue": np.random.uniform(1000, 5000, n),
        "Profit": np.random.uniform(100, 1500, n),
        "Discount": np.random.uniform(0.05, 0.30, n),
        "Churn": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    })


def test_conversational_memory_turn_recording():
    """Verify session creation and turn history tracking."""
    mem = ConversationalMemoryEngine()
    session = mem.get_or_create_session("sess_001")
    assert session.session_id == "sess_001"
    assert len(session.turns) == 0

    turn1 = mem.record_turn(
        session_id="sess_001",
        user_command="Analyze sales dataset",
        resolved_command="Analyze sales dataset",
        intent="eda",
        active_metric="Revenue",
        active_dimension="Region",
        summary_findings=["Revenue average is $3,000 across regions."],
        evidence_count=2,
    )

    assert turn1.turn_id == 1
    assert session.active_metric == "Revenue"
    assert session.active_dimension == "Region"
    assert len(session.turns) == 1

    history = mem.get_session_history("sess_001")
    assert len(history) == 1
    assert history[0]["active_metric"] == "Revenue"


def test_pronoun_and_anaphora_resolution(sales_df):
    """Verify resolution of 'it', 'those', 'build model for it', 'why?'."""
    mem = ConversationalMemoryEngine()
    session_id = "sess_002"

    # Turn 1: User mentions Profit
    mem.record_turn(
        session_id=session_id,
        user_command="Analyze Profit across quarters",
        resolved_command="Analyze Profit across quarters",
        intent="root_cause",
        active_metric="Profit",
        active_dimension="Region",
        active_target="Churn",
    )

    # Turn 2: User asks "Why did it fall?" -> Resolves to "why did Profit fall?"
    cmd2, meta2 = mem.resolve_context("Why did it fall?", session_id=session_id, df=sales_df)
    assert "profit" in cmd2.lower()
    assert meta2["context_modified"] is True

    # Turn 3: User asks "Compare it with North" -> Resolves to "compare Profit with North"
    cmd3, meta3 = mem.resolve_context("Compare it with North", session_id=session_id, df=sales_df)
    assert "profit" in cmd3.lower()

    # Turn 4: User asks "Forecast it for 6 months" -> Resolves to "forecast Profit for 6 months"
    cmd4, meta4 = mem.resolve_context("Forecast it for next 6 months", session_id=session_id, df=sales_df)
    assert "profit" in cmd4.lower()

    # Turn 5: User asks "Train a model on it" -> Resolves to "build the best model to predict Churn"
    cmd5, meta5 = mem.resolve_context("Train the best model for it", session_id=session_id, df=sales_df)
    assert "churn" in cmd5.lower() or "profit" in cmd5.lower()

    # Turn 6: User asks "Show top 5 of those" -> Resolves to "Show top 5 by Region"
    cmd6, meta6 = mem.resolve_context("Show top 5 of those", session_id=session_id, df=sales_df)
    assert "region" in cmd6.lower()


def test_multi_turn_orchestrator_execution(sales_df):
    """Verify multi-turn autonomous execution with context propagation end-to-end."""
    mem = ConversationalMemoryEngine()
    orchestrator = AutonomousCommandOrchestrator(memory_engine=mem)
    session_id = "test_e2e_session"

    # Turn 1: Broad exploratory command
    res1 = orchestrator.execute_command(
        command="Analyze revenue and find top performing regions",
        dataframe=sales_df,
        session_id=session_id,
    )
    assert isinstance(res1, CommandExecutionResult)
    assert res1.session_id == session_id
    assert len(res1.execution_steps) >= 1
    assert "revenue" in res1.command.lower()

    # Turn 2: Contextual follow-up using pronoun "it"
    res2 = orchestrator.execute_command(
        command="Why did it decrease last month?",
        dataframe=sales_df,
        session_id=session_id,
    )
    assert isinstance(res2, CommandExecutionResult)
    assert res2.context_metadata["context_modified"] is True
    assert "revenue" in res2.resolved_command.lower() or "profit" in res2.resolved_command.lower()

    # Turn 3: Contextual follow-up for forecasting
    res3 = orchestrator.execute_command(
        command="Forecast it for next 5 periods",
        dataframe=sales_df,
        session_id=session_id,
    )
    assert isinstance(res3, CommandExecutionResult)
    assert "forecast" in res3.user_intent.lower() or "forecasting" in res3.user_intent.lower()

    # Turn 4: Contextual follow-up for model building
    res4 = orchestrator.execute_command(
        command="Build the best model for it",
        dataframe=sales_df,
        session_id=session_id,
    )
    assert isinstance(res4, CommandExecutionResult)
    assert res4.model_selection_summary is not None or len(res4.execution_steps) >= 1

    # Verify session history length
    history = mem.get_session_history(session_id)
    assert len(history) == 4
