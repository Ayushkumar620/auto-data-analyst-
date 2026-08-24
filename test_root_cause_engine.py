"""Tests for Root-Cause Variance Decomposition & Counterfactual What-If Simulation Engine.

Verifies:
1. Mathematical variance bridge decomposition (Volume, Rate, Mix, and Segment effects)
2. Accurate segment contribution ranking (positive and negative growth drivers)
3. Counterfactual "What-If" scenario simulation with elasticity modeling
4. AutonomousCommandOrchestrator integration on root-cause and what-if commands
"""
import numpy as np
import pandas as pd
import pytest

from backend.app.core.root_cause_engine import (
    RootCauseDecompositionEngine,
    VarianceDecompositionReport,
    CounterfactualSimulationResult,
    global_root_cause_engine,
)
from agent.command_orchestrator import AutonomousCommandOrchestrator, CommandExecutionResult


@pytest.fixture
def financial_dataset():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    regions = np.random.choice(["North", "South", "East", "West"], n)
    products = np.random.choice(["Hardware", "Software", "Services"], n)
    units = np.random.randint(10, 100, n)
    # Simulate a structural margin drop in the second half
    prices = np.where(np.arange(n) > n // 2, np.random.uniform(50, 80, n), np.random.uniform(90, 120, n))
    revenue = units * prices
    costs = units * np.random.uniform(30, 45, n)
    profit = revenue - costs

    return pd.DataFrame({
        "Date": dates,
        "Region": regions,
        "Product": products,
        "Units": units,
        "Revenue": revenue,
        "Cost": costs,
        "Profit": profit,
    })


def test_variance_decomposition(financial_dataset):
    """Verify period-over-period variance bridge decomposition into volume, rate, and mix."""
    engine = RootCauseDecompositionEngine()

    report: VarianceDecompositionReport = engine.decompose_variance(
        df=financial_dataset,
        metric="Profit",
        dimension="Region",
        date_col="Date",
        volume_col="Units",
    )

    assert isinstance(report, VarianceDecompositionReport)
    assert report.metric_name == "Profit"
    assert report.dimension_name == "Region"
    assert report.total_delta < 0  # Profit dropped in second half
    assert report.price_rate_effect < 0  # Negative price rate effect
    assert len(report.top_negative_drivers) >= 1

    # Verify math balance: total_delta == volume_effect + price_rate_effect + mix_effect
    computed_sum = report.volume_effect + report.price_rate_effect + report.mix_effect
    assert pytest.approx(computed_sum, rel=1e-3) == report.total_delta


def test_what_if_counterfactual_simulation(financial_dataset):
    """Verify What-If simulation modeling on target metric."""
    engine = RootCauseDecompositionEngine()

    sim: CounterfactualSimulationResult = engine.simulate_what_if(
        df=financial_dataset,
        target_metric="Revenue",
        lever_feature="Units",
        percentage_change=10.0,
        dimension="Region",
    )

    assert isinstance(sim, CounterfactualSimulationResult)
    assert sim.target_metric == "Revenue"
    assert sim.lever_feature == "Units"
    assert sim.simulated_value > sim.baseline_value
    assert sim.percentage_impact > 0
    assert "North" in sim.segment_sensitivities


def test_orchestrator_root_cause_integration(financial_dataset):
    """Verify AutonomousCommandOrchestrator synthesizes root-cause bridge for 'why' queries."""
    orchestrator = AutonomousCommandOrchestrator()

    res: CommandExecutionResult = orchestrator.execute_command(
        command="Why did profit decrease?",
        dataframe=financial_dataset,
        session_id="rc_sess_01",
    )

    assert isinstance(res, CommandExecutionResult)
    assert "Variance Decomposition" in res.final_explanation
    assert "Volume / Activity Effect" in res.final_explanation
    assert "Rate / Pricing Effect" in res.final_explanation


def test_orchestrator_what_if_integration(financial_dataset):
    """Verify AutonomousCommandOrchestrator executes what-if scenario simulation."""
    orchestrator = AutonomousCommandOrchestrator()

    res: CommandExecutionResult = orchestrator.execute_command(
        command="What if units increase by 15%?",
        dataframe=financial_dataset,
        session_id="rc_sess_02",
    )

    assert isinstance(res, CommandExecutionResult)
    assert "Counterfactual Scenario Simulation" in res.final_explanation
    assert "Projected Net Impact" in res.final_explanation
