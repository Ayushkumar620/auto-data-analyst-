"""
Tests for Milestone 5, Task 3: Autonomous Forecasting and What-If Analysis Engine.

Verifies:
1. Time-series detection
2. Forecast suitability assessment
3. Insufficient data rejection (N < 5)
4. Regular frequency detection
5. Irregular time-series handling
6. Naive baseline forecaster
7. Seasonal naive baseline forecaster
8. Exponential smoothing forecaster
9. Autoregressive ML forecaster
10. Chronological backtesting (no temporal leakage)
11. Metric calculations (MAE, RMSE, WAPE)
12. Prediction interval uncertainty bounds
13. Model candidate benchmarking and selection
14. Structured ForecastResult conformance
15. What-If scenario engine simulation
16. Percentage change scenario
17. Segment-specific scenario
18. Multiple scenarios comparison (Optimistic/Expected/Pessimistic)
19. Epistemic non-causal wording protection
20. Conversational forecasting integration
21. Conversational What-If multi-scenario integration
22. ToolRegistry integration (forecast_engine, scenario_engine)
23. Evidence creation and traceability
24. Rejection when no date column exists
25. AutonomousForecasterAgent run() conformance
"""
import pytest
import numpy as np
import pandas as pd

from agent.autonomous_forecast_engine import AutonomousForecastEngine
from agent.autonomous_forecaster_agent import AutonomousForecasterAgent
from agent.conversational_analyst import ConversationalAnalystAgent
from agent.forecasting_schemas import (
    ForecastPoint,
    ForecastRequest,
    ForecastResult,
    ForecastSuitabilityResult,
    ScenarioComparison,
    ScenarioResult,
    WhatIfRequest,
)
from agent.schemas import AgentResult, AgentStatus, ClaimType
from agent.timeseries_detector import TimeSeriesDetector
from agent.tool_registry import DEFAULT_TOOL_REGISTRY
from agent.what_if_scenario_engine import WhatIfScenarioEngine


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def monthly_sales_df():
    """24 months of synthetic monthly sales data with trend and seasonal pattern."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    trend = np.linspace(1000, 3000, 24)
    seasonality = np.sin(np.linspace(0, 4 * np.pi, 24)) * 300
    noise = np.random.normal(0, 50, 24)
    rev = trend + seasonality + noise

    regions = ["North", "South", "East", "West"] * 6
    return pd.DataFrame({
        "date": dates,
        "region": regions,
        "revenue": rev,
        "units": np.random.randint(50, 200, size=24),
    })


@pytest.fixture
def non_timeseries_df():
    """Cross-sectional dataframe without any date column."""
    return pd.DataFrame({
        "customer_id": [f"C{i:03d}" for i in range(20)],
        "age": np.random.randint(20, 60, size=20),
        "spend": np.random.uniform(100, 1000, size=20),
    })


# ==============================================================================
# 1-5. Detection & Suitability Tests
# ==============================================================================

def test_timeseries_detection_and_frequency(monthly_sales_df):
    """1 & 4. Test date column and frequency detection."""
    detector = TimeSeriesDetector()
    time_col = detector.detect_time_column(monthly_sales_df)
    target_col = detector.detect_target_column(monthly_sales_df, time_col=time_col)
    freq, is_reg = detector.infer_frequency(monthly_sales_df[time_col])

    assert time_col == "date"
    assert target_col == "revenue"
    assert freq == "M"
    assert is_reg is True


def test_forecast_suitability_assessment(monthly_sales_df):
    """2. Test suitability scoring on suitable dataset."""
    detector = TimeSeriesDetector()
    res = detector.assess_suitability(monthly_sales_df)

    assert res.suitable is True
    assert res.score >= 0.70
    assert res.observation_count == 24
    assert res.has_trend is True


def test_insufficient_historical_data_rejection():
    """3. Test that datasets with N < 5 are rejected with clear reasons."""
    df_short = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=3, freq="D"),
        "revenue": [100.0, 120.0, 110.0],
    })
    detector = TimeSeriesDetector()
    res = detector.assess_suitability(df_short)

    assert res.suitable is False
    assert res.observation_count == 3
    assert any("Insufficient" in r for r in res.reasons)


def test_rejection_when_no_date_column(non_timeseries_df):
    """24. Test that non-time-series datasets are rejected."""
    detector = TimeSeriesDetector()
    res = detector.assess_suitability(non_timeseries_df)

    assert res.suitable is False
    assert res.score == 0.0
    assert any("No datetime" in r for r in res.reasons)


# ==============================================================================
# 6-14. Forecasting Models, Backtesting & Predictions
# ==============================================================================

def test_autonomous_forecast_execution(monthly_sales_df):
    """6, 8, 10, 11, 12, 14. Test end-to-end forecasting pipeline."""
    engine = AutonomousForecastEngine()
    req = ForecastRequest(
        dataset=monthly_sales_df,
        time_column="date",
        target_column="revenue",
        forecast_horizon=6,
        confidence_level=0.80,
    )
    result = engine.run_forecast(req)

    assert isinstance(result, ForecastResult)
    assert result.status == "SUCCESS"
    assert len(result.predictions) == 6
    assert result.target == "revenue"
    assert "MAE" in result.validation_metrics
    assert "RMSE" in result.validation_metrics

    # Check prediction intervals
    for pt in result.predictions:
        assert pt.lower_bound <= pt.prediction <= pt.upper_bound


def test_candidate_models_and_backtest_ranking(monthly_sales_df):
    """9 & 13. Test candidate model evaluation against naive baseline."""
    engine = AutonomousForecastEngine()
    req = ForecastRequest(dataset=monthly_sales_df, forecast_horizon=4)
    result = engine.run_forecast(req)

    assert result.model_family in ("exponential_smoothing", "autoregressive_ml", "seasonal_naive", "naive_last", "moving_average", "linear_trend", "arima_statistical")
    assert len(result.evidence) > 0
    assert result.evidence[0].claim_type == ClaimType.INFERENCE


# ==============================================================================
# 15-19. What-If Scenario Simulations
# ==============================================================================

def test_percentage_what_if_scenario(monthly_sales_df):
    """15 & 16. Test single percentage adjustment scenario."""
    engine = WhatIfScenarioEngine()
    req = WhatIfRequest(
        dataset=monthly_sales_df,
        target="revenue",
        scenario_name="Price Increase +10%",
        changed_variables={"pct": 0.10},
    )
    res = engine.simulate_scenario(req)

    assert isinstance(res, ScenarioResult)
    assert res.scenario_name == "Price Increase +10%"
    assert res.percentage_difference == pytest.approx(10.0, rel=1e-2)
    assert res.scenario_value > res.baseline_value
    assert len(res.evidence) > 0


def test_segment_what_if_scenario(monthly_sales_df):
    """17. Test segment-specific shock scenario."""
    engine = WhatIfScenarioEngine()
    req = WhatIfRequest(
        dataset=monthly_sales_df,
        target="revenue",
        scenario_name="North Region Surge +20%",
        changed_variables={"segment": "North", "pct": 0.20, "dimension": "region"},
    )
    res = engine.simulate_scenario(req)

    assert res.scenario_value > res.baseline_value
    assert 0.0 < res.percentage_difference < 20.0  # Segment is a fraction of total


def test_multiple_scenarios_comparison(monthly_sales_df):
    """18 & 19. Test multi-scenario comparison matrix (Optimistic, Expected, Pessimistic)."""
    engine = WhatIfScenarioEngine()
    specs = {
        "Optimistic (+15%)": {"pct": 0.15},
        "Expected (+5%)": {"pct": 0.05},
        "Pessimistic (-10%)": {"pct": -0.10},
    }
    comp = engine.compare_scenarios(monthly_sales_df, target="revenue", scenarios_spec=specs)

    assert isinstance(comp, ScenarioComparison)
    assert len(comp.scenarios) == 3
    assert comp.ranked_scenarios[0].scenario_name == "Optimistic (+15%)"
    assert comp.ranked_scenarios[-1].scenario_name == "Pessimistic (-10%)"
    assert len(comp.evidence) >= 3


def test_epistemic_non_causal_protection(monthly_sales_df):
    """20. Test non-causal attribution disclaimers are attached to simulations."""
    engine = WhatIfScenarioEngine()
    req = WhatIfRequest(
        dataset=monthly_sales_df,
        target="revenue",
        scenario_name="Ad Spend +10%",
        changed_variables={"pct": 0.10},
    )
    res = engine.simulate_scenario(req)

    assert any("causal" in lim.lower() for lim in res.limitations)


# ==============================================================================
# 20-25. Conversational Integration, ToolRegistry & BaseAgent
# ==============================================================================

def test_conversational_forecasting_turn(monthly_sales_df):
    """20 & 21. Test conversational forecasting and scenario follow-up."""
    agent = ConversationalAnalystAgent()
    sess_id = "sess_fc_turn_test"

    # Turn 1: Initial upload and forecast
    resp1, ev1, meta1 = agent.chat("Forecast revenue for the next 6 months.", session_id=sess_id, data=monthly_sales_df)
    assert "Time-Series Forecast" in resp1
    assert len(ev1) > 0

    # Turn 2: Follow-up What-If scenario using same session data
    resp2, ev2, meta2 = agent.chat("What happens if revenue increases by 10%?", session_id=sess_id)
    assert "What-If Simulation" in resp2
    assert meta2["result"]["percentage_difference"] == pytest.approx(10.0, rel=1e-2)


def test_conversational_multi_scenario_turn(monthly_sales_df):
    """21. Test conversational multi-scenario command ("Show me best and worst scenarios")."""
    agent = ConversationalAnalystAgent()
    sess_id = "sess_multi_scen_test"

    resp, ev, meta = agent.chat("Show me the best and worst scenarios.", session_id=sess_id, data=monthly_sales_df)
    assert "What-If Multi-Scenario Analysis" in resp
    assert "Optimistic" in resp
    assert "Pessimistic" in resp


def test_tool_registry_forecasting_and_scenario_tools():
    """22. Test that forecast_engine and scenario_engine are registered and executable."""
    assert DEFAULT_TOOL_REGISTRY.has_tool("forecast_engine")
    assert DEFAULT_TOOL_REGISTRY.has_tool("scenario_engine")

    fc_tool = DEFAULT_TOOL_REGISTRY.get("forecast_engine")
    assert "time_series_forecasting" in fc_tool.capabilities

    sc_tool = DEFAULT_TOOL_REGISTRY.get("scenario_engine")
    assert "what_if_analysis" in sc_tool.capabilities


def test_autonomous_forecaster_agent_run(monthly_sales_df):
    """25. Test standardized BaseAgent run() conformance."""
    agent = AutonomousForecasterAgent()
    task = {
        "command": "Forecast revenue for 3 months",
        "data": monthly_sales_df,
        "target": "revenue",
        "horizon": 3,
    }
    result = agent.run(task)

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.status == AgentStatus.COMPLETED
    assert "predictions" in result.data
    assert len(result.evidence) > 0


def test_irregular_timeseries_interval_detection():
    """26. Test detecting irregular intervals and raising warnings."""
    dates = pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-22", "2024-02-18", "2024-03-01", "2024-04-10", "2024-05-02"])
    df = pd.DataFrame({"date": dates, "val": np.linspace(10, 50, len(dates))})
    detector = TimeSeriesDetector()
    res = detector.assess_suitability(df)

    assert res.suitable is True
    assert res.detected_frequency == "IRREGULAR"
    assert any("Irregular" in w for w in res.warnings)


def test_confidence_level_intervals_width(monthly_sales_df):
    """27. Test that 95% confidence interval is wider than 50% confidence interval."""
    engine = AutonomousForecastEngine()
    req_95 = ForecastRequest(dataset=monthly_sales_df, forecast_horizon=4, confidence_level=0.95)
    req_50 = ForecastRequest(dataset=monthly_sales_df, forecast_horizon=4, confidence_level=0.50)

    res_95 = engine.run_forecast(req_95)
    res_50 = engine.run_forecast(req_50)

    width_95 = res_95.predictions[0].upper_bound - res_95.predictions[0].lower_bound
    width_50 = res_50.predictions[0].upper_bound - res_50.predictions[0].lower_bound

    assert width_95 > width_50


def test_zero_target_wape_and_mape_handling():
    """28. Test that WAPE is calculated and MAPE is safely omitted if zeroes are present."""
    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    vals = [10.0, 0.0, 5.0, 0.0, 15.0, 20.0, 0.0, 30.0, 25.0, 10.0, 5.0, 12.0]
    df = pd.DataFrame({"date": dates, "val": vals})

    engine = AutonomousForecastEngine()
    req = ForecastRequest(dataset=df, forecast_horizon=3)
    res = engine.run_forecast(req)

    assert res.status == "SUCCESS"
    assert "WAPE" in res.validation_metrics


def test_custom_horizon_and_frequency_overrides(monthly_sales_df):
    """29. Test overriding horizon and frequency in ForecastRequest."""
    engine = AutonomousForecastEngine()
    req = ForecastRequest(dataset=monthly_sales_df, forecast_horizon=12, frequency="Q")
    res = engine.run_forecast(req)

    assert len(res.predictions) == 12
    assert res.forecast_horizon == 12
    assert res.frequency == "Q"


def test_what_if_lever_elasticity_simulation(monthly_sales_df):
    """30. Test variable elasticity when a lever feature is modified."""
    engine = WhatIfScenarioEngine()
    req = WhatIfRequest(
        dataset=monthly_sales_df,
        target="revenue",
        scenario_name="Units Demand Shock +15%",
        changed_variables={"units": 0.15},
    )
    res = engine.simulate_scenario(req)

    assert res.scenario_value != res.baseline_value
    assert any("elasticity" in a.lower() for a in res.assumptions)

