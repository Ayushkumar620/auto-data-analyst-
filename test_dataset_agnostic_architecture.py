"""
Comprehensive Test Suite for Dataset-Agnostic and Problem-Agnostic Architecture.

Validates that schema profiling, temporal detection, target resolution, problem classification,
model candidate generation, backtesting, and validation operate without any hardcoded column names
across 13 distinct synthetic dataset fixtures and modalities.
"""
import pytest
import numpy as np
import pandas as pd

from agent.timeseries_detector import TimeSeriesDetector
from agent.autonomous_forecast_engine import AutonomousForecastEngine
from agent.forecasting_schemas import ForecastRequest
from agent.predictor import DataPredictor
from agent.model_selection_agent import ModelSelectionAgent
from agent.model_selection_schemas import MLTaskType, DataModality
from agent.intent import CommandIntelligenceAgent, IntentType


# ==============================================================================
# 13 GENERIC SYNTHETIC FIXTURES
# ==============================================================================

@pytest.fixture
def fixture_1_monthly_sales():
    """1. Monthly longitudinal time series."""
    dates = pd.date_range("2022-01-01", periods=24, freq="ME")
    # Trend + seasonal pattern
    values = 100.0 + 3.5 * np.arange(24) + 15.0 * np.sin(np.arange(24) * np.pi / 6.0)
    return pd.DataFrame({"record_date": dates, "volume_units": values})


@pytest.fixture
def fixture_2_daily_traffic():
    """2. Daily website traffic time series."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    traffic = 5000 + 50 * np.arange(60) + np.random.RandomState(42).normal(0, 100, 60)
    return pd.DataFrame({"timestamp_utc": dates, "active_sessions": traffic})


@pytest.fixture
def fixture_3_financial_multi_measure():
    """3. Financial multi-measure dataset."""
    return pd.DataFrame({
        "fiscal_cycle_year": [2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023],
        "quarter_code": [1, 2, 3, 4, 1, 2, 3, 4],
        "metric_primary": [1200.0, 1450.0, 1600.0, 1900.0, 2100.0, 2350.0, 2500.0, 2800.0],
        "metric_secondary": [1150.0, 1400.0, 1550.0, 1850.0, 2000.0, 2300.0, 2450.0, 2750.0],
        "variance_delta": [50.0, 50.0, 50.0, 50.0, 100.0, 50.0, 50.0, 50.0],
    })


@pytest.fixture
def fixture_4_churn_classification():
    """4. Customer churn binary classification dataset."""
    rng = np.random.RandomState(42)
    n = 100
    return pd.DataFrame({
        "account_id": [f"ACC_{i:04d}" for i in range(n)],
        "tenure_months": rng.randint(1, 72, size=n),
        "monthly_fee": rng.uniform(20.0, 120.0, size=n),
        "support_calls": rng.randint(0, 10, size=n),
        "churn_label": rng.choice(["Yes", "No"], size=n, p=[0.25, 0.75]),
    })


@pytest.fixture
def fixture_5_house_price_regression():
    """5. Continuous tabular regression dataset."""
    rng = np.random.RandomState(42)
    n = 80
    sqft = rng.uniform(600, 4000, size=n)
    beds = rng.randint(1, 6, size=n)
    baths = rng.randint(1, 4, size=n)
    price = 50000 + sqft * 220 + beds * 15000 + baths * 10000 + rng.normal(0, 10000, n)
    return pd.DataFrame({
        "area_sqft": sqft,
        "bedroom_count": beds,
        "bathroom_count": baths,
        "valuation_amount": price,
    })


@pytest.fixture
def fixture_6_customer_segmentation():
    """6. Unsupervised customer segmentation dataset."""
    rng = np.random.RandomState(42)
    n = 60
    return pd.DataFrame({
        "annual_income_k": rng.uniform(25.0, 150.0, size=n),
        "spending_score": rng.uniform(1.0, 100.0, size=n),
        "age": rng.randint(18, 70, size=n),
    })


@pytest.fixture
def fixture_7_iot_sensor():
    """7. Sub-daily / Hourly IoT sensor time series."""
    dates = pd.date_range("2024-01-01 00:00:00", periods=48, freq="h")
    vibration = 0.05 + 0.02 * np.sin(np.arange(48) * np.pi / 12.0)
    return pd.DataFrame({"sensor_time": dates, "vibration_hz": vibration})


@pytest.fixture
def fixture_8_missing_values():
    """8. Dataset with missing values in both temporal and target features."""
    dates = pd.date_range("2023-01-01", periods=16, freq="ME")
    vals = [10.0, np.nan, 14.0, 18.0, np.nan, 22.0, 25.0, 30.0, 32.0, 35.0, np.nan, 40.0, 42.0, 45.0, 48.0, 50.0]
    return pd.DataFrame({"timestamp": dates, "sensor_readout": vals})


@pytest.fixture
def fixture_9_irregular_timestamps():
    """9. Dataset with irregularly spaced timestamps."""
    dates = [
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-01-03"),
        pd.Timestamp("2023-01-18"),
        pd.Timestamp("2023-02-02"),
        pd.Timestamp("2023-04-10"),
        pd.Timestamp("2023-05-01"),
        pd.Timestamp("2023-08-20"),
    ]
    return pd.DataFrame({"event_time": dates, "event_magnitude": [5.1, 4.2, 6.3, 3.8, 4.9, 5.5, 6.0]})


@pytest.fixture
def fixture_10_multiple_candidates():
    """10. Dataset with multiple numeric candidate measures."""
    dates = pd.date_range("2023-01-01", periods=12, freq="ME")
    return pd.DataFrame({
        "event_date": dates,
        "metric_alpha": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0],
        "metric_beta": [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0],
        "metric_gamma": [10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0, 17.0, 19.0, 21.0, 20.0],
    })


@pytest.fixture
def fixture_11_no_temporal_column():
    """11. Tabular dataset with no temporal column."""
    return pd.DataFrame({
        "category_tag": ["A", "B", "C", "D", "E", "F"],
        "attribute_one": [12.5, 14.2, 11.8, 16.0, 13.5, 15.0],
        "attribute_two": [100, 200, 150, 300, 250, 400],
    })


@pytest.fixture
def fixture_12_constant_target():
    """12. Dataset with constant (zero-variance) target."""
    dates = pd.date_range("2023-01-01", periods=12, freq="ME")
    return pd.DataFrame({"observation_date": dates, "constant_measure": [42.0] * 12})


@pytest.fixture
def fixture_13_small_sample():
    """13. Dataset with insufficient observations (N < 5)."""
    dates = pd.date_range("2023-01-01", periods=3, freq="ME")
    return pd.DataFrame({"date_col": dates, "metric_val": [10.0, 20.0, 30.0]})


# ==============================================================================
# TEST CASES
# ==============================================================================

def test_case_1_monthly_sales_forecasting(fixture_1_monthly_sales):
    """Verify generic monthly forecasting, frequency detection, candidate ranking, and validation metrics."""
    detector = TimeSeriesDetector()
    suitability = detector.assess_suitability(fixture_1_monthly_sales)
    assert suitability.suitable is True
    assert suitability.detected_time_column == "record_date"
    assert suitability.detected_target == "volume_units"
    assert suitability.detected_frequency == "M"

    engine = AutonomousForecastEngine()
    req = ForecastRequest(dataset=fixture_1_monthly_sales, forecast_horizon=6)
    result = engine.run_forecast(req)

    assert result.status == "SUCCESS"
    assert len(result.predictions) == 6
    assert result.target == "volume_units"
    assert "MAE" in result.validation_metrics
    assert "sMAPE" in result.validation_metrics
    assert result.validation_metrics["MAE"] >= 0.0


def test_case_2_daily_traffic_forecasting(fixture_2_daily_traffic):
    """Verify daily frequency detection and projection horizon."""
    detector = TimeSeriesDetector()
    suitability = detector.assess_suitability(fixture_2_daily_traffic)
    assert suitability.suitable is True
    assert suitability.detected_frequency == "D"
    assert suitability.detected_target == "active_sessions"

    engine = AutonomousForecastEngine()
    req = ForecastRequest(dataset=fixture_2_daily_traffic, forecast_horizon=14)
    res = engine.run_forecast(req)
    assert res.status == "SUCCESS"
    assert len(res.predictions) == 14
    assert res.frequency == "D"


def test_case_3_financial_multi_measure_explicit_override(fixture_3_financial_multi_measure):
    """Verify explicit target override and temporal year handling without hardcoded names."""
    detector = TimeSeriesDetector()
    time_col = detector.detect_time_column(fixture_3_financial_multi_measure)
    assert time_col in ("fiscal_cycle_year", "quarter_code")

    # Explicit user target override
    target = detector.detect_target_column(fixture_3_financial_multi_measure, hint="variance_delta")
    assert target == "variance_delta"

    # Default statistical selection chooses metric with highest continuous variance
    auto_target = detector.detect_target_column(fixture_3_financial_multi_measure, time_col=time_col)
    assert auto_target in ("metric_primary", "metric_secondary")


def test_case_4_churn_classification_not_forced_to_forecast(fixture_4_churn_classification):
    """Verify non-time-series classification dataset is detected as classification and not forced into forecasting."""
    agent = ModelSelectionAgent()
    task_type = agent.detect_task_type(fixture_4_churn_classification, target_column="churn_label")
    assert task_type == MLTaskType.BINARY_CLASSIFICATION

    modality = agent.detect_data_modality(fixture_4_churn_classification)
    assert modality == DataModality.TABULAR


def test_case_5_house_price_regression_routing(fixture_5_house_price_regression):
    """Verify continuous regression target is routed to regression and evaluated on R2/RMSE."""
    agent = ModelSelectionAgent()
    task_type = agent.detect_task_type(fixture_5_house_price_regression, target_column="valuation_amount")
    assert task_type == MLTaskType.REGRESSION

    primary_metric, secondaries = agent.select_evaluation_metrics(task_type)
    assert primary_metric == "r2"
    assert "rmse" in secondaries


def test_case_6_customer_segmentation_clustering(fixture_6_customer_segmentation):
    """Verify target-less segmentation dataset defaults to clustering."""
    agent = ModelSelectionAgent()
    task_type = agent.detect_task_type(fixture_6_customer_segmentation, target_column=None)
    assert task_type == MLTaskType.CLUSTERING


def test_case_7_iot_sensor_hourly_forecasting(fixture_7_iot_sensor):
    """Verify sub-daily/hourly IoT sensor series forecasting."""
    detector = TimeSeriesDetector()
    suitability = detector.assess_suitability(fixture_7_iot_sensor)
    assert suitability.suitable is True
    assert suitability.detected_target == "vibration_hz"

    engine = AutonomousForecastEngine()
    req = ForecastRequest(dataset=fixture_7_iot_sensor, forecast_horizon=8)
    res = engine.run_forecast(req)
    assert res.status == "SUCCESS"
    assert len(res.predictions) == 8


def test_case_8_missing_values_handled_safely(fixture_8_missing_values):
    """Verify missing values are handled gracefully during forecasting."""
    engine = AutonomousForecastEngine()
    req = ForecastRequest(dataset=fixture_8_missing_values, forecast_horizon=4)
    res = engine.run_forecast(req)
    assert res.status == "SUCCESS"
    assert len(res.predictions) == 4
    for pt in res.predictions:
        assert not np.isnan(pt.prediction)
        assert np.isfinite(pt.prediction)


def test_case_9_irregular_timestamps_warning(fixture_9_irregular_timestamps):
    """Verify irregular timestamps trigger appropriate suitability warning."""
    detector = TimeSeriesDetector()
    suitability = detector.assess_suitability(fixture_9_irregular_timestamps)
    assert suitability.detected_frequency == "IRREGULAR"
    assert any("irregular" in w.lower() for w in suitability.warnings)


def test_case_10_multiple_candidates_ambiguity_detection(fixture_10_multiple_candidates):
    """Verify ambiguity detection flags multiple close candidate targets for clarification."""
    detector = TimeSeriesDetector()
    ambiguous = detector.get_ambiguous_targets(fixture_10_multiple_candidates, time_col="event_date")
    assert len(ambiguous) >= 2
    assert "metric_alpha" in ambiguous
    assert "metric_beta" in ambiguous


def test_case_11_no_temporal_column_rejected_for_forecasting(fixture_11_no_temporal_column):
    """Verify dataset with no datetime dimension is rejected for time-series forecasting."""
    detector = TimeSeriesDetector()
    suitability = detector.assess_suitability(fixture_11_no_temporal_column)
    assert suitability.suitable is False
    assert any("datetime or timestamp" in r.lower() for r in suitability.reasons)


def test_case_12_constant_target_rejected(fixture_12_constant_target):
    """Verify zero-variance target is rejected for forecasting."""
    detector = TimeSeriesDetector()
    target = detector.detect_target_column(fixture_12_constant_target, time_col="observation_date")
    assert target is None  # Filtered out due to zero variance


def test_case_13_small_sample_size_rejected(fixture_13_small_sample):
    """Verify sample size below minimal threshold (N < 5) is rejected."""
    detector = TimeSeriesDetector()
    suitability = detector.assess_suitability(fixture_13_small_sample)
    assert suitability.suitable is False
    assert any("insufficient" in r.lower() for r in suitability.reasons)


def test_end_to_end_user_intent_priority(fixture_1_monthly_sales):
    """Verify explicit user command 'forecast volume_units for 8 periods' overrides defaults."""
    from agent.semantic_schema_agent import SemanticSchemaAgent
    ssa = SemanticSchemaAgent()
    dk = ssa.analyze_dataset(fixture_1_monthly_sales)

    cia = CommandIntelligenceAgent()
    intent = cia.analyze_intent("forecast volume_units for 8 periods", knowledge=dk)
    assert intent.intent_type == IntentType.FORECASTING
    assert "volume_units" in intent.metrics or any(m in "volume_units" for m in intent.metrics)

    engine = AutonomousForecastEngine()
    req = ForecastRequest(dataset=fixture_1_monthly_sales, target_column="volume_units", forecast_horizon=8)
    res = engine.run_forecast(req)
    assert res.status == "SUCCESS"
    assert res.forecast_horizon == 8
    assert res.target == "volume_units"
    assert len(res.predictions) == 8
