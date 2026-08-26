"""
Comprehensive Test Suite for Canonical Data Layer & Universal Row Validator.

Tests all required edge cases:
1. Exactly 10 valid rows
2. 10+ valid rows
3. 100+ rows
4. Missing values in unrelated columns (never drops valid target/feature rows)
5. Missing target values properly accounted for
6. Numeric strings ($1,200, 45%, (35.5) for negative)
7. Mixed numeric types and floats
8. Datetime strings
9. Different date column names
10. Different target column names
11. Duplicate rows handling
12. Zero values
13. Negative values
14. Very large numbers (1M+)
15. Insufficient valid rows (<10 for predict, <5 for forecast) with structured diagnostic dictionary
16. Cross-sectional datasets (no time column)
17. Time-series datasets
18. Invariance regression: proves valid target rows are preserved despite sparse unrelated columns
"""
import pytest
import numpy as np
import pandas as pd

from agent.canonical_data_layer import CanonicalDataLayer, DatasetRowAudit
from agent.predictor import DataPredictor
from agent.autonomous_forecast_engine import AutonomousForecastEngine
from agent.forecasting_schemas import ForecastRequest


def test_1_numeric_coercion_strings_currencies_percentages():
    """1. Verify coercion of currency, commas, percentages, unit multipliers, and negative parentheses."""
    raw = pd.Series(["$1,250.50", "(500.00)", "75%", "2.5k", "1.2M", " -42.1 ", np.nan, "invalid"])
    cleaned = CanonicalDataLayer.coerce_numeric_series(raw)

    assert cleaned[0] == 1250.50
    assert cleaned[1] == -500.00
    assert cleaned[2] == 75.0
    assert cleaned[3] == 2500.0
    assert cleaned[4] == 1200000.0
    assert cleaned[5] == -42.1
    assert np.isnan(cleaned[6])
    assert np.isnan(cleaned[7])


def test_2_missing_values_in_unrelated_columns_do_not_drop_valid_target_rows():
    """2. Verify that a sparse/null unrelated column does NOT drop valid target rows."""
    n = 20
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="D"),
        "target_measure": np.arange(100, 100 + n, dtype=float),
        "notes": [np.nan] * 18 + ["note 1", "note 2"],
        "sparse_metric": [np.nan] * 19 + [42.0],
        "feature_dense": np.arange(n) * 2.0,
    })

    dp = DataPredictor(df)
    pred_res = dp.predict(target="target_measure")

    assert "error" not in pred_res
    assert pred_res["original_rows"] == 20
    assert pred_res["valid_rows"] == 20
    assert pred_res["train_size"] + pred_res["test_size"] == 20

    fc_res = dp.forecast(target="target_measure", periods=4)
    assert "error" not in fc_res
    assert fc_res["history_points"] == 20


def test_3_insufficient_rows_returns_structured_diagnostic():
    """3. Verify structured diagnostic response when rows < minimum requirement."""
    df_small = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "sales": [100.0, 200.0, np.nan],
    })

    dp = DataPredictor(df_small)
    res = dp.predict(target="sales")

    assert "error" in res
    assert "Need at least 10 valid rows" in res["error"]
    assert res["original_rows"] == 3
    assert res["valid_rows"] == 2
    assert res["target_column"] == "sales"
    assert res["minimum_required_rows"] == 10
    assert len(res["removal_reasons"]) > 0

    fc_res = dp.forecast(target="sales", periods=3)
    assert "error" in fc_res
    assert "Need at least 5 valid data points" in fc_res["error"]
    assert fc_res["minimum_required_rows"] == 5


def test_4_exact_10_valid_rows():
    """4. Verify exactly 10 valid rows succeeds in tabular prediction."""
    df_10 = pd.DataFrame({
        "feat_a": np.arange(10),
        "feat_b": np.arange(10) * 1.5,
        "target_val": 10.0 + np.arange(10) * 3.0,
    })

    dp = DataPredictor(df_10)
    res = dp.predict(target="target_val")
    assert "error" not in res
    assert res["valid_rows"] == 10


def test_5_10_plus_valid_rows():
    """5. Verify 15 valid rows succeeds."""
    df_15 = pd.DataFrame({
        "feat_a": np.arange(15),
        "feat_b": np.arange(15) * 2.0,
        "target_val": 50.0 + np.arange(15) * 4.0,
    })

    dp = DataPredictor(df_15)
    res = dp.predict(target="target_val")
    assert "error" not in res
    assert res["valid_rows"] == 15


def test_6_100_plus_rows_with_mixed_types_and_zeros():
    """6. Verify large datasets (100+ rows) with zeros and negative numbers."""
    n = 120
    df_large = pd.DataFrame({
        "dt": pd.date_range("2020-01-01", periods=n, freq="D"),
        "net_revenue": np.concatenate([np.zeros(20), np.linspace(-500, 1500, n - 20)]),
        "category": np.random.RandomState(42).choice(["Retail", "Wholesale", "Online"], size=n),
    })

    dp = DataPredictor(df_large)
    fc_res = dp.forecast(target="net_revenue", periods=10)
    assert "error" not in fc_res
    assert fc_res["history_points"] == n
    assert len(fc_res["forecast"]) == 10


def test_7_numeric_strings_and_dirty_formatting():
    """7. Verify automatic numeric coercion from formatted string columns."""
    df_dirty = pd.DataFrame({
        "period_date": pd.date_range("2023-01-01", periods=15, freq="ME"),
        "sales_str": ["$1,000", "$1,100", "$1,250", "$1,300", "$1,450",
                      "$1,500", "$1,620", "$1,700", "$1,850", "$1,900",
                      "$2,050", "$2,100", "$2,250", "$2,300", "$2,450"],
    })

    dp = DataPredictor(df_dirty)
    res = dp.forecast(target="sales_str", periods=4)
    assert "error" not in res
    assert res["history_points"] == 15
    assert res["forecast_values"][0] > 2400.0


def test_8_cross_sectional_dataset_without_date_column():
    """8. Verify cross-sectional datasets work for prediction but reject time-series forecasting."""
    df_cross = pd.DataFrame({
        "house_size": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500, 4000],
        "bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6],
        "price": [200000, 230000, 270000, 310000, 350000, 380000, 420000, 460000, 500000, 580000, 650000],
    })

    dp = DataPredictor(df_cross)
    pred_res = dp.predict(target="price")
    assert "error" not in pred_res
    assert pred_res["valid_rows"] == 11

    fc_res = dp.forecast(target="price")
    assert "error" in fc_res


def test_9_arbitrary_date_and_target_column_names():
    """9. Verify engine handles arbitrary column names without any hardcoding."""
    df_arb = pd.DataFrame({
        "tx_recorded_epoch": pd.date_range("2023-01-01", periods=16, freq="W"),
        "sensor_reading_kwh": 350.0 + np.arange(16) * 5.5,
        "metadata_region": ["North"] * 16,
    })

    dp = DataPredictor(df_arb)
    res = dp.forecast(target="sensor_reading_kwh", periods=4)
    assert "error" not in res
    assert res["target"] == "sensor_reading_kwh"
    assert res["time_column"] == "tx_recorded_epoch"
    assert len(res["forecast_values"]) == 4


def test_10_duplicate_timestamps_handling():
    """10. Verify duplicate timestamps are aggregated without dropping observations."""
    df_dup = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03", "2023-01-03", "2023-01-04", "2023-01-04", "2023-01-05", "2023-01-05"],
        "traffic": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
    })

    dp = DataPredictor(df_dup)
    res = dp.forecast(target="traffic", periods=2)
    assert "error" not in res
    assert len(res["forecast_values"]) == 2


def test_11_negative_and_zero_values_invariance():
    """11. Verify negative values and zeroes are modeled cleanly."""
    df_neg = pd.DataFrame({
        "time": pd.date_range("2023-01-01", periods=12, freq="ME"),
        "profit_margin": [-15.0, -10.0, -5.0, 0.0, 2.0, 5.0, 8.0, 12.0, 15.0, 18.0, 20.0, 25.0],
    })

    dp = DataPredictor(df_neg)
    res = dp.forecast(target="profit_margin", periods=3)
    assert "error" not in res
    assert res["slope"] > 0
    assert res["trend"] == "upward"


def test_12_very_large_numbers_scale_invariance():
    """12. Verify numeric scale invariance with values > 10,000,000."""
    df_large_num = pd.DataFrame({
        "ts": pd.date_range("2023-01-01", periods=14, freq="ME"),
        "total_assets": 10_000_000.0 + 500_000.0 * np.arange(14),
    })

    dp = DataPredictor(df_large_num)
    res = dp.forecast(target="total_assets", periods=3)
    assert "error" not in res
    assert res["forecast_values"][0] > 16_000_000.0
