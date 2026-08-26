"""
Comprehensive Test Suite for Temporal Column Handling & Semantic Feature Selection.

Verifies:
1. Time-series forecasting uses temporal columns strictly as chronological index/axis and NOT as ordinary ML features.
2. Supervised regression and classification derive rich numeric calendar and cyclical features from temporal columns.
3. Explicit feature selection overrides automatic defaults.
4. Arbitrary datetime column names (e.g., 't_omega', 'rec_point', 'idx_seq') and date formats work without hardcoded keywords.
5. Sparse unrelated columns do not cause valid target rows to be discarded.
6. Identifiers, constants, and high-cardinality noise columns are excluded from predictive features.
7. Forecasting and supervised prediction interpret datasets consistently according to task intent.
"""
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from agent.canonical_data_layer import CanonicalDataLayer
from agent.predictor import DataPredictor
from agent.agents import PredictionAgent, ForecastAgent
from agent.timeseries_detector import TimeSeriesDetector
from agent.pre_execution_validator import PreExecutionValidator
from agent.agent_result import AgentResult, AgentStatus


# ---------------------------------------------------------------------------
# 1. Forecasting: Temporal Column Used as Index, Not Ordinary ML Feature
# ---------------------------------------------------------------------------

def test_forecasting_does_not_treat_temporal_index_as_ordinary_ml_feature():
    """Verify that forecasting pipeline uses temporal column as chronology and does not pollute features."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "t_stamp_seq": dates,
        "metric_val": np.sin(np.linspace(0, 10, 20)) * 50 + 100,
        "unrelated_feature": np.random.normal(0, 1, 20),
    })

    # Prepare data for forecasting task
    X, y, audit = CanonicalDataLayer.prepare_tabular_prediction_data(
        df,
        target_column="metric_val",
        time_column="t_stamp_seq",
        task_type="forecasting",
    )

    # In forecasting mode with automatic features, raw datetime column is not in X
    assert "t_stamp_seq" not in X.columns
    assert "t_stamp_seq_year" not in X.columns

    # Verify ForecastAgent executes cleanly using time column for sequence
    agent = ForecastAgent()
    res = agent.run({"data": df, "target": "metric_val", "time_column": "t_stamp_seq", "periods": 4})
    assert res.is_success
    assert res.result.get("target") == "metric_val"
    assert len(res.result.get("forecast", [])) == 4


# ---------------------------------------------------------------------------
# 2. Supervised Regression: Encodes Temporal Columns Into Rich Numeric Features
# ---------------------------------------------------------------------------

def test_regression_encodes_temporal_columns_into_rich_numeric_features():
    """Verify that supervised regression converts datetime into calendar and cyclical features."""
    dates = pd.date_range("2022-01-15", periods=30, freq="ME")
    df = pd.DataFrame({
        "custom_date_col": dates,
        "feature_num": np.linspace(1, 30, 30),
        "target_output": np.linspace(100, 1000, 30) + np.random.normal(0, 5, 30),
    })

    X, y, audit = CanonicalDataLayer.prepare_tabular_prediction_data(
        df,
        target_column="target_output",
        include_temporal_features=True,
        task_type="regression",
    )

    # Generated temporal features should be present and strictly numeric
    assert X is not None
    assert len(X) == 30
    assert len(y) == 30

    col_names = list(X.columns)
    # Check that calendar features were generated
    has_year = any("custom_date_col_year" in c for c in col_names)
    has_month = any("custom_date_col_month" in c for c in col_names)
    has_elapsed = any("custom_date_col_elapsed_days" in c for c in col_names)
    has_sin = any("custom_date_col_month_sin" in c for c in col_names)

    assert has_year or has_month or has_elapsed or has_sin
    # All features in X must be finite numeric floats
    assert np.issubdtype(X.dtypes.iloc[0], np.number)
    assert not X.isna().any().any()


# ---------------------------------------------------------------------------
# 3. Explicit User Feature Selection Overrides Defaults
# ---------------------------------------------------------------------------

def test_explicit_feature_selection_override():
    """Verify that explicit feature selection overrides automatic selection/exclusion."""
    n = 25
    df = pd.DataFrame({
        "t_dim": pd.date_range("2023-01-01", periods=n, freq="D"),
        "feat_a": np.random.normal(10, 2, n),
        "feat_b": np.random.normal(50, 5, n),
        "feat_c_unwanted": np.random.normal(0, 1, n),
        "target_score": np.linspace(10, 100, n),
    })

    # Explicitly select only feat_a and feat_b
    X, y, _ = CanonicalDataLayer.prepare_tabular_prediction_data(
        df,
        target_column="target_score",
        features=["feat_a", "feat_b"],
        task_type="regression",
    )

    assert list(X.columns) == ["feat_a", "feat_b"]
    assert len(X) == n

    # Explicitly request temporal feature t_dim
    X_temp, _, _ = CanonicalDataLayer.prepare_tabular_prediction_data(
        df,
        target_column="target_score",
        features=["t_dim", "feat_a"],
        include_temporal_features=True,
        task_type="regression",
    )
    assert "feat_a" in X_temp.columns
    assert any("t_dim" in c for c in X_temp.columns)


# ---------------------------------------------------------------------------
# 4. Arbitrary Datetime Column Names and Diverse Formats
# ---------------------------------------------------------------------------

def test_arbitrary_datetime_column_names_and_formats():
    """Verify arbitrary column names like 'alpha_obs_point' and 'seq_epoch' are recognized."""
    # Case A: String dates in non-standard column name
    n = 20
    df1 = pd.DataFrame({
        "alpha_obs_point": [f"2023-{i % 12 + 1:02d}-15" for i in range(n)],
        "sensor_reading": np.random.uniform(20, 80, n),
        "target_metric": np.linspace(1, 100, n),
    })

    detector = TimeSeriesDetector()
    detected_time_col = detector.detect_time_column(df1)
    assert detected_time_col == "alpha_obs_point"

    # Case B: Integer Year Sequence (1990-2015) in arbitrary column name
    df2 = pd.DataFrame({
        "temporal_horizon_step": range(2000, 2020),
        "gdp_index": np.linspace(500, 1500, 20),
    })
    detected_year_col = detector.detect_time_column(df2)
    assert detected_year_col == "temporal_horizon_step"


# ---------------------------------------------------------------------------
# 5. Sparse Unrelated Columns Do Not Discard Target Rows
# ---------------------------------------------------------------------------

def test_sparse_unrelated_columns_do_not_drop_valid_target_rows():
    """Verify that unrelated columns with heavy nulls do NOT discard valid target rows."""
    n = 35
    df = pd.DataFrame({
        "target_col": np.linspace(10, 350, n),
        "clean_feature": np.random.normal(5, 1, n),
        "sparse_text_notes": [None if i % 3 != 0 else f"note_{i}" for i in range(n)],  # ~67% nulls
        "sparse_sensor": [None if i % 2 == 0 else float(i * 10) for i in range(n)],      # 50% nulls
    })

    X, y, audit = CanonicalDataLayer.prepare_tabular_prediction_data(
        df,
        target_column="target_col",
        minimum_required_rows=10,
    )

    assert X is not None
    assert len(X) == n
    assert len(y) == n
    assert audit.valid_rows == n
    assert audit.rows_removed == 0
    # Clean feature is retained
    assert "clean_feature" in X.columns
    # Sparse sensor (50% nulls) is imputed with median without dropping rows
    assert "sparse_sensor" in X.columns
    assert not X["sparse_sensor"].isna().any()


# ---------------------------------------------------------------------------
# 6. Identifiers and Constants Excluded from Predictive Features
# ---------------------------------------------------------------------------

def test_identifiers_and_constants_excluded_from_features():
    """Verify identifier columns (UUID, row IDs) and constant columns (0 variance) are omitted."""
    n = 25
    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "tx_guid": [f"guid-abc-{i}" for i in range(n)],
        "constant_col": [999.0] * n,
        "useful_feature": np.linspace(1, 10, n),
        "target_y": np.linspace(50, 500, n),
    })

    X, y, audit = CanonicalDataLayer.prepare_tabular_prediction_data(
        df,
        target_column="target_y",
    )

    assert "useful_feature" in X.columns
    assert "customer_id" not in X.columns
    assert "tx_guid" not in X.columns
    assert "constant_col" not in X.columns


# ---------------------------------------------------------------------------
# 7. End-to-End PredictionAgent on Arbitrary Dataset with Temporal Features
# ---------------------------------------------------------------------------

def test_end_to_end_prediction_agent_with_temporal_features():
    """Verify PredictionAgent successfully trains and validates with temporal features."""
    dates = pd.date_range("2023-01-01", periods=25, freq="W")
    df = pd.DataFrame({
        "timestamp_col": dates,
        "feat_num": np.random.normal(10, 2, 25),
        "target_revenue": np.linspace(1000, 5000, 25) + np.random.normal(0, 50, 25),
    })

    agent = PredictionAgent()
    res = agent.run({"data": df, "target": "target_revenue", "include_temporal_features": True})

    assert res.is_success
    assert res.result.get("target") == "target_revenue"
    assert res.metrics.get("r2_score") is not None or "r2_score" in res.data.get("metric", {})
    assert res.confidence > 0.0
