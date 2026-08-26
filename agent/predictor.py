"""
Data Predictor - Builds ML models and time-series forecasts using canonical validation.
"""
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


class DataPredictor:
    """Builds simple predictive models and forecasts from tabular data using CanonicalDataLayer."""

    def __init__(self, data):
        self.data = data

    def _get_main_df(self):
        if isinstance(self.data, dict):
            for df in self.data.values():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            return None
        return self.data if isinstance(self.data, pd.DataFrame) else None

    def forecast(self, target=None, periods=5):
        """Autonomous time-series forecast delegating to AutonomousForecastEngine as single source of truth."""
        df = self._get_main_df()
        if df is None or df.empty:
            return {"error": "No tabular data available for forecasting."}

        from agent.autonomous_forecast_engine import AutonomousForecastEngine
        from agent.forecasting_schemas import ForecastRequest
        from agent.timeseries_detector import TimeSeriesDetector
        from agent.canonical_data_layer import CanonicalDataLayer

        detector = TimeSeriesDetector()
        date_col = detector.detect_time_column(df)

        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty and not any(pd.to_numeric(df[c], errors="coerce").notna().sum() >= 5 for c in df.columns):
            return {"error": "No numeric column available for forecasting."}

        chosen_target = target if (target and target in df.columns and pd.api.types.is_numeric_dtype(df[target])) else (
            detector.detect_target_column(df, time_col=date_col) or (numeric.columns[0] if not numeric.empty else df.columns[-1])
        )

        audit, target_clean, time_clean = CanonicalDataLayer.audit_dataset_for_target(
            df,
            target_column=chosen_target,
            time_column=date_col,
            minimum_required_rows=5,
        )

        if audit.valid_rows < 5:
            diag_reasons = audit.removal_reasons or ["Insufficient historical records with valid target and time values."]
            return {
                "error": f"Need at least 5 valid data points for forecasting. Found {audit.valid_rows}.",
                "original_rows": audit.original_rows,
                "parsed_rows": audit.parsed_rows,
                "valid_rows": audit.valid_rows,
                "target_column": chosen_target,
                "time_column": date_col,
                "target_valid_rows": audit.target_valid_rows,
                "time_series_valid_rows": audit.time_series_valid_rows,
                "rows_removed": audit.rows_removed,
                "removal_reasons": diag_reasons,
                "minimum_required_rows": 5,
            }

        req = ForecastRequest(
            dataset=df,
            target_column=chosen_target,
            time_column=date_col,
            forecast_horizon=periods,
        )
        engine = AutonomousForecastEngine()
        res = engine.run_forecast(req)

        if res.status != "SUCCESS":
            error_msg = res.warnings[0] if res.warnings else "Forecasting not supported for this dataset."
            return {
                "error": error_msg,
                "original_rows": audit.original_rows,
                "parsed_rows": audit.parsed_rows,
                "valid_rows": audit.valid_rows,
                "target_column": chosen_target,
                "time_column": date_col,
                "target_valid_rows": audit.target_valid_rows,
                "time_series_valid_rows": audit.time_series_valid_rows,
                "rows_removed": audit.rows_removed,
                "removal_reasons": audit.removal_reasons or res.warnings,
                "minimum_required_rows": 5,
            }

        y_valid = target_clean.dropna()

        # Build history records
        history = [{"index": i + 1, "actual": round(float(val), 4)} for i, val in enumerate(y_valid)]

        # Build forecast records
        forecast_records = [
            {"period": len(y_valid) + i + 1, "forecast": round(float(pt.prediction), 4), "timestamp": pt.timestamp}
            for i, pt in enumerate(res.predictions)
        ]

        slope_val = res.slope if res.slope is not None else 0.0
        trend = "upward" if slope_val > 0 else ("downward" if slope_val < 0 else "flat")
        last_val = float(y_valid.iloc[-1]) if not y_valid.empty else 0.0

        return {
            "target": res.target,
            "target_column": res.target,
            "date_col": res.time_column,
            "time_column": res.time_column,
            "history_points": int(len(y_valid)),
            "forecast_periods": res.forecast_horizon,
            "forecast_horizon": res.forecast_horizon,
            "forecast_values": [r["forecast"] for r in forecast_records],
            "slope": round(float(slope_val), 4),
            "trend": trend,
            "last_value": round(last_val, 4),
            "projected_change_percent": res.projected_change_pct,
            "projected_change_pct": res.projected_change_pct,
            "history": history,
            "forecast": forecast_records,
            "model_name": res.model_name,
            "model_family": res.model_family,
            "validation_metrics": res.validation_metrics,
            "confidence_level": res.confidence_level,
            "original_rows": audit.original_rows,
            "valid_rows": audit.valid_rows,
        }

    def predict(self, target=None):
        """Train a model to predict a target column with canonical non-destructive validation."""
        df = self._get_main_df()
        if df is None or df.empty:
            return {"error": "No tabular data available for prediction."}

        from agent.canonical_data_layer import CanonicalDataLayer
        from agent.timeseries_detector import TimeSeriesDetector

        # Resolve target
        if target and target in df.columns:
            chosen_target = target
        else:
            detector = TimeSeriesDetector()
            date_col = detector.detect_time_column(df)
            chosen_target = detector.detect_target_column(df, time_col=date_col) or df.columns[-1]

        # Use canonical data layer to prepare features and target safely
        X, y, audit = CanonicalDataLayer.prepare_tabular_prediction_data(
            df,
            target_column=chosen_target,
            minimum_required_rows=10,
        )

        if X is None or y is None or len(X) < 10:
            diag_reasons = audit.removal_reasons or ["Insufficient valid observations for tabular prediction."]
            return {
                "error": f"Need at least 10 valid rows for prediction. Found {audit.valid_rows}.",
                "original_rows": audit.original_rows,
                "parsed_rows": audit.parsed_rows,
                "valid_rows": audit.valid_rows,
                "target_column": chosen_target,
                "time_column": audit.time_column,
                "target_valid_rows": audit.target_valid_rows,
                "time_series_valid_rows": audit.time_series_valid_rows,
                "rows_removed": audit.rows_removed,
                "removal_reasons": diag_reasons,
                "minimum_required_rows": 10,
            }

        is_classification = df[chosen_target].dtype == object or df[chosen_target].nunique() <= 10

        if is_classification:
            le = LabelEncoder()
            y_train_full = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
        else:
            y_train_full = y

        features = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_train_full, test_size=0.2, random_state=42
        )

        if is_classification:
            model = LogisticRegression(max_iter=1000)
        else:
            model = LinearRegression()

        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        except Exception as e:
            return {"error": f"Model training failed: {e}"}

        # Metrics
        if is_classification:
            preds = np.rint(preds).astype(int)
            metric = {"model": "Logistic Regression", "type": "classification", "accuracy": round(accuracy_score(y_test, preds), 4)}
        else:
            metric = {
                "model": "Linear Regression",
                "type": "regression",
                "r2_score": round(r2_score(y_test, preds), 4),
                "mean_squared_error": round(mean_squared_error(y_test, preds), 4),
            }

        coefs = {}
        if hasattr(model, "coef_"):
            coefs = {f: round(float(c), 4) for f, c in zip(features, model.coef_.flatten())}

        return {
            "target": chosen_target,
            "features": features,
            "metric": metric,
            "coefficients": coefs,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "original_rows": audit.original_rows,
            "valid_rows": audit.valid_rows,
        }
