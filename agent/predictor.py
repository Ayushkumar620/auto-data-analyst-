"""
Data Predictor - Builds ML models and time-series forecasts using canonical validation.
"""
import re
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder


class DataPredictor:
    """Builds predictive models and forecasts from tabular data using CanonicalDataLayer."""

    def __init__(self, data):
        self.data = data

    def _get_main_df(self):
        if isinstance(self.data, dict):
            for df in self.data.values():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            return None
        return self.data if isinstance(self.data, pd.DataFrame) else None

    def forecast(self, target=None, periods=5, time_column=None):
        """Autonomous time-series forecast delegating to AutonomousForecastEngine as single source of truth."""
        df = self._get_main_df()
        if df is None or df.empty:
            return {"error": "No tabular data available for forecasting."}

        from agent.autonomous_forecast_engine import AutonomousForecastEngine
        from agent.forecasting_schemas import ForecastRequest
        from agent.timeseries_detector import TimeSeriesDetector
        from agent.canonical_data_layer import CanonicalDataLayer

        detector = TimeSeriesDetector()
        date_col = time_column if (time_column and time_column in df.columns) else detector.detect_time_column(df)

        if target and target in df.columns:
            chosen_target = target
        else:
            numeric = df.select_dtypes(include=[np.number])
            chosen_target = detector.detect_target_column(df, time_col=date_col) or (numeric.columns[0] if not numeric.empty else df.columns[-1])

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

        y_valid = CanonicalDataLayer.coerce_numeric_series(target_clean).dropna()

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

    def predict(self, target=None, features=None, include_temporal_features=True):
        """Train a model to predict a target column with multi-candidate benchmarking and canonical validation."""
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
            features=features,
            include_temporal_features=include_temporal_features,
            task_type="tabular_supervised",
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
            y_encoded = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
        else:
            y_encoded = y

        features = list(X.columns)
        n_samples = len(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        # Multi-Candidate Benchmark
        candidates = []
        if is_classification:
            candidates.append(("Logistic Regression", "Linear", LogisticRegression(max_iter=1000, random_state=42)))
            candidates.append(("Random Forest Classifier", "Ensemble", RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)))
            candidates.append(("Gradient Boosting Classifier", "Ensemble", GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)))
        else:
            candidates.append(("Linear Regression", "Linear", LinearRegression()))
            candidates.append(("Ridge Regression", "Linear", Ridge(alpha=1.0, random_state=42)))
            candidates.append(("Random Forest Regressor", "Ensemble", RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)))
            candidates.append(("Gradient Boosting Regressor", "Ensemble", GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)))

        best_name = None
        best_family = None
        best_model = None
        best_score = -float("inf")
        leaderboard = []

        cv_folds = max(2, min(5, len(X_train) // 3))
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if is_classification else KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for name, family, model_inst in candidates:
            try:
                # Cross-validation
                scores = []
                for tr_idx, val_idx in cv.split(X_train, y_train):
                    m_fold = model_inst.__class__(**model_inst.get_params())
                    m_fold.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                    p_val = m_fold.predict(X_train.iloc[val_idx])
                    if is_classification:
                        scores.append(accuracy_score(y_train.iloc[val_idx], np.rint(p_val).astype(int)))
                    else:
                        scores.append(r2_score(y_train.iloc[val_idx], p_val))

                mean_cv = float(np.mean(scores))

                # Fit on full train and test on holdout
                model_inst.fit(X_train, y_train)
                p_test = model_inst.predict(X_test)

                if is_classification:
                    p_test_int = np.rint(p_test).astype(int)
                    test_metric_val = float(accuracy_score(y_test, p_test_int))
                    leaderboard.append({
                        "model_name": name,
                        "family": family,
                        "cv_accuracy": round(mean_cv, 4),
                        "test_accuracy": round(test_metric_val, 4),
                    })
                else:
                    test_metric_val = float(r2_score(y_test, p_test))
                    leaderboard.append({
                        "model_name": name,
                        "family": family,
                        "cv_r2": round(mean_cv, 4),
                        "test_r2": round(test_metric_val, 4),
                    })

                # Selection with slight Occam's razor preference for simpler models
                simplicity_bonus = 0.02 if family == "Linear" else 0.0
                adjusted_score = mean_cv + simplicity_bonus

                if adjusted_score > best_score or best_model is None:
                    best_score = adjusted_score
                    best_name = name
                    best_family = family
                    best_model = model_inst
            except Exception:
                continue

        if best_model is None:
            best_model = LogisticRegression(max_iter=1000) if is_classification else LinearRegression()
            best_model.fit(X_train, y_train)
            best_name = "Logistic Regression" if is_classification else "Linear Regression"
            best_family = "Linear"

        preds = best_model.predict(X_test)

        if is_classification:
            preds_int = np.rint(preds).astype(int)
            acc = round(float(accuracy_score(y_test, preds_int)), 4)
            metric = {
                "model": best_name,
                "type": "classification",
                "accuracy": acc,
                "f1_score": round(float(f1_score(y_test, preds_int, average="weighted", zero_division=0)), 4),
            }
        else:
            r2 = round(float(r2_score(y_test, preds)), 4)
            mse = round(float(mean_squared_error(y_test, preds)), 4)
            mae = round(float(mean_absolute_error(y_test, preds)), 4)
            metric = {
                "model": best_name,
                "type": "regression",
                "r2_score": r2,
                "mean_squared_error": mse,
                "mean_absolute_error": mae,
            }

        coefs = {}
        if hasattr(best_model, "coef_"):
            coefs = {f: round(float(c), 4) for f, c in zip(features, best_model.coef_.flatten())}
        elif hasattr(best_model, "feature_importances_"):
            coefs = {f: round(float(c), 4) for f, c in zip(features, best_model.feature_importances_)}

        return {
            "target": chosen_target,
            "features": features,
            "metric": metric,
            "model_name": best_name,
            "model_family": best_family,
            "leaderboard": leaderboard,
            "coefficients": coefs,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "original_rows": audit.original_rows,
            "valid_rows": audit.valid_rows,
        }
