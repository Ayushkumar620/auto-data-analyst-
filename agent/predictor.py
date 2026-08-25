"""
Data Predictor - Builds simple ML models using scikit-learn.
"""
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


class DataPredictor:
    """Builds simple predictive models from tabular data."""

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
        """Simple time-series forecast for a numeric column ordered by a date column.

        Uses linear regression on position (time index) to predict future values.
        """
        df = self._get_main_df()
        if df is None:
            return {"error": "No tabular data available for forecasting."}

        # Find date column
        date_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break
        if not date_col:
            for col in df.columns:
                c_low = col.lower()
                tokens = re.sub(r"[^\w]", " ", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col)).lower().split()
                if any(t in tokens or t in c_low for t in ("date", "datetime", "time", "timestamp", "month", "year", "quarter", "period", "fy", "cy")):
                    series = df[col].dropna()
                    if not series.empty:
                        if pd.api.types.is_numeric_dtype(df[col]) and series.between(1800, 2150).all():
                            date_col = col
                            break
                        elif pd.to_datetime(series.head(10), errors="coerce").notna().mean() >= 0.7:
                            date_col = col
                            break

        # Determine numeric target
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            return {"error": "No numeric column available for forecasting."}

        # 1. User explicit target always wins if present
        if target and target in df.columns:
            chosen_target = target
        else:
            # 2. Filter candidate business measures (exclude date_col, integer years, IDs, high nulls)
            candidate_cols = []
            for col in numeric.columns:
                if col == date_col:
                    continue
                series = df[col].dropna()
                if len(series) < 3 or series.nunique() <= 1:
                    continue
                tokens = re.sub(r"[^\w]", " ", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col)).lower().split()
                if any(t in tokens for t in ("year", "fy", "cy", "quarter", "qtr", "date", "timestamp", "month", "id", "key", "uuid", "sku", "code")):
                    if series.between(1800, 2150).all() or series.nunique() / len(series) > 0.8:
                        continue
                candidate_cols.append(col)

            metric_keywords = ["actual", "revenue", "sales", "demand", "profit", "budget", "volume", "usd", "amount", "spend", "units", "quantity", "price", "cost", "value"]
            if candidate_cols:
                scored = []
                for c in candidate_cols:
                    score = 0
                    tokens = re.sub(r"[^\w]", " ", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", c)).lower().split()
                    for idx, kw in enumerate(metric_keywords):
                        if kw in tokens or kw in c.lower():
                            score += (len(metric_keywords) - idx) * 10
                    score += min(10, df[c].dropna().nunique())
                    scored.append((c, score))
                scored.sort(key=lambda x: x[1], reverse=True)
                chosen_target = scored[0][0]
            else:
                chosen_target = numeric.columns[0]

        target = chosen_target

        # Get the target series
        y = pd.to_numeric(df[target], errors="coerce").dropna()
        if len(y) < 5:
            return {"error": "Need at least 5 valid data points for forecasting."}

        # Sort by date if available
        if date_col:
            try:
                dates = pd.to_datetime(df[date_col], errors="coerce")
                combined = pd.DataFrame({"_date": dates, "_value": y})
                combined = combined.dropna().sort_values("_date")
                y = combined["_value"].reset_index(drop=True)
            except Exception:
                pass

        # Simple position-based linear regression
        X = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)

        # In-sample predictions
        in_sample = model.predict(X)

        # Future predictions
        future_X = np.arange(len(y), len(y) + periods).reshape(-1, 1)
        future = model.predict(future_X)

        # Compute trend
        slope = float(model.coef_[0])
        trend = "upward" if slope > 0 else "downward"

        # Build forecast records
        forecast_records = []
        for i, val in enumerate(future):
            forecast_records.append({
                "period": len(y) + i + 1,
                "forecast": round(float(val), 4),
            })

        # Build history records
        history = []
        for i, val in enumerate(y):
            history.append({
                "index": i + 1,
                "actual": round(float(val), 4),
            })

        last_value = float(y.iloc[-1])
        projected_change = ((future[-1] - last_value) / last_value * 100) if last_value != 0 else 0

        return {
            "target": target,
            "target_column": target,
            "date_col": date_col,
            "time_column": date_col,
            "history_points": int(len(y)),
            "forecast_periods": periods,
            "forecast_horizon": periods,
            "forecast_values": [r["forecast"] for r in forecast_records],
            "slope": round(slope, 4),
            "trend": trend,
            "last_value": round(last_value, 4),
            "projected_change_percent": round(projected_change, 2),
            "history": history,
            "forecast": forecast_records,
        }

    def predict(self, target=None):
        """Train a model to predict a target column."""
        df = self._get_main_df()
        if df is None:
            return {"error": "No tabular data available for prediction."}

        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            cat = df.select_dtypes(include=["object"])
            if target and target in df.columns:
                features = [c for c in df.columns if c != target]
            elif len(cat.columns) >= 1:
                target = cat.columns[0]
                features = [c for c in df.columns if c != target]
            else:
                return {"error": "Not enough columns for prediction."}
        else:
            # Choose target if valid, else use last numeric column
            if not target or target not in df.columns:
                target = numeric.columns[-1]
            features = [c for c in df.columns if c != target]

        # Prepare data
        X = df[features].copy()
        y = df[target].copy()

        # Handle datetime columns in X
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = pd.to_datetime(X[col]).astype("int64") // 10**9

        # Encode categorical features
        for col in X.select_dtypes(include=["object", "string", "category"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Handle target
        is_classification = df[target].dtype == object or df[target].nunique() <= 10
        if is_classification:
            le = LabelEncoder()
            y_series = pd.Series(le.fit_transform(y.astype(str)), index=X.index)
        else:
            y_series = pd.to_numeric(y, errors="coerce")

        # Drop rows with NaN
        mask = X.notna().all(axis=1) & y_series.notna()
        X = X[mask]
        y = y_series[mask]

        if len(X) < 10:
            return {"error": "Need at least 10 valid rows for prediction."}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
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

        # Feature importance-ish (coefficients)
        coefs = {}
        if hasattr(model, "coef_"):
            coefs = {f: round(float(c), 4) for f, c in zip(features, model.coef_.flatten())}

        return {
            "target": target,
            "features": features,
            "metric": metric,
            "coefficients": coefs,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        }
