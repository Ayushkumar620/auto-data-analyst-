from __future__ import annotations
import re
import pandas as pd
import numpy as np


class TimeSeriesDetector:
    DATE_WORDS = ("date", "time", "timestamp", "month", "year", "quarter", "period", "fy", "cy")

    def detect(self, dataframe: pd.DataFrame, date_column: str | None = None, target: str | None = None) -> dict[str, str | None]:
        if dataframe.empty:
            return {"date_column": None, "target": None}

        # 1. Detect Date/Time Column
        parsed = None
        if date_column and date_column in dataframe.columns:
            parsed = date_column
        else:
            # Datetime dtypes first
            for c in dataframe.columns:
                if pd.api.types.is_datetime64_any_dtype(dataframe[c]):
                    parsed = c
                    break
            # Integer years or matching date strings
            if not parsed:
                for c in dataframe.columns:
                    c_low = c.casefold()
                    if any(w in c_low for w in self.DATE_WORDS):
                        series = dataframe[c].dropna()
                        if not series.empty:
                            if pd.api.types.is_numeric_dtype(dataframe[c]) and series.between(1800, 2150).all():
                                parsed = c
                                break
                            elif pd.to_datetime(series.head(20), errors="coerce").notna().mean() >= 0.7:
                                parsed = c
                                break
            # Fallback parse string/object columns only (numeric columns are not arbitrary epoch dates)
            if not parsed:
                for c in dataframe.columns:
                    if dataframe[c].dtype == object or str(dataframe[c].dtype).startswith("str"):
                        series = dataframe[c].dropna()
                        if len(series) >= 3 and pd.to_datetime(series.head(20), errors="coerce").notna().mean() >= 0.8:
                            parsed = c
                            break

        # 2. Detect Target Metric Column
        numeric = list(dataframe.select_dtypes(include="number").columns)
        # If user explicitly specified target, use it
        if target and target in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[target]):
            metric = target
        else:
            # Filter out date columns, year columns, and IDs from candidate numeric targets
            candidates = [c for c in numeric if c != parsed]
            filtered_candidates = []
            for c in candidates:
                series = dataframe[c].dropna()
                if len(series) < 3 or series.nunique() <= 1:
                    continue
                tokens = re.sub(r"[^\w]", " ", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", c)).lower().split()
                if any(t in tokens for t in ("year", "fy", "cy", "quarter", "qtr", "date", "timestamp", "month", "id", "key", "uuid", "sku", "code")):
                    if series.between(1800, 2150).all() or series.nunique() / len(series) > 0.8:
                        continue
                filtered_candidates.append(c)

            # Score candidates
            metric_keywords = ["actual", "revenue", "sales", "demand", "profit", "budget", "volume", "usd", "amount", "spend", "units", "quantity", "price", "cost", "value", "count"]
            if filtered_candidates:
                scored = []
                for c in filtered_candidates:
                    score = 0
                    tokens = re.sub(r"[^\w]", " ", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", c)).lower().split()
                    for idx, kw in enumerate(metric_keywords):
                        if kw in tokens or kw in c.lower():
                            score += (len(metric_keywords) - idx) * 10
                    score += min(10, dataframe[c].dropna().nunique())
                    scored.append((c, score))
                scored.sort(key=lambda x: x[1], reverse=True)
                metric = scored[0][0]
            elif candidates:
                metric = candidates[0]
            else:
                metric = numeric[0] if numeric else None

        return {"date_column": parsed, "target": metric}
