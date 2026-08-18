"""Extract facts with pandas; no language model performs calculations here."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


class FactAnalyzer:
    def extract(self, dataframe: pd.DataFrame, eda: Dict[str, Any]) -> Dict[str, Any]:
        rows, columns = dataframe.shape
        facts: Dict[str, Any] = {
            "row_count": int(rows),
            "column_count": int(columns),
            "missing_percentage": round(float(dataframe.isna().sum().sum() / (rows * columns) * 100), 2)
            if rows and columns else 0.0,
            "correlations": eda.get("correlations", []),
            "anomalies": eda.get("anomalies", {}).get("anomalies", []),
            "category_shares": [],
            "growth": [],
        }
        value_columns = list(dataframe.select_dtypes(include="number").columns)
        business_metric = next((column for column in value_columns
                                if any(token in str(column).lower() for token in ("revenue", "sales", "income", "amount"))), None)
        for column in dataframe.select_dtypes(exclude="number").columns:
            if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
                continue
            values = dataframe[column].dropna()
            if values.empty or values.nunique() > 50:
                continue
            counts = values.value_counts()
            category = self._json_value(counts.index[0])
            if business_metric:
                grouped = dataframe[[column, business_metric]].dropna().groupby(column)[business_metric].sum()
                total = grouped.sum()
                if total:
                    category = self._json_value(grouped.idxmax())
                    share = round(float(grouped.max() / total * 100), 2)
                    facts["category_shares"].append({"column": str(column), "category": category,
                                                      "share": share, "metric": str(business_metric)})
                    continue
            facts["category_shares"].append({
                "column": str(column), "category": category,
                "share": round(float(counts.iloc[0] / len(values) * 100), 2), "metric": "record_count",
            })
        facts["growth"] = self._growth_facts(dataframe)
        return facts

    def _growth_facts(self, dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
        candidate_date_columns = [
            column for column in dataframe.columns
            if pd.api.types.is_datetime64_any_dtype(dataframe[column])
            or any(token in str(column).lower() for token in ("date", "time", "month", "day", "year"))
        ]

        date_columns: List[str] = []
        for column in candidate_date_columns:
            series = dataframe[column]
            converted = pd.to_datetime(series, errors="coerce")
            if converted.notna().sum() >= max(1, series.notna().sum() * 0.8):
                date_columns.append(column)

        numeric_columns = list(dataframe.select_dtypes(include="number").columns)
        if not date_columns or not numeric_columns:
            return []

        date_column = date_columns[0]
        data = dataframe[[date_column, *numeric_columns]].copy()
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.dropna(subset=[date_column]).sort_values(date_column)

        results: List[Dict[str, Any]] = []
        for numeric_column in numeric_columns:
            numeric_data = data[[date_column, numeric_column]].dropna(subset=[numeric_column])
            if numeric_data.empty or len(numeric_data) < 2:
                continue
            first_value = float(numeric_data[numeric_column].iloc[0])
            last_value = float(numeric_data[numeric_column].iloc[-1])
            if first_value == 0:
                continue
            results.append({"date_column": str(date_column), "column": str(numeric_column),
                            "growth_percentage": round((last_value - first_value) / abs(first_value) * 100, 2),
                            "start_value": first_value, "end_value": last_value})
        return results

    @staticmethod
    def _json_value(value: Any) -> Any:
        return value.item() if hasattr(value, "item") else value
