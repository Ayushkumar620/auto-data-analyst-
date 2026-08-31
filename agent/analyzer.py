"""
Data Analyzer - Produces statistical summaries, reports, and insights.
"""
import pandas as pd
import numpy as np


class DataAnalyzer:
    """Performs statistical analysis on DataFrames."""

    def __init__(self, data):
        # data may be a single DataFrame or a dict of DataFrames (e.g. SQLite)
        self.data = data

    def _get_frames(self):
        """Return a list of (name, DataFrame) pairs."""
        if isinstance(self.data, dict):
            return list(self.data.items())
        return [("data", self.data)]

    def summary(self):
        """Full overview: shape, dtypes, sample, describe, nulls."""
        reports = []
        for name, df in self._get_frames():
            if not isinstance(df, pd.DataFrame):
                continue
            rep = {
                "name": name,
                "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
                "columns": list(df.columns),
                "dtypes": {str(c): str(dt) for c, dt in df.dtypes.items()},
                "describe": df.describe(include="all").fillna("").to_dict(),
                "nulls": df.isnull().sum().to_dict(),
                "head": df.head(10).to_dict(orient="records"),
            }
            reports.append(rep)
        return reports

    def describe(self):
        """Statistical summary of numeric columns."""
        reports = []
        for name, df in self._get_frames():
            if not isinstance(df, pd.DataFrame):
                continue
            numeric = df.select_dtypes(include=[np.number])
            if numeric.empty:
                reports.append(
                    {"name": name, "note": "No numeric columns found."}
                )
            else:
                reports.append(
                    {
                        "name": name,
                        "describe": numeric.describe().fillna("").to_dict(),
                    }
                )
        return reports

    def nulls(self):
        """Missing value analysis."""
        reports = []
        for name, df in self._get_frames():
            if not isinstance(df, pd.DataFrame):
                continue
            null_counts = df.isnull().sum()
            null_rows = df.isnull().any(axis=1).sum()
            reports.append(
                {
                    "name": name,
                    "null_counts": null_counts.to_dict(),
                    "total_null_cells": int(null_counts.sum()),
                    "rows_with_nulls": int(null_rows),
                    "total_rows": int(len(df)),
                    "null_percentage": round(null_counts.sum() / (len(df) * len(df.columns)) * 100, 2)
                    if len(df) and len(df.columns)
                    else 0,
                }
            )
        return reports

    def correlation(self):
        """Correlation matrix and full bivariate statistical relationships for numeric columns."""
        reports = []
        from agent.statistical_analysis_engine import StatisticalAnalysisEngine
        engine = StatisticalAnalysisEngine()
        for name, df in self._get_frames():
            if not isinstance(df, pd.DataFrame):
                continue
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] < 2:
                reports.append(
                    {"name": name, "note": "Need at least 2 numeric columns for correlation."}
                )
            else:
                stats_res = engine.analyze(data=df)
                rels = stats_res.get("relationships", [])
                top_rels = stats_res.get("top_relationships", rels)
                corr = stats_res.get("correlation_matrix") or numeric.corr().fillna("").to_dict()
                reports.append(
                    {
                        "name": name,
                        "matrix": corr,
                        "columns": list(numeric.columns),
                        "relationships": rels,
                        "top_relationships": top_rels,
                        "subgroup_analysis": stats_res.get("subgroup_analysis", {}),
                    }
                )
        return reports

    def head(self, n=10):
        """First N rows."""
        reports = []
        for name, df in self._get_frames():
            if not isinstance(df, pd.DataFrame):
                continue
            reports.append(
                {"name": name, "rows": df.head(n).to_dict(orient="records")}
            )
        return reports

    def unique_values(self):
        """Unique value counts per column."""
        reports = []
        for name, df in self._get_frames():
            if not isinstance(df, pd.DataFrame):
                continue
            uniques = {str(c): int(df[c].nunique()) for c in df.columns}
            reports.append({"name": name, "unique_counts": uniques})
        return reports

    def aggregate(self, metric=None, dimension=None, agg="sum"):
        """Calculate dimensional aggregation and summary."""
        reports = []
        for name, df in self._get_frames():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            from agent.canonical_data_layer import CanonicalDataLayer
            profile = CanonicalDataLayer.ingest(df).profile
            m = metric or (profile.numeric_columns[0] if profile.numeric_columns else None)
            d = dimension or (profile.categorical_columns[0] if profile.categorical_columns else None)
            if not m or not d or m not in df.columns or d not in df.columns:
                reports.append({"name": name, "note": "Valid metric and dimension required for aggregation."})
                continue
            s_num = CanonicalDataLayer.coerce_numeric_series(df[m])
            temp_df = pd.DataFrame({d: df[d], m: s_num}).dropna()
            if temp_df.empty:
                reports.append({"name": name, "note": "No valid observations for aggregation."})
                continue
            grouped = temp_df.groupby(d)[m].agg(["sum", "mean", "count", "min", "max"]).reset_index()
            grouped = grouped.sort_values("sum", ascending=False)
            reports.append({
                "name": name,
                "metric": m,
                "dimension": d,
                "aggregation": agg,
                "records": grouped.to_dict(orient="records"),
                "total": float(s_num.sum()),
                "mean": float(s_num.mean()),
            })
        return reports
