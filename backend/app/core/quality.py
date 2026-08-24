"""Universal Data Quality Engine.

The single centralized engine for data-quality assessment used by the UI,
agents, reports and APIs:

  - missing values
  - duplicate rows
  - invalid values (unparseable numbers/dates)
  - inconsistent types (mixed-type columns)
  - impossible values (negatives inside predominantly non-negative columns,
    values outside plausible bounds)
  - outliers (delegates to AnomalyDetectionEngine - one shared implementation)
  - high-cardinality columns
  - constant columns

The same result object is reused everywhere; individual consumers never
re-implement quality checks with different thresholds.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.app.core.anomalies import AnomalyDetectionEngine
from backend.app.core.temporal import TemporalIntelligenceEngine

HIGH_CARDINALITY_RATIO = 0.9
MISSING_CRITICAL_PCT = 20.0
MISSING_WARNING_PCT = 5.0


class DataQualityEngine:
    """Centralized data quality assessment."""

    def __init__(self) -> None:
        self.anomalies = AnomalyDetectionEngine()
        self.temporal = TemporalIntelligenceEngine()

    def assess(self, dataframe: pd.DataFrame,
               semantic_roles: dict[str, str] | None = None,
               identifiers: list[str] | None = None) -> dict[str, Any]:
        roles = semantic_roles or {}
        identifier_set = set(identifiers or [])
        rows = len(dataframe)
        issues: list[dict[str, Any]] = []

        missing = self._missing_assessment(dataframe, rows, issues)
        duplicates = self._duplicate_assessment(dataframe, rows, issues)
        invalid = self._invalid_assessment(dataframe, issues)
        type_issues = self._type_assessment(dataframe, issues)
        impossible = self._impossible_assessment(dataframe, roles, issues)
        constants = self._constant_assessment(dataframe, issues)
        high_card = self._high_cardinality_assessment(dataframe, identifier_set,
                                                      list(dataframe.columns), issues)
        outlier_report = self.anomalies.detect(dataframe)
        for column, result in outlier_report.get("columns", {}).items():
            if result["outlier_count"] > 0:
                issues.append({
                    "column": column,
                    "issue": "outliers",
                    "severity": "warning",
                    "count": result["outlier_count"],
                    "percentage": round(result["outlier_count"] / max(1, rows) * 100, 2),
                    "method": result["method"],
                    "detail": f"{result['outlier_count']} outlier value(s) flagged "
                              f"using the {result['method']} method.",
                })

        recommendations = self._recommendations(issues)
        quality_score = self._quality_score(rows, len(dataframe.columns), issues)

        return {
            "quality_score": quality_score,
            "row_count": rows,
            "column_count": len(dataframe.columns),
            "issues": issues,
            "summary": {
                "missing_values": missing,
                "duplicate_rows": duplicates,
                "invalid_values": invalid,
                "type_issues": len(type_issues),
                "possible_impossible_values": len(impossible),
                "outlier_columns": [
                    issue["column"] for issue in issues if issue["issue"] == "outliers"
                ],
                "constant_columns": [issue["column"] for issue in constants],
                "high_cardinality_columns": [issue["column"] for issue in high_card],
            },
            "recommendations": recommendations,
        }

    # -- individual assessments --------------------------------------------
    def _missing_assessment(self, dataframe: pd.DataFrame, rows: int,
                            issues: list[dict[str, Any]]) -> dict[str, int]:
        missing_counts: dict[str, int] = {}
        for column in dataframe.columns:
            count = int(dataframe[column].isna().sum())
            percentage = round(count / rows * 100, 2) if rows else 0.0
            if count == 0:
                continue
            missing_counts[str(column)] = count
            severity = ("critical" if percentage >= MISSING_CRITICAL_PCT
                        else "warning" if percentage >= MISSING_WARNING_PCT else "info")
            issues.append({
                "column": str(column),
                "issue": "missing_values",
                "severity": severity,
                "count": count,
                "percentage": percentage,
                "detail": f"{percentage}% of values are missing.",
            })
        return missing_counts

    def _duplicate_assessment(self, dataframe: pd.DataFrame, rows: int,
                              issues: list[dict[str, Any]]) -> dict[str, int]:
        count = int(dataframe.duplicated().sum())
        if count:
            issues.append({
                "column": None,
                "issue": "duplicate_rows",
                "severity": "warning",
                "count": count,
                "percentage": round(count / rows * 100, 2) if rows else 0.0,
                "detail": f"{count} fully duplicated row(s).",
            })
        return {"duplicate_rows": count}

    def _invalid_assessment(self, dataframe: pd.DataFrame,
                            issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for column in dataframe.columns:
            series = dataframe[column].dropna()
            if series.empty or pd.api.types.is_datetime64_any_dtype(series):
                continue
            guessed = self.temporal.detect_fields(dataframe[[column]])
            if not guessed:
                continue
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            invalid_count = int(parsed.isna().sum())
            if invalid_count:
                issues.append({
                    "column": str(column),
                    "issue": "invalid_values",
                    "severity": "warning",
                    "count": invalid_count,
                    "percentage": round(invalid_count / len(series) * 100, 2),
                    "detail": f"{invalid_count} value(s) do not parse as dates.",
                })
                found.append({"column": str(column), "invalid_count": invalid_count})
        return found

    def _type_assessment(self, dataframe: pd.DataFrame,
                         issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for column in dataframe.columns:
            series = dataframe[column].dropna()
            if series.empty or pd.api.types.is_numeric_dtype(series):
                continue
            if pd.api.types.is_datetime64_any_dtype(series):
                continue
            parsed = pd.to_numeric(series, errors="coerce")
            numeric_fraction = float(parsed.notna().mean())
            if 0.05 < numeric_fraction < 0.95:
                issues.append({
                    "column": str(column),
                    "issue": "inconsistent_types",
                    "severity": "warning",
                    "count": int(series.notna().sum()),
                    "percentage": round(numeric_fraction * 100, 2),
                    "detail": f"Column mixes numeric ({numeric_fraction:.0%}) and non-numeric values.",
                })
                found.append({"column": str(column), "numeric_fraction": numeric_fraction})
        return found

    def _impossible_assessment(self, dataframe: pd.DataFrame,
                               roles: dict[str, str],
                               issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for column in dataframe.columns:
            series = dataframe[column]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            values = series.dropna()
            if values.empty:
                continue
            role = roles.get(str(column), "")
            non_negative_like = role in {"metric", "derived_metric", "count", "category"}
            fraction_non_negative = float((values >= 0).mean())
            negative_count = int((values < 0).sum())
            if non_negative_like and 0.05 < (1 - fraction_non_negative) < 0.5:
                issues.append({
                    "column": str(column),
                    "issue": "impossible_values",
                    "severity": "warning",
                    "count": negative_count,
                    "percentage": round(negative_count / len(values) * 100, 2),
                    "detail": f"{negative_count} negative value(s) in a predominantly non-negative column.",
                })
                found.append({"column": str(column), "negative_count": negative_count})
        return found

    def _constant_assessment(self, dataframe: pd.DataFrame,
                             issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        rows = len(dataframe)
        for column in dataframe.columns:
            series = dataframe[column].dropna()
            if series.empty:
                continue
            if series.nunique(dropna=True) <= 1:
                issues.append({
                    "column": str(column),
                    "issue": "constant_column",
                    "severity": "info",
                    "count": int(rows),
                    "percentage": 100.0,
                    "detail": "Column contains a single distinct value.",
                })
                found.append({"column": str(column)})
        return found

    def _high_cardinality_assessment(self, dataframe: pd.DataFrame,
                                     identifier_set: set[str],
                                     columns: list[str],
                                     issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        rows = len(dataframe)
        for column in columns:
            if column in identifier_set:
                continue
            series = dataframe[column].dropna()
            if series.empty or rows == 0:
                continue
            # Skip numeric and datetime columns (they naturally have continuous unique values)
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
                continue
            unique_ratio = series.nunique(dropna=True) / rows
            if unique_ratio >= HIGH_CARDINALITY_RATIO:
                issues.append({
                    "column": column,
                    "issue": "high_cardinality",
                    "severity": "info",
                    "count": int(series.nunique(dropna=True)),
                    "percentage": round(unique_ratio * 100, 2),
                    "detail": f"Column has {series.nunique(dropna=True)} distinct values "
                              f"({unique_ratio:.0%} of rows); not treated as a grouping dimension.",
                })
                found.append({"column": column, "unique_ratio": round(unique_ratio, 3)})
        return found

    # -- scoring and recommendations ----------------------------------------
    def _quality_score(self, rows: int, columns: int, issues: list[dict[str, Any]]) -> int:
        if rows == 0 or columns == 0:
            return 0
        penalty = 0.0
        for issue in issues:
            if issue["issue"] == "outliers":
                continue
            percentage = min(float(issue.get("percentage") or 0.0), 50.0)
            if issue["severity"] == "critical":
                penalty += percentage * 1.5
            elif issue["severity"] == "warning":
                penalty += percentage * 0.6
            else:
                penalty += percentage * 0.2
        return max(0, min(100, int(round(100 - penalty))))

    def _recommendations(self, issues: list[dict[str, Any]]) -> list[str]:
        recommendations: list[str] = []
        if any(issue["issue"] == "missing_values" and issue["severity"] == "critical"
               for issue in issues):
            recommendations.append("Impute or drop columns with critically high missing values.")
        if any(issue["issue"] == "duplicate_rows" for issue in issues):
            recommendations.append("Remove fully duplicated rows.")
        if any(issue["issue"] == "impossible_values" for issue in issues):
            recommendations.append("Review impossible values (e.g. negatives or out-of-range figures).")
        if any(issue["issue"] == "inconsistent_types" for issue in issues):
            recommendations.append("Normalize columns that mix numeric and non-numeric values.")
        if any(issue["issue"] == "constant_column" for issue in issues):
            recommendations.append("Consider dropping constant columns; they carry no analytical signal.")
        return recommendations
