from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from agent.json_utils import sanitize_for_json


class DatasetProfiler:
    def profile(self, dataframe: pd.DataFrame, filename: str, file_type: str, file_size: str) -> Dict[str, Any]:
        numeric_columns = [col for col in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[col]) and not pd.api.types.is_bool_dtype(dataframe[col])]
        categorical_columns = [col for col in dataframe.columns if col not in numeric_columns and not pd.api.types.is_datetime64_any_dtype(dataframe[col])]
        date_columns = [col for col in dataframe.columns if pd.api.types.is_datetime64_any_dtype(dataframe[col])]

        column_profiles = []
        for column in dataframe.columns:
            series = dataframe[column]
            column_profiles.append(self._profile_column(series, column))

        missing_summary = self._build_missing_summary(dataframe)
        duplicate_analysis = self._build_duplicate_analysis(dataframe)
        quality_score = self._score_quality(dataframe, missing_summary, duplicate_analysis)
        recommendations = self._build_recommendations(missing_summary, duplicate_analysis, dataframe, date_columns)

        payload = {
            "dataset": {
                "name": filename,
                "file_type": file_type,
                "rows": int(dataframe.shape[0]),
                "columns": int(dataframe.shape[1]),
                "size_mb": file_size,
            },
            "profile": {
                "quality_score": quality_score,
                "missing_values": int(dataframe.isna().sum().sum()),
                "duplicate_rows": int(dataframe.duplicated().sum()),
                "memory_usage": self._format_memory_usage(dataframe.memory_usage(deep=True).sum()),
            },
            "column_analysis": column_profiles,
            "numeric_analysis": self._build_numeric_analysis(dataframe, numeric_columns),
            "categorical_analysis": self._build_categorical_analysis(dataframe, categorical_columns),
            "missing_values": missing_summary,
            "duplicate_analysis": duplicate_analysis,
            "recommendations": recommendations,
            "preview": dataframe.head(20).to_dict(orient="records"),
        }
        return sanitize_for_json(payload)

    def _profile_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        return {
            "column_name": column_name,
            "data_type": str(series.dtype),
            "missing_values": int(series.isna().sum()),
            "unique_values": int(series.nunique(dropna=True)),
            "null_percentage": round((series.isna().mean() * 100), 2),
            "example_values": [self._clean_value(value) for value in series.dropna().head(5).tolist()],
        }

    def _build_numeric_analysis(self, dataframe: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
        analysis: Dict[str, Any] = {}
        for column in numeric_columns:
            series = dataframe[column]
            analysis[column] = {
                "mean": round(float(series.mean()), 4) if not pd.isna(series.mean()) else None,
                "median": round(float(series.median()), 4) if not pd.isna(series.median()) else None,
                "std": round(float(series.std()), 4) if not pd.isna(series.std()) else None,
                "min": round(float(series.min()), 4) if not pd.isna(series.min()) else None,
                "max": round(float(series.max()), 4) if not pd.isna(series.max()) else None,
                "quartiles": [round(float(value), 4) for value in series.quantile([0.25, 0.5, 0.75]).tolist()] if not series.empty else [],
            }
        return analysis

    def _build_categorical_analysis(self, dataframe: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, Any]:
        analysis: Dict[str, Any] = {}
        for column in categorical_columns:
            series = dataframe[column].dropna()
            if series.empty:
                analysis[column] = {"categories": 0, "top_category": None, "frequency": None, "cardinality": 0}
                continue
            value_counts = series.value_counts()
            top_category = value_counts.idxmax()
            analysis[column] = {
                "categories": int(value_counts.shape[0]),
                "top_category": self._clean_value(top_category),
                "frequency": int(value_counts.iloc[0]),
                "cardinality": int(value_counts.shape[0]),
            }
        return analysis

    def _build_missing_summary(self, dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
        missing = dataframe.isna().sum()
        return [
            {
                "column_name": column,
                "missing_values": int(count),
                "missing_percentage": round((count / len(dataframe)) * 100, 2) if len(dataframe) else 0,
            }
            for column, count in missing.items()
        ]

    def _build_duplicate_analysis(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        duplicate_rows = dataframe[dataframe.duplicated(keep=False)]
        duplicate_count = int(duplicate_rows.shape[0])
        return {
            "duplicate_rows": duplicate_count,
            "duplicate_ids": int(dataframe.duplicated(subset=[col for col in dataframe.columns if 'id' in col.lower()]).sum()) if any('id' in col.lower() for col in dataframe.columns) else 0,
            "duplicate_names": int(dataframe.duplicated(subset=[col for col in dataframe.columns if 'name' in col.lower()]).sum()) if any('name' in col.lower() for col in dataframe.columns) else 0,
        }

    def _score_quality(self, dataframe: pd.DataFrame, missing_summary: List[Dict[str, Any]], duplicate_analysis: Dict[str, Any]) -> int:
        total_penalty = 0
        total_penalty += sum(item["missing_percentage"] for item in missing_summary)
        total_penalty += duplicate_analysis["duplicate_rows"] / max(1, len(dataframe)) * 100
        quality_score = max(0, 100 - int(total_penalty))
        return min(100, quality_score)

    def _build_recommendations(self, missing_summary: List[Dict[str, Any]], duplicate_analysis: Dict[str, Any], dataframe: pd.DataFrame, date_columns: List[str]) -> List[str]:
        recommendations: List[str] = []
        if duplicate_analysis["duplicate_rows"] > 0:
            recommendations.append("Remove duplicate rows")
        if any(item["missing_percentage"] > 20 for item in missing_summary):
            recommendations.append("Consider imputing missing values or removing high-missing columns")
        for item in missing_summary:
            if item["missing_percentage"] > 0 and item["missing_percentage"] <= 20:
                recommendations.append(f"Fill missing values in {item['column_name']}")
                break
        for column in date_columns:
            recommendations.append(f"Convert {column} to datetime")
            break
        if not recommendations:
            recommendations.append("Dataset looks healthy")
        return recommendations

    def _clean_value(self, value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()
        return value

    def _format_memory_usage(self, size_bytes: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024 or unit == "GB":
                return f"{size_bytes:.2f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} GB"
