import os
from typing import Any, Dict, List

import pandas as pd
from werkzeug.utils import secure_filename

from backend.app.schemas.dataset import DatasetProfile


class FileService:
    def __init__(self, upload_folder: str):
        self.upload_folder = upload_folder
        os.makedirs(self.upload_folder, exist_ok=True)

    def upload_and_profile(self, uploaded_file: Any) -> Dict[str, Any]:
        if uploaded_file is None or getattr(uploaded_file, "filename", "") == "":
            raise ValueError("Please upload a data file.")

        filename = secure_filename(uploaded_file.filename)
        if not filename:
            raise ValueError("Please upload a data file.")

        destination = os.path.join(self.upload_folder, filename)
        uploaded_file.save(destination)

        dataframe = self._read_dataframe(destination)
        profile = self._build_profile(dataframe, filename)
        return self._to_dict(profile)

    def _read_dataframe(self, file_path: str) -> pd.DataFrame:
        extension = os.path.splitext(file_path)[1].lower()
        if extension == ".csv":
            return pd.read_csv(file_path)
        if extension in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

    def _build_profile(self, dataframe: pd.DataFrame, filename: str) -> DatasetProfile:
        preview_rows = dataframe.head(20).to_dict(orient="records")
        preview_rows = [self._normalize_value(row) for row in preview_rows]

        data_types = {col: str(dtype) for col, dtype in dataframe.dtypes.items()}
        numeric_columns = [col for col in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[col])]
        categorical_columns = [col for col in dataframe.columns if col not in numeric_columns and not pd.api.types.is_datetime64_any_dtype(dataframe[col])]
        date_columns = [col for col in dataframe.columns if pd.api.types.is_datetime64_any_dtype(dataframe[col])]

        profile = DatasetProfile(
            dataset_name=filename,
            rows=int(dataframe.shape[0]),
            columns=int(dataframe.shape[1]),
            column_names=list(dataframe.columns),
            missing_values=int(dataframe.isna().sum().sum()),
            duplicates=int(dataframe.duplicated().sum()),
            preview=preview_rows,
            data_types=data_types,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            date_columns=date_columns,
            memory_usage=self._format_memory_usage(dataframe.memory_usage(deep=True).sum()),
        )
        return profile

    def _to_dict(self, profile: DatasetProfile) -> Dict[str, Any]:
        return {
            "dataset_name": profile.dataset_name,
            "rows": profile.rows,
            "columns": profile.columns,
            "column_names": profile.column_names,
            "missing_values": profile.missing_values,
            "duplicates": profile.duplicates,
            "preview": profile.preview,
            "data_types": profile.data_types,
            "numeric_columns": profile.numeric_columns,
            "categorical_columns": profile.categorical_columns,
            "date_columns": profile.date_columns,
            "memory_usage": profile.memory_usage,
        }

    def _normalize_value(self, row: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                normalized[key] = None
            elif isinstance(value, (pd.Timestamp,)):
                normalized[key] = value.isoformat()
            else:
                normalized[key] = value
        return normalized

    def _format_memory_usage(self, size_bytes: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024 or unit == "GB":
                return f"{size_bytes:.2f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} GB"
