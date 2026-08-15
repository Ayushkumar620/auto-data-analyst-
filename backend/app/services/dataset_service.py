import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
from werkzeug.utils import secure_filename

from backend.app.cleaning.cleaner import DataCleaner
from backend.app.profilers.dataset_profiler import DatasetProfiler


def _get_upload_stream(uploaded_file: Any):
    if hasattr(uploaded_file, "file"):
        return uploaded_file.file
    return getattr(uploaded_file, "stream", uploaded_file)


class DatasetService:
    def __init__(self, upload_folder: str):
        self.upload_folder = upload_folder
        os.makedirs(self.upload_folder, exist_ok=True)

    def upload_dataset(self, uploaded_file: Any) -> Dict[str, Any]:
        if uploaded_file is None:
            raise ValueError("Empty file upload. Please choose a file.")

        filename = getattr(uploaded_file, "filename", None) or getattr(uploaded_file, "name", "")
        if not filename:
            raise ValueError("Empty file upload. Please choose a file.")

        filename = secure_filename(filename)
        if not filename:
            raise ValueError("Empty file upload. Please choose a file.")

        extension = os.path.splitext(filename)[1].lower()
        if extension not in {".csv", ".xlsx", ".xls"}:
            raise ValueError("Unsupported file type. Only CSV and Excel files are allowed.")

        stream = _get_upload_stream(uploaded_file)
        file_size = 0
        try:
            stream.seek(0, os.SEEK_END)
            file_size = stream.tell()
            stream.seek(0)
        except Exception:
            try:
                content = stream.read()
                file_size = len(content)
                stream.seek(0)
            except Exception:
                file_size = 0

        if file_size > 100 * 1024 * 1024:
            raise ValueError("File is too large. Maximum size is 100MB.")

        destination = os.path.join(self.upload_folder, filename)
        if hasattr(uploaded_file, "save"):
            uploaded_file.save(destination)
        else:
            with open(destination, "wb") as destination_file:
                stream.seek(0)
                destination_file.write(stream.read())

        try:
            dataframe = self._read_dataframe(destination)
        except Exception as exc:
            raise ValueError(f"Corrupted or unreadable file: {exc}") from exc

        metadata = self._generate_metadata(dataframe, filename, file_size)
        preview = self._generate_preview(dataframe)

        dataset_object = {
            "id": f"dataset_{uuid.uuid4().hex[:8]}",
            "name": filename,
            "file_type": extension.lstrip("."),
            "rows": metadata["rows"],
            "columns": metadata["columns"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "uploaded",
        }

        return {
            "dataset": dataset_object,
            "metadata": metadata,
            "preview": preview,
        }

    def profile_dataset(self, dataframe: pd.DataFrame, filename: str, file_type: str, file_size: int) -> Dict[str, Any]:
        profiler = DatasetProfiler()
        size_value = self._coerce_size_bytes(file_size)
        result = profiler.profile(
            dataframe=dataframe,
            filename=filename,
            file_type=file_type,
            file_size=self._format_memory_usage(size_value),
        )
        return result

    def clean_dataset(self, uploaded_file: Any) -> Dict[str, Any]:
        result = self.upload_dataset(uploaded_file)
        dataframe = self._read_dataframe(os.path.join(self.upload_folder, result["dataset"]["name"]))
        cleaner = DataCleaner(dataframe)
        cleaned = cleaner.clean()
        profile = self.profile_dataset(
            dataframe=dataframe,
            filename=result["dataset"]["name"],
            file_type=result["dataset"]["file_type"],
            file_size=result["metadata"]["file_size"],
        )

        return {
            "status": "cleaned",
            "dataset": result["dataset"],
            "quality_before": cleaned["quality_before"],
            "quality_after": cleaned["quality_after"],
            "rows_removed": cleaned["rows_removed"],
            "missing_values_fixed": cleaned["missing_values_fixed"],
            "datatype_conversions": cleaned["datatype_conversions"],
            "outliers_detected": cleaned["outliers_detected"],
            "cleaning_report": cleaned["cleaning_report"],
            "preview": cleaned["cleaned_data"],
            "profile": profile["profile"],
        }

    def _read_dataframe(self, file_path: str) -> pd.DataFrame:
        extension = os.path.splitext(file_path)[1].lower()
        if extension == ".csv":
            return pd.read_csv(file_path)
        if extension in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)
        raise ValueError("Unsupported file type. Only CSV and Excel files are allowed.")

    def _generate_metadata(self, dataframe: pd.DataFrame, filename: str, file_size: int) -> Dict[str, Any]:
        return {
            "name": filename,
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
            "column_names": list(dataframe.columns),
            "data_types": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
            "memory_usage": self._format_memory_usage(dataframe.memory_usage(deep=True).sum()),
            "missing_values": int(dataframe.isna().sum().sum()),
            "duplicate_rows": int(dataframe.duplicated().sum()),
            "file_size": self._format_memory_usage(file_size),
        }

    def _generate_preview(self, dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
        preview_rows = dataframe.head(20).to_dict(orient="records")
        normalized: List[Dict[str, Any]] = []
        for row in preview_rows:
            normalized.append({key: None if pd.isna(value) else value for key, value in row.items()})
        return normalized

    def _coerce_size_bytes(self, size_value: Any) -> int:
        if isinstance(size_value, (int, float)):
            return int(size_value)
        if isinstance(size_value, str):
            numeric = size_value.split(" ")[0].strip()
            try:
                return int(float(numeric))
            except ValueError:
                return 0
        return 0

    def _format_memory_usage(self, size_bytes: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024 or unit == "GB":
                return f"{size_bytes:.2f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} GB"
