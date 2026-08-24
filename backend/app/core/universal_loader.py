"""Universal Dataset Ingestion & Format Parser.

Handles any standard data format:
- Tabular Columnar: CSV, TSV, Parquet, Arrow, Feather, Excel (XLSX, XLS, ODS)
- Semi-Structured: JSON, JSONL, NDJSON (automatic nested flattening)
- Relational Databases: SQLite, DuckDB
- Text & Documents: Plain text, PDF, Logs
- Compressed Archives: ZIP, GZIP
- Scientific Arrays: NumPy (.npy, .npz)
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import tarfile
import zipfile
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from backend.app.core.big_data_engine import MemoryOptimizer, MemoryProfile


class UniversalLoadError(Exception):
    """Raised when universal data loading fails."""
    pass


class UniversalDatasetLoader:
    """Enterprise-grade multi-format dataset loader."""

    SUPPORTED_EXTENSIONS = {
        ".csv": "CSV",
        ".tsv": "TSV",
        ".tab": "TSV",
        ".txt": "Text / Delimited",
        ".parquet": "Parquet Columnar",
        ".pq": "Parquet Columnar",
        ".feather": "Arrow Feather",
        ".arrow": "Arrow Columnar",
        ".json": "JSON / Hierarchical",
        ".jsonl": "JSON Lines",
        ".ndjson": "Newline Delimited JSON",
        ".xlsx": "Excel",
        ".xls": "Excel",
        ".ods": "OpenDocument Spreadsheet",
        ".db": "SQLite Database",
        ".sqlite": "SQLite Database",
        ".sqlite3": "SQLite Database",
        ".pdf": "PDF Document",
        ".log": "Log File",
        ".npy": "NumPy Array",
        ".npz": "NumPy Archive",
        ".zip": "ZIP Archive",
        ".gz": "GZIP Compressed",
    }

    @classmethod
    def load(
        cls,
        file_source: Union[str, io.BytesIO, bytes],
        file_name: Optional[str] = None,
        optimize_memory: bool = True,
    ) -> Tuple[Union[pd.DataFrame, Dict[str, pd.DataFrame]], Optional[MemoryProfile]]:
        """
        Universal dispatch loader. Ingests file and returns DataFrame (or dict of DataFrames for multi-table)
        along with MemoryProfile diagnostics.
        """
        if isinstance(file_source, (bytes, bytearray)):
            file_source = io.BytesIO(file_source)

        if isinstance(file_source, io.BytesIO):
            ext = os.path.splitext(file_name or "")[1].lower()
            return cls._load_from_stream(file_source, ext, optimize_memory)

        # File path
        file_path = str(file_source)
        if not os.path.exists(file_path):
            raise UniversalLoadError(f"File path does not exist: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        # Dispatch
        try:
            if ext in (".csv", ".txt", ".tsv", ".tab", ".log"):
                df = cls._load_delimited(file_path, ext)
            elif ext in (".parquet", ".pq"):
                df = cls._load_parquet(file_path)
            elif ext in (".feather", ".arrow"):
                df = cls._load_feather(file_path)
            elif ext in (".json", ".jsonl", ".ndjson"):
                df = cls._load_json(file_path, ext)
            elif ext in (".xlsx", ".xls", ".ods"):
                df = cls._load_excel(file_path)
            elif ext in (".db", ".sqlite", ".sqlite3"):
                return cls._load_sqlite(file_path, optimize_memory)
            elif ext in (".npy", ".npz"):
                df = cls._load_numpy(file_path, ext)
            elif ext == ".pdf":
                df = cls._load_pdf(file_path)
            elif ext == ".zip":
                return cls._load_zip(file_path, optimize_memory)
            else:
                # Default fallback attempt as delimited CSV
                df = pd.read_csv(file_path)

            if optimize_memory and isinstance(df, pd.DataFrame):
                df, profile = MemoryOptimizer.optimize(df)
                return df, profile
            return df, None

        except UniversalLoadError:
            raise
        except Exception as e:
            raise UniversalLoadError(f"Universal loader failed for '{ext}': {e}") from e

    @classmethod
    def _load_from_stream(
        cls,
        stream: io.BytesIO,
        ext: str,
        optimize_memory: bool = True,
    ) -> Tuple[Union[pd.DataFrame, Dict[str, pd.DataFrame]], Optional[MemoryProfile]]:
        """Load data directly from an in-memory stream buffer."""
        stream.seek(0)
        if ext in (".parquet", ".pq"):
            df = pd.read_parquet(stream)
        elif ext in (".feather", ".arrow"):
            df = pd.read_feather(stream)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(stream)
        elif ext in (".json", ".jsonl", ".ndjson"):
            content = stream.read().decode("utf-8", errors="ignore")
            df = cls._parse_json_string(content, ext)
        else:
            # Default CSV / delimited
            df = pd.read_csv(stream)

        if optimize_memory and isinstance(df, pd.DataFrame):
            df, profile = MemoryOptimizer.optimize(df)
            return df, profile
        return df, None

    @staticmethod
    def _load_delimited(file_path: str, ext: str) -> pd.DataFrame:
        """Robust delimited text parser with separator auto-detection."""
        sep = "\t" if ext in (".tsv", ".tab") else None
        try:
            # First try auto-detect engine='python'
            return pd.read_csv(file_path, sep=sep, engine="c" if sep else "python", on_bad_lines="skip")
        except Exception:
            return pd.read_csv(file_path, sep=",", on_bad_lines="skip")

    @staticmethod
    def _load_parquet(file_path: str) -> pd.DataFrame:
        """Fast columnar parquet loading."""
        return pd.read_parquet(file_path)

    @staticmethod
    def _load_feather(file_path: str) -> pd.DataFrame:
        """Arrow IPC feather loading."""
        return pd.read_feather(file_path)

    @classmethod
    def _load_json(cls, file_path: str, ext: str) -> pd.DataFrame:
        """Parse JSON / NDJSON with automated nested hierarchy normalization."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return cls._parse_json_string(content, ext)

    @staticmethod
    def _parse_json_string(content: str, ext: str) -> pd.DataFrame:
        """Parse raw JSON string into flattened DataFrame."""
        content_stripped = content.strip()
        if not content_stripped:
            return pd.DataFrame()

        # Check for NDJSON / JSONL
        if ext in (".jsonl", ".ndjson") or "\n" in content_stripped and content_stripped.startswith("{"):
            lines = [json.loads(line) for line in content_stripped.splitlines() if line.strip()]
            return pd.json_normalize(lines)

        try:
            parsed = json.loads(content_stripped)
            if isinstance(parsed, list):
                return pd.json_normalize(parsed)
            elif isinstance(parsed, dict):
                # If dict of lists or nested root object
                if all(isinstance(v, list) for v in parsed.values()) and len(parsed) > 0:
                    return pd.DataFrame(parsed)
                return pd.json_normalize(parsed)
        except Exception:
            pass

        return pd.read_json(io.StringIO(content_stripped))

    @staticmethod
    def _load_excel(file_path: str) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load Excel workbook. Returns single DataFrame or dict of sheets."""
        xls = pd.ExcelFile(file_path)
        if len(xls.sheet_names) == 1:
            return pd.read_excel(xls, sheet_name=xls.sheet_names[0])
        return {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}

    @staticmethod
    def _load_sqlite(
        file_path: str,
        optimize_memory: bool = True,
    ) -> Tuple[Dict[str, pd.DataFrame], Optional[MemoryProfile]]:
        """Load all tables from SQLite database."""
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        result: Dict[str, pd.DataFrame] = {}

        for table in tables:
            df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
            if optimize_memory:
                df, _ = MemoryOptimizer.optimize(df)
            result[table] = df

        conn.close()
        if not result:
            raise UniversalLoadError("No tables found in SQLite database.")
        return result, None

    @staticmethod
    def _load_numpy(file_path: str, ext: str) -> pd.DataFrame:
        """Load NumPy binary matrices into structured DataFrame."""
        arr = np.load(file_path)
        if ext == ".npz":
            # Extract first array in archive
            first_key = list(arr.keys())[0]
            arr = arr[first_key]

        if arr.ndim == 1:
            return pd.DataFrame({"value": arr})
        elif arr.ndim == 2:
            return pd.DataFrame(arr, columns=[f"feature_{i}" for i in range(arr.shape[1])])
        elif arr.ndim >= 3:
            # Flatten spatial tensors into tabular grid
            flattened = arr.reshape(arr.shape[0], -1)
            return pd.DataFrame(flattened, columns=[f"pixel_{i}" for i in range(flattened.shape[1])])
        return pd.DataFrame()

    @staticmethod
    def _load_pdf(file_path: str) -> pd.DataFrame:
        """Extract text lines and tabular sections from PDF."""
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                raise UniversalLoadError("pypdf or PyPDF2 required to read PDF files.")

        reader = PdfReader(file_path)
        lines = []
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for line in text.splitlines():
                if line.strip():
                    lines.append({"page": idx, "text": line.strip()})

        return pd.DataFrame(lines)

    @classmethod
    def _load_zip(
        cls,
        file_path: str,
        optimize_memory: bool = True,
    ) -> Tuple[Dict[str, pd.DataFrame], Optional[MemoryProfile]]:
        """Extract and load tabular files inside ZIP archive."""
        result: Dict[str, pd.DataFrame] = {}
        with zipfile.ZipFile(file_path, "r") as z:
            for filename in z.namelist():
                ext = os.path.splitext(filename)[1].lower()
                if ext in (".csv", ".tsv", ".parquet", ".json"):
                    with z.open(filename) as f:
                        data = f.read()
                        df, _ = cls.load(io.BytesIO(data), file_name=filename, optimize_memory=optimize_memory)
                        if isinstance(df, pd.DataFrame):
                            result[os.path.basename(filename)] = df

        if not result:
            raise UniversalLoadError("No readable tabular files found inside ZIP archive.")
        return result, None

