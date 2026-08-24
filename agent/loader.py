"""
Data Loader - Handles loading of various data formats.
Supports: CSV, TSV, Parquet, Arrow, Feather, Excel, JSON, NDJSON, TXT, PDF, SQL (SQLite), NumPy, ZIP, and video.
"""
import os
import json
import sqlite3
import re
import pandas as pd

from backend.app.core.universal_loader import UniversalDatasetLoader, UniversalLoadError
from backend.app.core.big_data_engine import MemoryOptimizer


class DataLoadError(Exception):
    """Raised when data cannot be loaded."""
    pass


class DataLoader:
    """Loads data from various file formats into a pandas-friendly structure."""

    SUPPORTED_EXTS = {
        ".csv": "CSV",
        ".tsv": "TSV",
        ".tab": "TSV",
        ".parquet": "Parquet",
        ".pq": "Parquet",
        ".feather": "Feather",
        ".arrow": "Arrow",
        ".xlsx": "Excel",
        ".xls": "Excel",
        ".ods": "Excel",
        ".json": "JSON",
        ".jsonl": "JSON Lines",
        ".ndjson": "Newline Delimited JSON",
        ".txt": "Text",
        ".pdf": "PDF",
        ".db": "SQLite",
        ".sqlite": "SQLite",
        ".sqlite3": "SQLite",
        ".npy": "NumPy",
        ".npz": "NumPy",
        ".zip": "ZIP",
        ".mp4": "Video",
        ".avi": "Video",
        ".mov": "Video",
        ".mkv": "Video",
    }

    def __init__(self, file_path):
        self.file_path = file_path
        self.ext = os.path.splitext(file_path)[1].lower()

    def load(self):
        """Dispatch to the appropriate loader based on file extension."""
        if self.ext not in self.SUPPORTED_EXTS:
            raise DataLoadError(
                f"Unsupported file type '{self.ext}'. Supported types: "
                f"{', '.join(sorted(self.SUPPORTED_EXTS))}"
            )

        if self.ext in (".mp4", ".avi", ".mov", ".mkv"):
            return self._load_video()
        elif self.ext == ".txt":
            return self._load_text()
        elif self.ext == ".pdf":
            return self._load_pdf()
        elif self.ext in (".db", ".sqlite", ".sqlite3"):
            return self._load_sqlite()
        elif self.ext in (".csv", ".tsv", ".tab"):
            return self._load_csv()
        elif self.ext in (".xlsx", ".xls", ".ods"):
            return self._load_excel()
        elif self.ext in (".json", ".jsonl", ".ndjson"):
            return self._load_json()
        elif self.ext in (".parquet", ".pq", ".feather", ".arrow", ".npy", ".npz", ".zip"):
            try:
                res, _ = UniversalDatasetLoader.load(self.file_path)
                return res
            except Exception as e:
                raise DataLoadError(f"Failed to load {self.ext}: {e}")

        return None

    def _load_csv(self):
        try:
            return pd.read_csv(self.file_path)
        except Exception as e:
            raise DataLoadError(f"Failed to read CSV: {e}")

    def _load_excel(self):
        try:
            return pd.read_excel(self.file_path)
        except Exception as e:
            raise DataLoadError(f"Failed to read Excel: {e}")

    def _load_json(self):
        try:
            return UniversalDatasetLoader._load_json(self.file_path, self.ext)
        except Exception as e:
            raise DataLoadError(f"Failed to read JSON: {e}")

    def _load_text(self):
        """Load a text file. If it's a single column of numbers, return numeric series."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            if not lines:
                raise DataLoadError("Text file is empty.")
            # Try to parse as a single-column dataframe
            try:
                numbers = [float(x) for x in lines]
                return pd.DataFrame({"value": numbers})
            except ValueError:
                # Fall back to raw text lines
                return pd.DataFrame({"line": lines})
        except DataLoadError:
            raise
        except Exception as e:
            raise DataLoadError(f"Failed to read text file: {e}")

    def _load_pdf(self):
        """Extract text from a PDF file and return it as a DataFrame."""
        try:
            # Try PyPDF2 first, then fall back to pypdf
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                from pypdf import PdfReader

            reader = PdfReader(self.file_path)
            pages_text = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages_text.append(text)

            if not pages_text or not any(t.strip() for t in pages_text):
                raise DataLoadError("No extractable text found in PDF.")

            # Split text into lines and build a DataFrame
            lines = []
            for idx, text in enumerate(pages_text, start=1):
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped:
                        lines.append({"page": idx, "text": stripped})

            if not lines:
                raise DataLoadError("No text lines extracted from PDF.")

            df = pd.DataFrame(lines)

            # Try to detect tabular data: if lines contain delimiters, parse as table
            delimiters = [",", "\t", ";", "|"]
            for delim in delimiters:
                split_counts = [len(re.split(re.escape(delim), line)) for line in df["text"].head(20)]
                if split_counts and max(split_counts) >= 2 and len(set(split_counts)) == 1:
                    parsed = df["text"].str.split(delim, expand=True)
                    parsed.columns = [f"col_{i}" for i in range(parsed.shape[1])]
                    return parsed

            return df
        except DataLoadError:
            raise
        except Exception as e:
            raise DataLoadError(f"Failed to read PDF: {e}")

    def _load_sqlite(self):
        """Load all tables from a SQLite database into a dict of DataFrames."""
        try:
            conn = sqlite3.connect(self.file_path)
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = pd.read_sql_query(query, conn)
            result = {}
            for table in tables["name"]:
                result[table] = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
            conn.close()
            if not result:
                raise DataLoadError("No tables found in SQLite database.")
            return result
        except DataLoadError:
            raise
        except Exception as e:
            raise DataLoadError(f"Failed to read SQLite database: {e}")

    def _load_video(self):
        """Extract metadata and frames from a video file for basic analysis."""
        try:
            import cv2
            cap = cv2.VideoCapture(self.file_path)
            if not cap.isOpened():
                raise DataLoadError("Could not open video file.")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            # Extract a few sample frame brightness values
            sample_brightness = []
            for i in range(0, frame_count, max(1, frame_count // 10)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sample_brightness.append(float(gray.mean()))
            cap.release()

            meta = {
                "File": self.file_path,
                "Frame Count": frame_count,
                "FPS": round(fps, 2),
                "Width": width,
                "Height": height,
                "Duration (sec)": round(duration, 2),
            }
            video_info = pd.DataFrame([meta])
            if sample_brightness:
                brightness_df = pd.DataFrame(
                    {
                        "frame_index": list(range(len(sample_brightness))),
                        "brightness": sample_brightness,
                    }
                )
                return {"metadata": video_info, "brightness_samples": brightness_df}
            return {"metadata": video_info}
        except DataLoadError:
            raise
        except Exception as e:
            raise DataLoadError(f"Failed to process video: {e}")


def load_data(file_path):
    """Convenience function to load data from a file."""
    loader = DataLoader(file_path)
    return loader.load()
