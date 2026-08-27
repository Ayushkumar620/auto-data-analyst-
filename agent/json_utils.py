"""
Universal JSON-Safe Sanitization & Serialization Module.

Ensures every JSON boundary across agents, orchestrators, schemas, and FastAPI
safely serializes non-standard floating point values (NaN, +Infinity, -Infinity),
numpy scalars, pandas NA/NaT, and nested data structures into compliant JSON (RFC 8259).

Rules:
- NaN -> None (serializes to null in JSON)
- +Infinity -> None (serializes to null in JSON)
- -Infinity -> None (serializes to null in JSON)
- numpy integers -> int
- numpy floats -> float (or None if NaN/Inf)
- numpy bool_ -> bool
- numpy ndarray -> list
- pandas Timestamp/Timedelta -> ISO-8601 string
- pandas NA/NaT -> None
- finite numbers -> preserved as numeric (not converted to strings)
"""
from __future__ import annotations

import collections.abc
import datetime
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from starlette.responses import JSONResponse


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize any Python, numpy, or pandas data structure
    so that it is 100% compliant with standard JSON (RFC 8259).
    """
    if obj is None:
        return None

    # Booleans must be checked before int (in Python bool is a subclass of int)
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)

    # Standard ints and numpy integer types
    if isinstance(obj, (int, np.integer)):
        return int(obj)

    # Standard floats and numpy floating types
    if isinstance(obj, (float, np.floating)):
        f_val = float(obj)
        if math.isnan(f_val) or math.isinf(f_val):
            return None
        return f_val

    # Pandas NA, NaT, or NaN-like singletons
    if obj is pd.NA or obj is pd.NaT:
        return None

    # Strings and bytes
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return str(obj)

    # Datetime / Date / Time / Timedelta
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        if pd.isna(obj):
            return None
        return str(obj)

    # Pandas Series and DataFrame
    if isinstance(obj, pd.DataFrame):
        return [sanitize_for_json(row) for row in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return [sanitize_for_json(v) for v in obj.tolist()]

    # Numpy Arrays
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(v) for v in obj.tolist()]

    # Dictionaries (including subclasses)
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}

    # Lists, Tuples, Sets, Deques, and other non-string Sequences
    if isinstance(obj, (list, tuple, set, frozenset, collections.abc.Sequence)):
        return [sanitize_for_json(item) for item in obj]

    # Pydantic v2 model_dump or v1 dict
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return sanitize_for_json(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return sanitize_for_json(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return sanitize_for_json(obj.dict())
        except Exception:
            pass

    # Scalar check with pd.isna
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        pass

    return str(obj)


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """
    Serialize any object into a strict, compliant JSON string with
    allow_nan=False, ensuring no invalid tokens (NaN, Infinity) are produced.
    """
    cleaned = sanitize_for_json(obj)
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("allow_nan", False)
    return json.dumps(cleaned, **kwargs)


class SafeJSONResponse(JSONResponse):
    """
    Starlette/FastAPI JSONResponse subclass that guarantees all NaN,
    Infinity, and non-serializable objects are sanitized to null before rendering.
    """
    def render(self, content: Any) -> bytes:
        cleaned = sanitize_for_json(content)
        return json.dumps(
            cleaned,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")