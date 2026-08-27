"""
Regression Tests: JSON-Safe Serialization of NaN / Infinity / -Infinity
across all API and Agent Boundaries.

Tests:
  A. NaN in uploaded dataset
  B. Infinity in uploaded dataset
  C. -Infinity in uploaded dataset
  D. Nested NaN values
  E. Missing categorical values
  F. Missing numeric values
  G. Ordinary finite values remain unchanged
  H. AgentResult containing NaN
  I. Metrics containing NaN
  J. Evidence containing NaN
  K. FastAPI endpoint with missing values (using TestClient)
  L. Complete upload flow + strict JSON validation
"""
from __future__ import annotations

import json
import math
import io
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.json_utils import sanitize_for_json, safe_json_dumps
from agent.agent_result import AgentResult, AgentStatus, Evidence, ClaimType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strict_json_ok(text: str) -> bool:
    """Return True if text is 100% standard JSON (no NaN/Infinity tokens)."""
    try:
        def _no_nan(val):
            raise ValueError(f"Non-finite JSON token: {val}")
        json.loads(text, parse_constant=_no_nan)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# A–C: sanitize_for_json on scalar special floats
# ---------------------------------------------------------------------------

def test_A_nan_sanitized_to_none():
    """A. NaN -> None."""
    assert sanitize_for_json(float("nan")) is None
    assert sanitize_for_json(np.nan) is None
    assert sanitize_for_json(np.float32("nan")) is None


def test_B_infinity_sanitized_to_none():
    """B. Infinity -> None."""
    assert sanitize_for_json(float("inf")) is None
    assert sanitize_for_json(np.inf) is None
    assert sanitize_for_json(np.float64("inf")) is None


def test_C_neg_infinity_sanitized_to_none():
    """C. -Infinity -> None."""
    assert sanitize_for_json(float("-inf")) is None
    assert sanitize_for_json(-np.inf) is None


# ---------------------------------------------------------------------------
# D: Nested NaN values
# ---------------------------------------------------------------------------

def test_D_nested_nan_sanitized():
    """D. Nested NaN values in dict/list -> None."""
    nested = {
        "level1": {
            "value": float("nan"),
            "list": [1.0, float("inf"), float("-inf"), np.nan, 3.0],
        },
        "other": "text",
    }
    result = sanitize_for_json(nested)
    assert result["level1"]["value"] is None
    assert result["level1"]["list"] == [1.0, None, None, None, 3.0]
    assert result["other"] == "text"


# ---------------------------------------------------------------------------
# E–F: Missing categorical and numeric values
# ---------------------------------------------------------------------------

def test_E_missing_categorical_sanitized():
    """E. Missing categorical values (None/pd.NA) sanitized to None."""
    series = pd.Categorical(["a", None, "b", float("nan")])
    result = sanitize_for_json(list(series))
    assert None in result
    assert "a" in result
    assert "b" in result


def test_F_missing_numeric_sanitized():
    """F. Missing numeric values (NaN) sanitized to None."""
    df = pd.DataFrame({"score": [1.0, float("nan"), 3.5, float("nan")]})
    records = df.to_dict(orient="records")
    result = sanitize_for_json(records)
    assert result[0]["score"] == 1.0
    assert result[1]["score"] is None
    assert result[2]["score"] == 3.5
    assert result[3]["score"] is None


# ---------------------------------------------------------------------------
# G: Ordinary finite numeric values remain unchanged
# ---------------------------------------------------------------------------

def test_G_finite_values_unchanged():
    """G. Ordinary finite numbers must stay as numbers."""
    payload = {
        "int_val": 42,
        "float_val": 3.14,
        "zero": 0,
        "neg": -7.5,
        "large": 1_000_000,
    }
    result = sanitize_for_json(payload)
    assert result["int_val"] == 42
    assert abs(result["float_val"] - 3.14) < 1e-10
    assert result["zero"] == 0
    assert result["neg"] == -7.5
    assert result["large"] == 1_000_000


# ---------------------------------------------------------------------------
# H: AgentResult containing NaN
# ---------------------------------------------------------------------------

def test_H_agent_result_nan_sanitized():
    """H. AgentResult.to_dict() must produce NaN-free dicts."""
    res = AgentResult(
        success=True,
        status=AgentStatus.SUCCESS,
        agent_name="TestAgent",
        result={"metric": float("nan"), "other": float("inf"), "valid": 42.0},
        metrics={"score": float("nan"), "accuracy": 0.95},
    )
    d = res.to_dict()
    assert d["result"]["metric"] is None
    assert d["result"]["other"] is None
    assert d["result"]["valid"] == 42.0
    assert d["metrics"]["score"] is None
    assert d["metrics"]["accuracy"] == 0.95
    # Must produce valid JSON
    assert _strict_json_ok(json.dumps(d))


# ---------------------------------------------------------------------------
# I: Metrics containing NaN
# ---------------------------------------------------------------------------

def test_I_metrics_nan_sanitized():
    """I. Metrics with NaN sanitized."""
    res = AgentResult(
        success=True,
        status=AgentStatus.SUCCESS,
        agent_name="MetricsAgent",
        metrics={"r2": float("nan"), "rmse": float("inf"), "mae": 0.5},
    )
    d = res.to_dict()
    assert d["metrics"]["r2"] is None
    assert d["metrics"]["rmse"] is None
    assert d["metrics"]["mae"] == 0.5


# ---------------------------------------------------------------------------
# J: Evidence containing NaN
# ---------------------------------------------------------------------------

def test_J_evidence_nan_sanitized():
    """J. Evidence.to_dict() must produce NaN-free output."""
    ev = Evidence(
        operation="compute_mean",
        result=float("nan"),
        raw_value=float("inf"),
        confidence=0.9,
        claim_type=ClaimType.OBSERVATION,
    )
    d = ev.to_dict()
    assert d["result"] is None
    assert d["raw_value"] is None
    assert _strict_json_ok(json.dumps(d))


# ---------------------------------------------------------------------------
# K: FastAPI endpoint with missing values
# ---------------------------------------------------------------------------

def test_K_fastapi_upload_endpoint_strict_json():
    """K. /api/v1/datasets/upload with missing-value dataset returns strict JSON."""
    from backend.app.main import app
    client = TestClient(app)

    # Build a minimal CSV with NaN/missing values
    csv_content = "id,score,category\n1,1.5,A\n2,,B\n3,3.0,\n"
    f = io.BytesIO(csv_content.encode())

    resp = client.post(
        "/api/v1/datasets/upload",
        files={"file": ("test_missing.csv", f, "text/csv")},
    )
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    nan_idx = resp.text.find("NaN")
    if nan_idx != -1:
        snippet = resp.text[max(0, nan_idx - 30): nan_idx + 30]
        assert False, f"Response contains non-JSON floats (NaN/Infinity). Snippet: {snippet}"
    assert _strict_json_ok(resp.text), f"Response is not strict JSON: {resp.text[:200]}"
# ---------------------------------------------------------------------------
# K2: Flask /api/analyze (templates/index.html) with missing values.
# The standalone web UI calls this endpoint via fetch() + resp.json(), whose
# underlying JSON.parse rejects a literal `NaN` token. Missing cells in a
# `head` result like ``sample_notes`` were leaking as ``NaN`` before the app
# was configured with a sanitizing JSON provider.
# ---------------------------------------------------------------------------

def test_M_flask_analyze_head_strict_json():
    """M. /api/analyze 'head' command on a dataset with missing cells must
    never emit NaN/Infinity tokens (must be strict RFC 8259 JSON)."""
    from app import app

    csv_content = "id,sample_notes\n1,hello\n2,\n3,world\n"
    f = io.BytesIO(csv_content.encode())

    client = app.test_client()
    resp = client.post(
        "/api/analyze",
        data={"file": (f, "missing_notes.csv"), "command": "head"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200, f"Analyze failed: {resp.text[:300]}"
    body = resp.get_data(as_text=True)

    # A literal NaN / Infinity token would crash the browser JSON.parse.
    for token in ("NaN", "Infinity", "-Infinity"):
        idx = body.find(token)
        assert idx == -1, (
            f"Response contains non-JSON float token `{token}`. "
            f"Snippet: {body[max(0, idx - 40): idx + 40]}"
        )
    assert _strict_json_ok(body), f"Response is not strict JSON: {body[:200]}"

    # The missing cell must decode as null (not NaN).
    parsed = json.loads(body)
    assert parsed.get("type") == "head", parsed.get("type")
    reports = parsed.get("reports") or []
    assert reports, "Expected head reports in response"
    missing = reports[0]["rows"][1]["sample_notes"] if reports[0].get("rows") else None
    assert missing is None, f"Missing cell should be null, got {missing!r}"


# ---------------------------------------------------------------------------
# L: Complete upload flow with auto_data_analyst_reliability_sales.csv
# ---------------------------------------------------------------------------

def test_L_reliability_sales_csv_strict_json():
    """L. Full upload flow with the real reliability dataset must produce strict JSON."""
    from backend.app.main import app
    client = TestClient(app)

    with open("uploads/auto_data_analyst_reliability_sales.csv", "rb") as f:
        resp = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("auto_data_analyst_reliability_sales.csv", f, "text/csv")},
        )

    assert resp.status_code == 200, f"Upload failed: {resp.text[:300]}"
    nan_idx = resp.text.find("NaN")
    if nan_idx != -1:
        snippet = resp.text[max(0, nan_idx - 30): nan_idx + 30]
        assert False, f"Reliability sales CSV contains non-JSON floats. Snippet: {snippet}"
    assert _strict_json_ok(resp.text), f"Response is not strict JSON: {resp.text[:200]}"


# ---------------------------------------------------------------------------
# Bonus: safe_json_dumps never produces NaN/Infinity tokens
# ---------------------------------------------------------------------------

def test_safe_json_dumps_produces_strict_json():
    """safe_json_dumps must always emit RFC 8259-compliant JSON."""
    payload = {
        "a": float("nan"),
        "b": float("inf"),
        "c": float("-inf"),
        "d": {"nested": np.nan},
        "e": [1, float("nan"), 3],
        "f": 42.0,
    }
    s = safe_json_dumps(payload)
    assert _strict_json_ok(s)
    parsed = json.loads(s)
    assert parsed["a"] is None
    assert parsed["b"] is None
    assert parsed["c"] is None
    assert parsed["d"]["nested"] is None
    assert parsed["e"] == [1, None, 3]
    assert parsed["f"] == 42.0


def test_numpy_types_sanitized():
    """numpy integer, float, bool types are properly converted."""
    result = sanitize_for_json({
        "np_int": np.int32(5),
        "np_float": np.float32(3.14),
        "np_bool": np.bool_(True),
        "np_nan": np.float64("nan"),
        "np_inf": np.float64("inf"),
    })
    assert result["np_int"] == 5
    assert isinstance(result["np_int"], int)
    assert abs(result["np_float"] - 3.14) < 0.01
    assert result["np_bool"] is True
    assert result["np_nan"] is None
    assert result["np_inf"] is None
    assert _strict_json_ok(json.dumps(result))