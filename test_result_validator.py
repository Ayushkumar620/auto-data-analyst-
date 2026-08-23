"""Tests for the Agent Result Validator (reliability task: validation)."""
from datetime import datetime

import pandas as pd
import pytest

from agent.base import BaseAgent
from agent.result_validator import ResultValidator, validate_agent_result
from agent.schemas import (
    AgentError,
    AgentResult,
    AgentStatus,
    ClaimType,
    Evidence,
    ErrorCategory,
    ValidationSeverity,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def make_result(
    status: AgentStatus = AgentStatus.COMPLETED,
    agent: str = "Test Agent",
    role: str = "test",
    confidence: float = 0.9,
    evidence=None,
    errors=None,
    finished_at=None,
    output=None,
    duration_ms: float = 1.0,
):
    started = datetime(2026, 1, 1, 12, 0, 0)
    return AgentResult(
        agent=agent,
        role=role,
        agent_id="t-1",
        status=status,
        started_at=started,
        finished_at=finished_at if finished_at is not None else datetime(2026, 1, 1, 12, 0, 1),
        duration_ms=duration_ms,
        output=output if output is not None else {"ok": True},
        evidence=evidence or [],
        confidence=confidence,
        errors=errors or [],
    )


def _fact_evidence(conf=0.9, claim_type=ClaimType.FACT, data_ref=None, method="pandas.DataFrame.describe()"):
    return Evidence(
        source="Test Agent",
        method=method,
        data_ref=data_ref if data_ref is not None else {"frame": "data",
                                                        "rows": 100,
                                                        "columns": 3,
                                                        "column_names": ["a", "b", "c"]},
        confidence=conf,
        claim_type=claim_type,
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def test_valid_completed_result_passes():
    r = make_result()
    vr = ResultValidator().validate(r)
    assert vr.passed is True
    assert r.validation is vr  # attached for downstream


def test_confidence_out_of_range_fails():
    r = make_result(confidence=1.5)
    vr = ResultValidator().validate(r)
    assert vr.passed is False
    codes = {i.code for i in vr.issues}
    assert "CONFIDENCE_OUT_OF_RANGE" in codes


def test_missing_agent_name_fails():
    r = make_result(agent="  ")
    vr = ResultValidator().validate(r)
    assert vr.passed is False
    assert any(i.code == "MISSING_AGENT_NAME" for i in vr.issues)


def test_error_status_without_error_object_passes_but_warns():
    r = make_result(status=AgentStatus.ERROR, errors=[])
    vr = ResultValidator().validate(r)
    # ERROR without errors is a warning, not a hard failure
    assert vr.passed is True
    assert any(i.code == "ERROR_WITHOUT_ERRORS" for i in vr.issues)


def test_wrong_input_type_raises():
    with pytest.raises(TypeError):
        ResultValidator().validate({"status": "completed"})


# ---------------------------------------------------------------------------
# Evidence validation
# ---------------------------------------------------------------------------
def test_evidence_with_bad_confidence_fails():
    r = make_result(evidence=[_fact_evidence(conf=3.0)])
    vr = ResultValidator().validate(r)
    assert vr.passed is False
    assert any(i.code == "EVIDENCE_CONFIDENCE_RANGE" for i in vr.issues)


def test_evidence_with_invalid_claim_type_fails():
    ev = _fact_evidence()
    ev.claim_type = "not-a-claim-type"  # type: ignore[assignment]
    r = make_result(evidence=[ev])
    vr = ResultValidator().validate(r)
    assert vr.passed is False
    assert any(i.code == "INVALID_CLAIM_TYPE" for i in vr.issues)


# ---------------------------------------------------------------------------
# Claim integrity: correlation must never claim causation
# ---------------------------------------------------------------------------
def test_correlation_evidence_claiming_causation_fails():
    ev = Evidence(
        source="Insight Agent",
        method="pearson_correlation",
        data_ref={"left": "discount", "right": "profit"},
        confidence=0.9,
        claim_type=ClaimType.CORRELATION,
        metadata={"interpretation": "Higher discounting causes lower profit."},
    )
    r = make_result(evidence=[ev])
    vr = ResultValidator().validate(r)
    assert vr.passed is False
    assert any(i.code == "CORRELATION_AS_CAUSATION" for i in vr.issues)


def test_correlation_evidence_without_causation_passes():
    ev = Evidence(
        source="Insight Agent",
        method="correlation_pearson",
        data_ref={"left": "discount", "right": "profit"},
        confidence=0.9,
        claim_type=ClaimType.CORRELATION,
        metadata={"interpretation": "Discount and profit show a strong negative "
                                   "association in the sample."},
        )
    r = make_result(evidence=[ev])
    vr = ResultValidator().validate(r)
    assert vr.passed is True


# ---------------------------------------------------------------------------
# Data cross-checking
# ---------------------------------------------------------------------------
def test_cross_check_unknown_column_fails():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    ev = _fact_evidence(data_ref={"frame": "data", "column_names": ["a", "ghost"]})
    r = make_result(evidence=[ev])
    vr = ResultValidator().validate(r, {"dataframe": df})
    assert vr.passed is False
    assert any(i.code == "EVIDENCE_UNKNOWN_COLUMN" for i in vr.issues)


def test_cross_check_row_count_exceeded_warns():
    df = pd.DataFrame({"a": [1, 2]})  # 2 rows
    ev = _fact_evidence(data_ref={"frame": "data", "rows": 42})
    r = make_result(evidence=[ev])
    vr = ResultValidator().validate(r, {"dataframe": df})
    assert any(i.code == "EVIDENCE_ROWS_EXCEEDED" for i in vr.issues)


def test_cross_check_recomputes_null_statistics():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, None]})  # 2 nulls total
    ev = Evidence(
        source="Analysis Agent",
        method="pandas.DataFrame.isnull().sum()",
        data_ref={"frame": "data"},
        confidence=0.99,
        claim_type=ClaimType.FACT,
        raw_value=2.0,  # matches the real null count
    )
    r = make_result(evidence=[ev])
    vr = ResultValidator().validate(r, {"dataframe": df})
    assert vr.passed is True

    bad = Evidence(
        source="Analysis Agent",
        method="pandas.DataFrame.isnull().sum()",
        data_ref={"frame": "data"},
        confidence=0.99,
        claim_type=ClaimType.FACT,
        raw_value=99.0,  # does not match the real null count
    )
    r2 = make_result(evidence=[bad])
    vr2 = ResultValidator().validate(r2, {"dataframe": df})
    assert vr2.passed is False
    assert any(i.code == "CALCULATION_MISMATCH" for i in vr2.issues)


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------
def test_repair_clamps_confidence_and_marks_repaired():
    r = make_result(confidence=2.0)
    repaired, vr = ResultValidator().repair(r)
    assert repaired.confidence == 1.0
    assert vr.repaired is True
    assert any("Clamped" in a for a in vr.repair_actions)


def test_repair_drops_broken_evidence():
    r = make_result(evidence=[_fact_evidence(conf=7.0), _fact_evidence(conf=0.8)])
    repaired, vr = ResultValidator().repair(r)
    assert len(repaired.evidence) == 1
    assert repaired.evidence[0].confidence == 0.8
    assert vr.repaired is True


def test_repair_drops_hallucinated_column_evidence():
    df = pd.DataFrame({"a": [1, 2]})
    ev = _fact_evidence(data_ref={"frame": "data", "column_names": ["a", "ghost"]})
    r = make_result(evidence=[ev])
    repaired, vr = ResultValidator().repair(r, {"dataframe": df})
    assert len(repaired.evidence) == 0
    assert vr.repaired is True


def test_repair_stamps_missing_finished_at():
    r = make_result(finished_at=None)
    repaired, vr = ResultValidator().repair(r)
    assert repaired.finished_at is not None
    assert vr.repaired is True


def test_repair_noop_on_clean_result():
    r = make_result()
    repaired, vr = ResultValidator().repair(r)
    assert vr.repaired is False
    assert repaired.evidence == []


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def test_validate_agent_result_convenience_returns_validation():
    r = make_result()
    vr = validate_agent_result(r)
    assert vr.passed is True
    assert r.validation is vr


# ---------------------------------------------------------------------------
# End-to-end with a real BaseAgent + real data
# ---------------------------------------------------------------------------
def test_real_agent_output_passes_validation():
    from agent.loader import load_data

    data = load_data("sample_data.csv")

    class _MiniAgent(BaseAgent):
        name = "Mini Analysis Agent"
        role = "analysis"

        def run(self, task):
            self._start()
            return self._finish(
                {"request": "summary", "reports": []},
                evidence=[self.make_evidence(
                    method="pandas.DataFrame.describe(include='all')",
                    data_ref={"frame": "data",
                              "rows": len(data),
                              "columns": len(data.columns)},
                    confidence=0.95,
                    claim_type=ClaimType.FACT,
                )],
            )

    result = _MiniAgent().run({})
    vr = ResultValidator().validate(result, {"dataframe": data})
    assert vr.passed is True


def test_real_agent_output_with_bad_context_fails_and_repairs():
    from agent.loader import load_data

    data = load_data("sample_data.csv")

    class _BadAgent(BaseAgent):
        name = "Bad Agent"
        role = "analysis"

        def run(self, task):
            self._start()
            return self._finish(
                {"request": "summary"},
                evidence=[self.make_evidence(
                    method="pandas.DataFrame.describe()",
                    data_ref={"frame": "data", "column_names": ["a", "ghost_col"]},
                    confidence=0.9,
                    claim_type=ClaimType.FACT,
                )],
            )

    result = _BadAgent().run({})
    vr = ResultValidator().validate(result, {"dataframe": data})
    assert vr.passed is False

    repaired, after = ResultValidator().repair(result, {"dataframe": data})
    assert len(repaired.evidence) == 0
    assert after.repaired is True
    assert after.passed is True