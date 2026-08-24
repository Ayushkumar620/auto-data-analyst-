"""Comprehensive test suite for Phase 1: Universal Agent Protocol & Standardized AgentResult Unification."""
from datetime import datetime
import pandas as pd
import pytest

from agent.schemas import (
    AgentResult,
    AgentError,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
    SemanticMapping,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from agent.base import BaseAgent
from agent.result_validator import ResultValidator, validate_agent_result
from agent.agents import (
    DataLoadingAgent,
    AnalysisAgent,
    VisualizationAgent,
    PredictionAgent,
    ForecastAgent,
    CleaningAgent,
    InsightAgent,
    ReportAgent,
)
from agent.planner import PlannerAgent


# ==============================================================================
# 1. AgentResult & Schema Tests
# ==============================================================================

def test_agent_result_success_factory():
    """Test creating an AgentResult via the success factory."""
    now = datetime.now()
    ev = Evidence(
        source="TestAgent",
        method="pandas.describe",
        data_ref={"rows": 10},
        confidence=0.95,
        claim_type=ClaimType.FACT,
    )
    result = AgentResult.success(
        agent="TestAgent",
        role="tester",
        agent_id="test_001",
        started_at=now,
        output={"key": "value"},
        evidence=[ev],
        confidence=0.95,
        duration_ms=12.5,
        warnings=["Test warning"],
        metadata={"run_env": "test"},
    )

    assert result.agent == "TestAgent"
    assert result.role == "tester"
    assert result.status == AgentStatus.COMPLETED
    assert result.is_success is True
    assert result.is_error is False
    assert result.has_evidence is True
    assert result.has_errors is False
    assert result.confidence == 0.95
    assert result.output == {"key": "value"}
    assert len(result.warnings) == 1
    assert result.metadata["run_env"] == "test"


def test_agent_result_failure_factory():
    """Test creating an AgentResult via the failure factory."""
    now = datetime.now()
    err = AgentError(
        category=ErrorCategory.COMPUTATION,
        message="Divide by zero error",
        recoverable=False,
    )
    result = AgentResult.failure(
        agent="FailingAgent",
        role="tester",
        agent_id="fail_001",
        started_at=now,
        errors=[err],
        duration_ms=5.0,
        output={"error": "Divide by zero error"},
    )

    assert result.status == AgentStatus.ERROR
    assert result.is_success is False
    assert result.is_error is True
    assert result.has_errors is True
    assert result.confidence == 0.0
    assert len(result.errors) == 1
    assert result.errors[0].message == "Divide by zero error"


def test_agent_result_dict_backward_compatibility():
    """Test that AgentResult allows dictionary-like subscripting and .get() access."""
    now = datetime.now()
    result = AgentResult.success(
        agent="LegacyAccessAgent",
        role="tester",
        agent_id="leg_001",
        started_at=now,
        output={"metric_value": 42},
    )

    assert result["agent"] == "LegacyAccessAgent"
    assert result["status"] == "completed"
    assert result["output"] == {"metric_value": 42}
    assert result.get("agent") == "LegacyAccessAgent"
    assert result.get("status") == "completed"
    assert result.get("non_existent", "default_val") == "default_val"
    assert "agent" in result
    assert "status" in result


def test_agent_result_serialization_and_deserialization():
    """Test that AgentResult roundtrips through to_dict() and from_dict()."""
    now = datetime.now()
    ev = Evidence(
        source="SerializerAgent",
        method="aggregation",
        data_ref={"column": "sales"},
        confidence=0.88,
        claim_type=ClaimType.OBSERVATION,
        raw_value=123.45,
    )
    err = AgentError(
        category=ErrorCategory.DATA_QUALITY,
        message="Low row count warning",
        recoverable=True,
    )
    val = ValidationResult(passed=True, repaired=True, repair_actions=["Clamped confidence"])

    original = AgentResult(
        agent="SerializerAgent",
        role="profiler",
        agent_id="ser_001",
        status=AgentStatus.COMPLETED,
        started_at=now,
        finished_at=now,
        duration_ms=45.2,
        output={"mean_sales": 123.45},
        evidence=[ev],
        confidence=0.88,
        validation=val,
        errors=[err],
        warnings=["Low data rows"],
        metadata={"version": "1.0"},
        retry_count=1,
    )

    d = original.to_dict()
    assert isinstance(d, dict)
    assert d["agent"] == "SerializerAgent"
    assert d["status"] == "completed"
    assert len(d["evidence"]) == 1
    assert d["evidence"][0]["claim_type"] == "observation"
    assert d["validation"]["repaired"] is True

    reconstituted = AgentResult.from_dict(d)
    assert reconstituted.agent == original.agent
    assert reconstituted.role == original.role
    assert reconstituted.status == AgentStatus.COMPLETED
    assert reconstituted.is_success is True
    assert reconstituted.confidence == 0.88
    assert len(reconstituted.evidence) == 1
    assert reconstituted.evidence[0].claim_type == ClaimType.OBSERVATION
    assert reconstituted.evidence[0].raw_value == 123.45
    assert len(reconstituted.errors) == 1
    assert reconstituted.errors[0].category == ErrorCategory.DATA_QUALITY
    assert reconstituted.validation.passed is True
    assert reconstituted.validation.repaired is True


# ==============================================================================
# 2. BaseAgent Lifecycle & Retry Tests
# ==============================================================================

class DummySuccessAgent(BaseAgent):
    name = "Dummy Success Agent"
    role = "tester"

    def run(self, task):
        self._start()
        ev = [self.make_evidence("dummy_compute", {"test": True}, confidence=1.0, claim_type=ClaimType.FACT)]
        return self._finish({"computed": 100}, evidence=ev, confidence=1.0)


class DummyFailingThenSucceedingAgent(BaseAgent):
    name = "Flaky Agent"
    role = "tester"

    def __init__(self):
        super().__init__()
        self.call_count = 0

    def run(self, task):
        self._start()
        self.call_count += 1
        if self.call_count < 2:
            return self._error("Transient failure", category=ErrorCategory.COMPUTATION, recoverable=True)
        return self._finish({"recovered": True}, confidence=0.9)


class DummyFatalAgent(BaseAgent):
    name = "Fatal Agent"
    role = "tester"

    def run(self, task):
        self._start()
        return self._error("Unrecoverable syntax error", category=ErrorCategory.INPUT_VALIDATION, recoverable=False)


def test_base_agent_lifecycle_success():
    """Test normal successful lifecycle execution."""
    agent = DummySuccessAgent()
    assert agent.status == AgentStatus.IDLE
    result = agent.run({"param": 1})
    assert result.is_success is True
    assert agent.status == AgentStatus.COMPLETED
    assert len(result.evidence) == 1
    assert result.evidence[0].source == "Dummy Success Agent"


def test_base_agent_retry_recovers_transient_error():
    """Test execute_with_retry recovers from transient recoverable errors."""
    agent = DummyFailingThenSucceedingAgent()
    result = agent.execute_with_retry({"param": 1}, max_retries=2, retry_delay_ms=10)
    assert result.is_success is True
    assert result.output == {"recovered": True}
    assert agent.call_count == 2
    assert result.retry_count == 1


def test_base_agent_retry_aborts_on_unrecoverable_error():
    """Test execute_with_retry does not waste attempts on unrecoverable errors."""
    agent = DummyFatalAgent()
    result = agent.execute_with_retry({"param": 1}, max_retries=3, retry_delay_ms=10)
    assert result.is_error is True
    assert result.errors[0].recoverable is False


# ==============================================================================
# 3. ResultValidator & Claim Integrity Tests
# ==============================================================================

def test_validator_blocks_correlation_as_causation():
    """Test that ResultValidator catches correlation claims containing causal verbs."""
    validator = ResultValidator()
    now = datetime.now()
    bad_evidence = Evidence(
        source="CausalityViolatingAgent",
        method="pearson_correlation",
        data_ref={"column": "sales", "with": "marketing"},
        confidence=0.85,
        claim_type=ClaimType.CORRELATION,
        metadata={"description": "High marketing causes increased sales significantly."},
    )
    result = AgentResult.success(
        agent="CausalityViolatingAgent",
        role="tester",
        agent_id="caus_001",
        started_at=now,
        output={"corr": 0.85},
        evidence=[bad_evidence],
    )

    vr = validator.validate(result)
    assert vr.passed is False
    assert any(issue.code == "CORRELATION_AS_CAUSATION" for issue in vr.issues)


def test_validator_cross_check_detects_unknown_columns():
    """Test that ResultValidator verifies column references against context dataframe."""
    df = pd.DataFrame({"actual_col_a": [1, 2, 3], "actual_col_b": [4, 5, 6]})
    validator = ResultValidator()
    now = datetime.now()

    fake_evidence = Evidence(
        source="HallucinatingAgent",
        method="column_analysis",
        data_ref={"column": "non_existent_column"},
        confidence=0.9,
        claim_type=ClaimType.FACT,
    )
    result = AgentResult.success(
        agent="HallucinatingAgent",
        role="tester",
        agent_id="hall_001",
        started_at=now,
        output={"stat": 10},
        evidence=[fake_evidence],
    )

    context = {"dataframe": df, "columns": list(df.columns), "row_count": len(df)}
    vr = validator.validate(result, context=context)
    assert vr.passed is False
    assert any(issue.code == "EVIDENCE_UNKNOWN_COLUMN" for issue in vr.issues)


def test_validator_repair_clamps_out_of_range_confidence():
    """Test that ResultValidator repair safely clamps out-of-range confidence values."""
    validator = ResultValidator()
    now = datetime.now()
    result = AgentResult(
        agent="BadConfidenceAgent",
        role="tester",
        agent_id="conf_001",
        status=AgentStatus.COMPLETED,
        started_at=now,
        finished_at=now,
        output={"metric": 1},
        confidence=1.5,  # Out of [0, 1] range
    )

    repaired_res, vr = validator.repair(result)
    assert vr.passed is True
    assert vr.repaired is True
    assert repaired_res.confidence == 1.0


# ==============================================================================
# 4. Specialized Agents Conformance Tests
# ==============================================================================

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "sales": [100.0, 110.0, 105.0, 115.0, 120.0, 130.0, 125.0, 140.0, 135.0, 150.0],
        "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    })


def test_data_loading_agent_conformance(sample_df):
    agent = DataLoadingAgent()
    result = agent.run({"data": sample_df, "source": "sample_df"})
    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.has_evidence is True
    assert result.evidence[0].claim_type == ClaimType.FACT


def test_analysis_agent_conformance(sample_df):
    agent = AnalysisAgent()
    result = agent.run({"data": sample_df, "request": "summary"})
    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.has_evidence is True
    assert "reports" in result.output


def test_cleaning_agent_conformance(sample_df):
    agent = CleaningAgent()
    result = agent.run({"data": sample_df})
    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.has_evidence is True
    assert "reports" in result.output


def test_prediction_agent_conformance(sample_df):
    agent = PredictionAgent()
    result = agent.run({"data": sample_df, "target": "sales"})
    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert "metric" in result.output


def test_forecast_agent_conformance(sample_df):
    agent = ForecastAgent()
    result = agent.run({"data": sample_df, "target": "sales", "periods": 3})
    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.evidence[0].claim_type == ClaimType.INFERENCE
    assert "forecast" in result.output


def test_visualization_agent_conformance(sample_df):
    agent = VisualizationAgent()
    result = agent.run({"data": sample_df, "chart_type": "bar", "x": "category", "y": "sales"})
    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.has_evidence is True


def test_insight_agent_conformance(sample_df):
    agent = InsightAgent()
    result = agent.run({"data": sample_df, "type": "smart"})
    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.has_evidence is True


def test_report_agent_conformance(sample_df):
    agent = ReportAgent()
    out1 = AnalysisAgent().run({"data": sample_df, "request": "summary"})
    result = agent.run({"agent_outputs": [out1], "request": "pipeline"})
    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert "report" in result.output


def test_planner_orchestration_with_validation(sample_df):
    planner = PlannerAgent(data=sample_df)
    res = planner.run_agent({"action": "summary"})
    assert isinstance(res, AgentResult)
    assert res.is_success is True
    assert res.validation is not None
    assert res.validation.passed is True
