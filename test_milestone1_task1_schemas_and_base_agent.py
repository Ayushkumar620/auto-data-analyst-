"""
Tests for Milestone 1, Task 1: Standardized AgentResult, AgentError, Evidence, and BaseAgent Architecture.

Verifies:
1. Successful AgentResult creation, typing, and validation
2. Failed AgentResult creation, errors list, and confidence
3. Partial AgentResult creation, status, and message
4. Invalid confidence rejection (confidence < 0.0 or > 1.0)
5. AgentError creation, code, recoverable flag, and details
6. Evidence creation, columns, operation, calculation, and confidence
7. BaseAgent execution contract (_start, _finish, _partial, _error)
8. BaseAgent exception handling (safe messages, no raw stack traces exposed to user, execution time tracked)
"""
from datetime import datetime
import pytest
from pydantic import ValidationError

from agent.schemas import (
    AgentResult,
    AgentError,
    AgentStatus,
    ClaimType,
    ErrorCategory,
    Evidence,
)
from agent.base import BaseAgent


# ==============================================================================
# 1. Schema Tests (AgentResult, AgentError, Evidence)
# ==============================================================================

def test_successful_agent_result():
    """Verify successful AgentResult instantiation and fields."""
    ev = Evidence(
        dataset_name="sales.csv",
        columns=["Revenue", "Profit"],
        operation="aggregation.mean",
        calculation="df[['Revenue', 'Profit']].mean()",
        result={"Revenue": 150000.0, "Profit": 30000.0},
        confidence=0.98,
    )
    res = AgentResult.success(
        agent_name="AnalysisAgent",
        task_id="task_101",
        data={"mean_revenue": 150000.0},
        message="Computed mean metrics.",
        evidence=[ev],
        confidence=0.98,
        execution_time=14.2,
        metadata={"engine": "pandas"},
    )

    assert res.is_success is True
    assert res.is_error is False
    assert res.is_partial is False
    assert res.status == AgentStatus.COMPLETED or res.status == "success" or res.status == "completed"
    assert res.agent_name == "AnalysisAgent"
    assert res.task_id == "task_101"
    assert res.data == {"mean_revenue": 150000.0}
    assert res.message == "Computed mean metrics."
    assert res.confidence == 0.98
    assert res.execution_time == 14.2
    assert len(res.evidence) == 1
    assert res.evidence[0].operation == "aggregation.mean"


def test_failed_agent_result():
    """Verify failed AgentResult instantiation, errors, and confidence."""
    err = AgentError(
        code="ZERO_DIVISION",
        message="Cannot divide by zero in profit margin calculation.",
        details={"column": "Profit", "denominator": 0},
        recoverable=False,
        agent_name="AnalysisAgent",
    )
    res = AgentResult.failure(
        agent_name="AnalysisAgent",
        task_id="task_102",
        errors=[err],
        message="AnalysisAgent failed: Cannot divide by zero in profit margin calculation.",
        execution_time=8.5,
    )

    assert res.is_success is False
    assert res.is_error is True
    assert res.is_partial is False
    assert res.status == AgentStatus.ERROR or res.status == "failed" or res.status == "error"
    assert res.confidence == 0.0
    assert len(res.errors) == 1
    assert res.errors[0].code == "ZERO_DIVISION"
    assert res.errors[0].recoverable is False
    assert res.has_errors is True


def test_partial_agent_result():
    """Verify partial AgentResult instantiation."""
    err = AgentError(
        code="HIGH_NULL_RATIO",
        message="Missing values detected in 2 columns.",
        recoverable=True,
    )
    res = AgentResult.partial(
        agent_name="CleaningAgent",
        task_id="task_103",
        data={"cleaned_rows": 850},
        message="Cleaning completed with imputations.",
        confidence=0.75,
        errors=[err],
        warnings=["Imputed 150 null values with median."],
        execution_time=22.1,
    )

    assert res.is_partial is True
    assert res.is_success is False
    assert res.status == AgentStatus.PARTIAL or res.status == "partial"
    assert res.confidence == 0.75
    assert len(res.warnings) == 1
    assert res.data["cleaned_rows"] == 850


def test_invalid_confidence_validation():
    """Verify that confidence < 0.0 or > 1.0 raises ValidationError."""
    # Confidence > 1.0
    with pytest.raises(ValidationError):
        AgentResult(
            agent_name="TestAgent",
            confidence=1.5,
        )

    # Confidence < 0.0
    with pytest.raises(ValidationError):
        AgentResult(
            agent_name="TestAgent",
            confidence=-0.2,
        )

    # Evidence confidence > 1.0
    with pytest.raises(ValidationError):
        Evidence(
            dataset_name="data.csv",
            confidence=1.1,
        )


def test_agent_error_structure():
    """Verify AgentError field validation and dictionary serialization."""
    err = AgentError(
        code="INPUT_ERROR",
        message="Required column 'Sales' not found.",
        details={"available_columns": ["Revenue", "Units"]},
        recoverable=True,
        agent_name="PredictorAgent",
    )

    assert err.code == "INPUT_ERROR"
    assert err.message == "Required column 'Sales' not found."
    assert err.recoverable is True
    assert err.agent_name == "PredictorAgent"

    d = err.to_dict()
    assert d["code"] == "INPUT_ERROR"
    assert d["recoverable"] is True
    assert "available_columns" in d["details"]


def test_evidence_structure():
    """Verify Evidence field structure and calculations."""
    ev = Evidence(
        dataset_id="ds_001",
        dataset_name="customers.csv",
        columns=["Age", "SpendingScore"],
        operation="pearson_correlation",
        calculation="scipy.stats.pearsonr(Age, SpendingScore)",
        source_reference="AnalysisAgent",
        result=0.82,
        confidence=0.95,
    )

    assert ev.dataset_id == "ds_001"
    assert ev.dataset_name == "customers.csv"
    assert ev.columns == ["Age", "SpendingScore"]
    assert ev.operation == "pearson_correlation"
    assert ev.result == 0.82
    assert ev.confidence == 0.95

    d = ev.to_dict()
    assert d["operation"] == "pearson_correlation"
    assert d["result"] == 0.82


# ==============================================================================
# 2. BaseAgent Execution & Lifecycle Tests
# ==============================================================================

class DummyCalculatorAgent(BaseAgent):
    name = "Dummy Calculator"
    role = "math"

    def run(self, task):
        self._start()
        a = task.get("a", 0)
        b = task.get("b", 0)
        op = task.get("op", "+")

        if op == "+":
            return self._finish(
                result={"sum": a + b},
                confidence=1.0,
                message=f"Added {a} + {b} = {a + b}",
            )
        elif op == "/":
            if b == 0:
                return self._error(
                    message="Division by zero is undefined.",
                    code="DIVIDE_BY_ZERO",
                    recoverable=False,
                )
            return self._finish(result={"quotient": a / b})
        elif op == "partial":
            return self._partial(
                result={"approx": a + b},
                confidence=0.6,
                message="Approximated result.",
            )
        elif op == "raise":
            # Simulate unexpected Python crash
            raise RuntimeError("Unexpected internal math hardware fault")

        return self._error(f"Unsupported operation {op}")


def test_base_agent_successful_execution():
    """Verify BaseAgent _start and _finish lifecycle."""
    agent = DummyCalculatorAgent()
    res = agent.run({"a": 10, "b": 20, "op": "+"})

    assert isinstance(res, AgentResult)
    assert res.is_success is True
    assert res.status == AgentStatus.COMPLETED
    assert res.data["sum"] == 30
    assert res.agent_name == "Dummy Calculator"
    assert res.execution_time >= 0.0


def test_base_agent_partial_execution():
    """Verify BaseAgent _partial lifecycle."""
    agent = DummyCalculatorAgent()
    res = agent.run({"a": 5, "b": 5, "op": "partial"})

    assert isinstance(res, AgentResult)
    assert res.is_partial is True
    assert res.status == AgentStatus.PARTIAL
    assert res.confidence == 0.6
    assert res.data["approx"] == 10


def test_base_agent_error_execution():
    """Verify BaseAgent _error lifecycle."""
    agent = DummyCalculatorAgent()
    res = agent.run({"a": 10, "b": 0, "op": "/"})

    assert isinstance(res, AgentResult)
    assert res.is_error is True
    assert res.status == AgentStatus.ERROR
    assert res.confidence == 0.0
    assert len(res.errors) == 1
    assert "Division by zero" in res.errors[0].message


def test_base_agent_safe_run_exception_handling():
    """Verify safe_run catches unhandled exceptions without leaking stack trace to user message."""
    agent = DummyCalculatorAgent()
    res = agent.safe_run({"op": "raise"})

    assert isinstance(res, AgentResult)
    assert res.is_error is True
    assert res.status == AgentStatus.ERROR
    assert len(res.errors) == 1
    # User-facing message must be clean and free of multi-line Python traceback
    assert "\n" not in res.errors[0].message
    assert "Unexpected internal math hardware fault" in res.errors[0].message
    # Technical traceback is preserved safely in details dict for debugging
    assert "traceback" in res.errors[0].details
    assert "RuntimeError" in res.errors[0].details["exception_type"]
    assert res.execution_time >= 0.0
