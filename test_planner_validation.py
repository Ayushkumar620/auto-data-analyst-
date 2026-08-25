"""Verify PlannerAgent runs ResultValidator.repair() on every AgentResult.

Regression guard: the happy path in test_chat.py stays green because repair is
a no-op on already-valid results; this file pins the new behaviour that a
malformed result is validated and repaired before it leaves the planner.
"""
import pandas as pd

from agent.base import BaseAgent
from agent.loader import load_data
from agent.planner import PlannerAgent
from agent.schemas import AgentStatus, ClaimType


SAMPLE_CSV = "sample_data.csv"


class _BrokenAgent(BaseAgent):
    """An agent that deliberately returns a *malformed* AgentResult."""

    name = "Broken Test Agent"
    role = "tester"

    def run(self, task):
        self._start()
        from datetime import datetime
        from agent.schemas import Evidence, AgentResult

        return AgentResult.model_construct(
            agent=self.name,
            agent_name=self.name,
            role=self.role,
            agent_id=self.agent_id,
            task_id=self.agent_id,
            status=AgentStatus.COMPLETED,
            started_at=self.started_at,
            timestamp=self.started_at,
            finished_at=datetime.now(),
            output={"ok": True},
            data={"ok": True},
            evidence=[Evidence.model_construct(
                source=self.name,
                source_reference=self.name,
                method="test.broken",
                operation="test.broken",
                data_ref={"column": "totally_nonexistent_column"},
                confidence=0.9,
                claim_type=ClaimType.FACT,
            )],
            confidence=5.0,  # out of [0, 1] on purpose
        )


def test_run_agent_validates_and_repairs_each_result(monkeypatch):
    data = load_data(SAMPLE_CSV)

    monkeypatch.setattr(
        PlannerAgent,
        "REQUEST_MAP",
        {
            "summary": PlannerAgent.REQUEST_MAP["summary"],
            "broken_for_test": {
                "action": "broken_for_test",
                "agent": _BrokenAgent,
                "task": lambda data, req: {"data": data},
            },
        },
    )

    planner = PlannerAgent(data)
    result = planner.run_agent({"action": "broken_for_test"})

    # The planner must have attached a validation result...
    assert result.validation is not None
    # ...which passed after repair...
    assert result.validation.passed is True
    # ...and recorded that a repair actually happened.
    assert result.validation.repaired is True

    # confidence clamped back into range, evidence referencing an unknown
    # column dropped, output preserved.
    assert result.confidence == 1.0
    assert result.evidence == []
    assert result.output.get("ok") is True
    assert result.status == AgentStatus.COMPLETED
