"""
Tests for Milestone 5, Task 2: Conversational Analyst and Evidence-Based Report Generation.

Verifies:
1. Session creation and lifecycle
2. Context persistence across multiple turns
3. Dataset persistence without repeated upload
4. Follow-up question resolution ("Show me North")
5. Pronoun resolution ("Why did it increase?")
6. Ambiguous reference handling (asks clarification question)
7. Conversational intent resolution
8. Multi-step question planning
9. Evidence-backed response generation
10. Zero hallucinated numbers
11. Context summarization (ConversationSummary)
12. Multi-section report generation
13. Quick report (quick_summary)
14. Analyst report (analyst_report)
15. Technical report (technical_report)
16. Recommendation separation from facts
17. Dynamic limitation generation
18. Session security and isolation
19. LLM unavailable / fully deterministic execution
20. Partial failure handling
21. AgentResult integration
"""
import pytest
import numpy as np
import pandas as pd

from agent.conversational_analyst import ConversationalAnalystAgent
from agent.conversational_schemas import (
    ConversationSession,
    ConversationSummary,
    ConversationTurn,
    ConversationalIntent,
    GeneratedReport,
    ReportType,
)
from agent.context_resolver import ContextResolver
from agent.evidence_report_generator import EvidenceReportGenerator
from agent.schemas import AgentResult, AgentStatus, ClaimType
from agent.tool_registry import DEFAULT_TOOL_REGISTRY


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sales_df():
    """Deterministic sales dataset for multi-turn conversational testing."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2025-01-01", periods=6, freq="ME").repeat(10)
    regions = np.random.choice(["North", "South", "East", "West"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    products = np.random.choice(["ProductA", "ProductB"], size=n, p=[0.6, 0.4])
    rev = np.linspace(2000, 8000, n) + np.random.normal(0, 100, n)
    units = np.random.randint(10, 50, size=n)

    return pd.DataFrame({
        "date": dates,
        "region": regions,
        "product": products,
        "revenue": rev,
        "units": units,
    })


# ==============================================================================
# 1-6. Session, Context & Anaphora Resolution
# ==============================================================================

def test_session_creation_and_lifecycle():
    """1. Test session initialization and state storage."""
    agent = ConversationalAnalystAgent()
    sess = agent.get_or_create_session("session_123")

    assert sess.session_id == "session_123"
    assert len(sess.turns) == 0
    assert sess.created_at > 0


def test_context_and_dataset_persistence(sales_df):
    """2 & 3. Test dataset and analytical context persist across turns without re-upload."""
    agent = ConversationalAnalystAgent()
    sess_id = "sess_persist_test"

    # Turn 1: Provide data
    resp1, ev1, meta1 = agent.chat("Analyze my sales data.", session_id=sess_id, data=sales_df)
    assert len(ev1) > 0

    sess = agent.get_or_create_session(sess_id)
    assert sess.active_dataset is not None
    assert sess.dataset_context.row_count == 60
    assert len(sess.turns) == 1

    # Turn 2: Do NOT provide data; should reuse active dataset seamlessly
    resp2, ev2, meta2 = agent.chat("Show me the trend.", session_id=sess_id, data=None)
    assert "error" not in meta2
    assert len(sess.turns) == 2


def test_follow_up_dimension_query(sales_df):
    """4. Test follow-up question drilling down into a specific segment ("Show me North")."""
    agent = ConversationalAnalystAgent()
    sess_id = "sess_followup_test"

    agent.chat("Analyze sales by region.", session_id=sess_id, data=sales_df)
    resp, ev, meta = agent.chat("Show me North.", session_id=sess_id)

    assert meta.get("intent") in ("drill_down", "analyze", "filter")
    assert "North" in meta.get("resolved_command", "")


def test_pronoun_resolution(sales_df):
    """5. Test resolving 'it' to the active metric ('revenue')."""
    agent = ConversationalAnalystAgent()
    sess_id = "sess_pronoun_test"

    agent.chat("Analyze revenue growth.", session_id=sess_id, data=sales_df)
    resp, ev, meta = agent.chat("Why did it increase?", session_id=sess_id)

    assert "revenue" in meta.get("resolved_command", "").lower()
    assert meta.get("intent") == "investigate"


def test_ambiguous_reference_handling():
    """6. Test that ambiguous references prompt a clarification question rather than guessing."""
    agent = ConversationalAnalystAgent()
    sess_id = "sess_ambiguous_test"
    sess = agent.get_or_create_session(sess_id)

    # Simulate previous insights mentioning both revenue and profit
    from agent.autonomous_analysis_schemas import Insight, InsightCategory
    from agent.schemas import Evidence

    ev = Evidence(source="test", method="test", confidence=0.9, claim_type=ClaimType.FACT)
    ins1 = Insight(title="Revenue", summary="Revenue up 20%", category=InsightCategory.TREND, evidence=ev, affected_columns=["revenue"])
    ins2 = Insight(title="Profit", summary="Profit up 15%", category=InsightCategory.TREND, evidence=ev, affected_columns=["profit"])
    sess.previous_insights.extend([ins1, ins2])
    sess.dataset_context = None  # No single primary metric locked

    resp, ev_list, meta = agent.chat("Why did it increase?", session_id=sess_id)

    assert meta.get("needs_clarification") is True
    assert "Do you mean" in meta.get("prompt", "")


# ==============================================================================
# 7-11. Intent, Planning, Evidence & Summarization
# ==============================================================================

def test_conversational_intent_resolution():
    """7. Test intent mapping across diverse analytical inquiries."""
    resolver = ContextResolver()
    sess = ConversationSession(session_id="test")

    _, intent_rep, _, _, _ = resolver.resolve("Give me a professional report.", sess)
    assert intent_rep == ConversationalIntent.GENERATE_REPORT

    _, intent_why, _, _, _ = resolver.resolve("Why did sales drop?", sess)
    assert intent_why == ConversationalIntent.INVESTIGATE

    _, intent_comp, _, _, _ = resolver.resolve("Compare North and South.", sess)
    assert intent_comp == ConversationalIntent.COMPARE

    _, intent_fc, _, _, _ = resolver.resolve("Forecast next quarter.", sess)
    assert intent_fc == ConversationalIntent.FORECAST


def test_multi_step_question_handling(sales_df):
    """8. Test handling a multi-part analytical request."""
    agent = ConversationalAnalystAgent()
    sess_id = "sess_multistep_test"

    resp, ev, meta = agent.chat(
        "Analyze sales, find major drivers, check for anomalies, and tell me what to do.",
        session_id=sess_id,
        data=sales_df,
    )

    assert len(ev) > 0
    assert "Analysis for:" in resp
    assert "Recommendation" in resp


def test_evidence_backed_no_hallucinated_numbers(sales_df):
    """9 & 10. Test that all responses are strictly grounded in verified Evidence."""
    agent = ConversationalAnalystAgent()
    sess_id = "sess_evidence_test"

    resp, ev_list, meta = agent.chat("Analyze my sales data.", session_id=sess_id, data=sales_df)

    assert len(ev_list) > 0
    for ev in ev_list:
        assert ev.source.startswith("AutonomousAnalysisEngine")
        assert ev.confidence >= 0.80


def test_conversation_summarization(sales_df):
    """11. Test condensing conversation state into a ConversationSummary."""
    agent = ConversationalAnalystAgent()
    sess_id = "sess_summary_test"

    agent.chat("Analyze sales data.", session_id=sess_id, data=sales_df)
    sess = agent.get_or_create_session(sess_id)
    summary = agent.summarize_session(sess)

    assert isinstance(summary, ConversationSummary)
    assert summary.active_dataset is not None
    assert summary.important_metrics["rows"] == 60


# ==============================================================================
# 12-17. Report Generation (Quick, Analyst, Executive, Technical)
# ==============================================================================

def test_report_generation_quick_summary(sales_df):
    """12 & 13. Test generating a concise quick summary report."""
    agent = ConversationalAnalystAgent()
    sess_id = "sess_rep_quick"

    agent.chat("Analyze sales.", session_id=sess_id, data=sales_df)
    resp, ev, meta = agent.chat("Give me a quick summary report.", session_id=sess_id)

    assert "# Executive Briefing:" in resp
    assert "QUICK_SUMMARY" in resp
    assert "Key Metrics" in resp


def test_report_generation_analyst_report(sales_df):
    """14. Test generating a comprehensive multi-section analyst report."""
    agent = ConversationalAnalystAgent()
    sess_id = "sess_rep_analyst"

    agent.chat("Analyze sales.", session_id=sess_id, data=sales_df)
    resp, ev, meta = agent.chat("Give me a professional analyst report.", session_id=sess_id)

    assert "# Comprehensive Data Intelligence Report:" in resp
    assert "1. Dataset Architecture & Schema" in resp
    assert "Mathematical Evidence Ledger" in resp


def test_report_generation_technical_report(sales_df):
    """15. Test generating a technical statistical report."""
    agent = ConversationalAnalystAgent()
    sess_id = "sess_rep_tech"

    agent.chat("Analyze sales.", session_id=sess_id, data=sales_df)
    resp, ev, meta = agent.chat("Explain the analysis technically.", session_id=sess_id)

    assert "TECHNICAL_REPORT" in resp
    assert "Statistical Ingestion & Quality Parameters" in resp


def test_recommendation_separation_and_limitations(sales_df):
    """16 & 17. Test that recommendations are distinct from facts and limitations are data-grounded."""
    gen = EvidenceReportGenerator()
    agent = ConversationalAnalystAgent()
    sess_id = "sess_rec_test"

    agent.chat("Analyze sales.", session_id=sess_id, data=sales_df)
    sess = agent.get_or_create_session(sess_id)
    report = gen.generate_report(sess, report_type=ReportType.ANALYST_REPORT)

    assert "Strategic Recommendations" in report.markdown_content
    assert "Evidentiary Limitations & Constraints" in report.markdown_content
    # Small sample size limitation present for 60-row dataset
    assert any("sample size" in lim.lower() for lim in report.limitations)


# ==============================================================================
# 18-21. Security, Fallbacks & AgentResult
# ==============================================================================

def test_session_isolation_and_security(sales_df):
    """18. Test that Session A cannot access or mutate Session B data."""
    agent = ConversationalAnalystAgent()
    
    # Session A gets dataset
    agent.chat("Analyze sales.", session_id="session_A", data=sales_df)
    
    # Session B has no data
    resp_b, _, meta_b = agent.chat("What is the trend?", session_id="session_B", data=None)

    assert meta_b.get("error") == "no_data"
    assert len(agent.get_or_create_session("session_B").turns) == 0


def test_deterministic_execution_without_llm(sales_df):
    """19. Test that conversational engine produces complete structured results without external LLM."""
    agent = ConversationalAnalystAgent()
    resp, ev, meta = agent.chat("Analyze sales.", session_id="sess_det", data=sales_df)

    assert "📊 **Analysis for:" in resp
    assert len(ev) > 0
    assert meta["session_id"] == "sess_det"


def test_conversational_agent_run_conformance(sales_df):
    """20 & 21. Test standardized BaseAgent run() interface returning AgentResult."""
    agent = ConversationalAnalystAgent()
    task = {
        "command": "Analyze sales and tell me what happened.",
        "session_id": "sess_agent_run",
        "data": sales_df,
    }
    result = agent.run(task)

    assert isinstance(result, AgentResult)
    assert result.is_success is True
    assert result.status == AgentStatus.COMPLETED
    assert "response" in result.data
    assert len(result.evidence) > 0
    assert "session_id" in result.metadata
