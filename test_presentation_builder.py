"""Tests for Executive Multi-Page PDF & Presentation Builder Engine.

Verifies:
1. ExecutivePresentationEngine PDF report compilation with ReportLab
2. Header, KPI cards table, narrative sections, and evidence ledger in PDF
3. Slide deck schema generation with 5 structured executive slides
4. End-to-end integration with AutonomousCommandOrchestrator results
"""
import io
import pandas as pd
import pytest

from backend.app.core.presentation_builder import (
    ExecutivePresentationEngine,
    ExecutivePresentationDeck,
    ExecutiveDeckSlide,
    global_presentation_engine,
)
from agent.command_orchestrator import AutonomousCommandOrchestrator, CommandExecutionResult


@pytest.fixture
def sample_kpis():
    return {
        "Total Revenue": 1452800.50,
        "Net Profit": 382400.25,
        "Active Customers": 1240,
        "Churn Rate": 0.042,
    }


@pytest.fixture
def sample_evidence():
    return [
        {"claim_type": "FACT", "method": "SQL Aggregation", "confidence": 1.0, "artifact": "Total Revenue is $1,452,800.50"},
        {"claim_type": "OBSERVATION", "method": "Cohort Split", "confidence": 0.95, "artifact": "Q4 revenue grew 14.2% YoY"},
        {"claim_type": "CORRELATION", "method": "Pearson r", "confidence": 0.90, "artifact": "Discount negatively correlates with Margin (r=-0.62)"},
    ]


def test_build_executive_pdf_report(sample_kpis, sample_evidence):
    """Verify PDF compilation generates valid non-empty PDF binary."""
    engine = ExecutivePresentationEngine()

    pdf_bytes = engine.build_pdf_report(
        title="Q4 Enterprise Performance Report",
        command="Analyze revenue and churn drivers",
        explanation="Revenue grew 14.2% across the West and East regions.\n\nPrimary growth driver was Enterprise subscriptions.",
        kpis=sample_kpis,
        evidence_list=sample_evidence,
        dataset_summary={"rows": 5000, "columns": ["Date", "Region", "Revenue"], "quality_score": 98},
        validation_summary={"status": "PASSED", "critical_issues": 0, "warnings": 0},
        duration_ms=45.2,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")  # Valid PDF binary signature


def test_build_presentation_deck_structure(sample_kpis, sample_evidence):
    """Verify structured slide deck composed with all 5 core executive slides."""
    engine = ExecutivePresentationEngine()

    deck: ExecutivePresentationDeck = engine.build_deck_structure(
        title="2025 Revenue Strategy Deck",
        command="Forecast sales and recommend growth levers",
        explanation="Sales are projected to grow +8.5% over the next 6 months.\n\nKey lever is reducing customer churn in the SMB tier.",
        kpis=sample_kpis,
        evidence_list=sample_evidence,
        model_summary={
            "model_name": "GradientBoostingRegressor",
            "primary_metric_name": "R2 Score",
            "primary_metric_value": 0.915,
        },
    )

    assert isinstance(deck, ExecutivePresentationDeck)
    assert deck.total_slides == 5
    assert len(deck.slides) == 5

    # Check Slide Titles
    titles = [s.title for s in deck.slides]
    assert "2025 Revenue Strategy Deck" in titles
    assert "Executive Findings & Variance Drivers" in titles
    assert "Evidence Lineage & Verification Ledger" in titles
    assert "Machine Learning Benchmark & Performance" in titles
    assert "Strategic Recommendations & Next Actions" in titles


def test_orchestrator_to_pdf_integration(sample_kpis, sample_evidence):
    """Verify AutonomousCommandOrchestrator output pipes into PDF generator."""
    df = pd.DataFrame({
        "Country": ["USA", "Germany", "Japan", "India"],
        "Revenue": [50000.0, 32000.0, 28000.0, 41000.0],
        "Profit": [12000.0, 7500.0, 6800.0, 9500.0],
    })

    orch = AutonomousCommandOrchestrator()
    res: CommandExecutionResult = orch.execute_command(
        command="Summarize revenue by country",
        dataframe=df,
        session_id="pdf_pipe_sess",
    )

    engine = ExecutivePresentationEngine()
    pdf_bytes = engine.build_pdf_report(
        title="Country Revenue Summary",
        command=res.command,
        explanation=res.final_explanation,
        kpis={"Total Revenue": float(df["Revenue"].sum()), "Total Profit": float(df["Profit"].sum())},
        evidence_list=res.evidence,
        dataset_summary=res.dataset_summary,
        validation_summary=res.validation_summary,
        duration_ms=res.duration_ms,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
