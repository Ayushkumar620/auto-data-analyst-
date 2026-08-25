import os
import pandas as pd
import pytest

from agent.semantic_schema_agent import SemanticSchemaAgent
from agent.dataset_knowledge import SemanticType
from agent.timeseries_detector import TimeSeriesDetector
from agent.predictor import DataPredictor
from agent.intent import CommandIntelligenceAgent
from agent.dynamic_planner import DynamicTaskPlanner
from agent.autonomous_forecast_engine import AutonomousForecastEngine
from agent.forecasting_schemas import ForecastRequest
from backend.app.forecasting.forecaster import Forecaster
from backend.app.eda.summary import SummaryAnalyzer
from backend.app.eda.orchestrator import EDAOrchestrator
from backend.app.reports.builder import ReportBuilder
from backend.app.reports.pdf import PdfReportGenerator


@pytest.fixture
def budget_df():
    excel_path = os.path.join('uploads', 'Budget-Forecast.xlsx')
    if os.path.exists(excel_path):
        return pd.read_excel(excel_path)
    return pd.DataFrame({
        'FiscalYear': [2024, 2024, 2024, 2024, 2025, 2025],
        'Dept': ['Finance', 'Finance', 'HR', 'HR', 'Finance', 'HR'],
        'Quarter': [1, 2, 1, 2, 1, 1],
        'BudgetUSD': [35000.0, 80000.0, 30000.0, 32000.0, 40000.0, 33000.0],
        'ForecastUSD': [36000.0, 85000.0, 31000.0, 33000.0, 42000.0, 34000.0],
        'ActualUSD': [37613.91, 86720.13, 30596.03, 31800.0, 39500.0, 32500.0],
        'VarianceUSD': [-2613.91, -6720.13, -596.03, 200.0, 500.0, 500.0],
        'Notes': [float('nan')] * 6,
    })


def test_1_fiscal_year_detected_as_temporal(budget_df):
    ssa = SemanticSchemaAgent()
    dk = ssa.analyze_dataset(budget_df, 'Budget-Forecast.xlsx')
    fy_ck = dk.get_column_knowledge('FiscalYear')
    assert fy_ck is not None
    assert fy_ck.role in ('date', 'temporal', 'dimension')
    assert fy_ck.semantic_type in (SemanticType.DATE, SemanticType.DATETIME, SemanticType.DIMENSION)
    primary_date = dk.get_primary_date_column()
    assert primary_date in ('FiscalYear', 'Quarter')


def test_2_actual_usd_detected_as_numeric_measure(budget_df):
    ssa = SemanticSchemaAgent()
    dk = ssa.analyze_dataset(budget_df, 'Budget-Forecast.xlsx')
    actual_ck = dk.get_column_knowledge('ActualUSD')
    assert actual_ck is not None
    assert actual_ck.role == 'metric'
    assert actual_ck.semantic_type == SemanticType.METRIC


def test_3_forecast_usd_detected_as_numeric_measure(budget_df):
    ssa = SemanticSchemaAgent()
    dk = ssa.analyze_dataset(budget_df, 'Budget-Forecast.xlsx')
    fc_ck = dk.get_column_knowledge('ForecastUSD')
    assert fc_ck is not None
    assert fc_ck.role == 'metric'
    assert fc_ck.semantic_type == SemanticType.METRIC


def test_4_explicit_forecast_actual_usd_selects_actual_usd(budget_df):
    tsd = TimeSeriesDetector()
    target = tsd.detect_target_column(budget_df, hint='ActualUSD')
    assert target == 'ActualUSD'
    dp = DataPredictor(budget_df)
    res = dp.forecast(target='ActualUSD')
    assert res.get('target_column') == 'ActualUSD'
    assert res.get('target') == 'ActualUSD'


def test_5_explicit_forecast_fiscal_year_still_works(budget_df):
    tsd = TimeSeriesDetector()
    target = tsd.detect_target_column(budget_df, hint='FiscalYear')
    assert target == 'FiscalYear'
    dp = DataPredictor(budget_df)
    res = dp.forecast(target='FiscalYear')
    assert res.get('target_column') == 'FiscalYear'
    assert res.get('target') == 'FiscalYear'


def test_6_generic_forecast_does_not_select_fiscal_year(budget_df):
    tsd = TimeSeriesDetector()
    time_col = tsd.detect_time_column(budget_df)
    target = tsd.detect_target_column(budget_df, time_col=time_col)
    assert target != 'FiscalYear'
    assert target in ('ActualUSD', 'BudgetUSD', 'ForecastUSD', 'VarianceUSD')
    dp = DataPredictor(budget_df)
    res = dp.forecast()
    assert res.get('target_column') != 'FiscalYear'
    assert res.get('target_column') in ('ActualUSD', 'BudgetUSD', 'ForecastUSD', 'VarianceUSD')
    ssa = SemanticSchemaAgent()
    dk = ssa.analyze_dataset(budget_df, 'Budget-Forecast.xlsx')
    primary_metric = dk.get_primary_metric()
    assert primary_metric != 'FiscalYear'
    assert primary_metric in ('ActualUSD', 'BudgetUSD', 'ForecastUSD', 'VarianceUSD')


def test_7_missing_count_is_correct(budget_df):
    expected_missing = int(budget_df.isna().sum().sum())
    sa = SummaryAnalyzer()
    summary = sa.analyze(budget_df)
    assert summary['missing_values'] == expected_missing
    assert summary['missing_count'] == expected_missing
    expected_pct = round((expected_missing / (budget_df.shape[0] * budget_df.shape[1])) * 100, 2)
    assert summary['missing_percentage'] == expected_pct


def test_8_duplicate_count_is_correct(budget_df):
    expected_dups = int(budget_df.duplicated().sum())
    sa = SummaryAnalyzer()
    summary = sa.analyze(budget_df)
    assert summary['duplicate_rows'] == expected_dups


def test_9_forecast_validation_rejects_unsuitable_targets():
    const_df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'constant_val': [100.0] * 10,
    })
    tsd = TimeSeriesDetector()
    target = tsd.detect_target_column(const_df)
    assert target is None
    engine = AutonomousForecastEngine()
    req_empty = ForecastRequest(dataset=pd.DataFrame(), target_column='target')
    res_empty = engine.run_forecast(req_empty)
    assert res_empty.status == 'NOT_SUPPORTED'


def test_10_existing_eda_command_still_works(budget_df):
    eda = EDAOrchestrator().analyze(budget_df)
    assert 'summary' in eda
    assert 'statistics' in eda
    assert 'correlations' in eda
    assert eda['summary']['row_count'] == len(budget_df)
    assert eda['summary']['missing_values'] == int(budget_df.isna().sum().sum())


def test_11_existing_report_command_still_works(budget_df):
    eda = EDAOrchestrator().analyze(budget_df)
    analysis = {
        'dataframe': budget_df,
        'eda': eda,
        'insights': [{'title': 'Budget Tracking', 'description': 'Actual vs Budget analyzed.'}],
        'cleaning': {},
    }
    report = ReportBuilder().build('test_ds', analysis)
    assert report.dataset_id == 'test_ds'
    assert report.dataset_overview['rows'] == len(budget_df)
    assert report.data_quality['missing_values'] == int(budget_df.isna().sum().sum())


def test_12_existing_pdf_generation_still_works(budget_df):
    eda = EDAOrchestrator().analyze(budget_df)
    analysis = {
        'dataframe': budget_df,
        'eda': eda,
        'insights': [{'title': 'Budget Tracking', 'description': 'Actual vs Budget analyzed.'}],
        'cleaning': {},
    }
    report = ReportBuilder().build('test_ds', analysis)
    pdf_bytes = PdfReportGenerator().generate(report)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b'%PDF')
