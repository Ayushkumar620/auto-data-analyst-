"""
Unit and end-to-end regression tests verifying the complete analytical pipeline
for grouping, ranking, extreme values, and secondary aggregations.
"""
import pytest
import pandas as pd
import numpy as np

from agent.engine import HighPerformanceExecutionEngine
from agent.filter_engine import FilterEngine
from agent.agents import AnalysisAgent, ReportAgent
from agent.dynamic_planner import DynamicTaskPlanner
from agent.orchestrator import UniversalOrchestrator
from agent.command_orchestrator import AutonomousCommandOrchestrator
from backend.app.core.semantic import SemanticSchemaAgent
from backend.app.chat.agent import ChatAgent
from app import app


@pytest.fixture
def sales_df():
    return pd.read_csv("uploads/basic_sales_test.csv")


@pytest.fixture
def repro_query():
    return (
        "Analyze total sales and total units by product and region. "
        "Show every product-region combination, rank the combinations by total sales "
        "from highest to lowest, and identify the highest and lowest sales combination. "
        "Also calculate the average sales for each product. Do not make causal claims."
    )


def test_1_execution_engine_grouped_aggregation(sales_df):
    engine = HighPerformanceExecutionEngine()
    res = engine.aggregate(
        sales_df,
        group_by=["product", "region"],
        metrics={"sales": ["sum"], "units": ["sum"]},
        sort_by="sales_sum",
        ascending=False,
    )
    df_res = res.data
    assert len(df_res) == 6
    assert list(df_res.columns) == ["product", "region", "sales_sum", "units_sum"]
    assert df_res.iloc[0]["product"] == "Laptop" and df_res.iloc[0]["region"] == "West"
    assert df_res.iloc[0]["sales_sum"] == 38000.0
    assert df_res.iloc[0]["units_sum"] == 30.0
    assert df_res.iloc[-1]["product"] == "Tablet" and df_res.iloc[-1]["region"] == "South"
    assert df_res.iloc[-1]["sales_sum"] == 7500.0
    assert df_res.iloc[-1]["units_sum"] == 18.0


def test_2_filter_engine_extremes_and_secondary(sales_df, repro_query):
    res = FilterEngine.execute(sales_df, repro_query)
    assert res.matching_rows == 10
    assert res.total_rows == 10
    assert res.group_by == ["product", "region"]
    assert len(res.grouped_records) == 6

    ranks = [r["Rank"] for r in res.grouped_records]
    assert ranks == [1, 2, 3, 4, 5, 6]
    sales = [r["sales"] for r in res.grouped_records]
    assert sales == [38000.0, 27000.0, 17000.0, 13000.0, 11000.0, 7500.0]

    assert res.highest_record["combination"] == "Laptop / West"
    assert res.highest_record["sales"] == 38000.0
    assert res.highest_record["units"] == 30.0
    assert res.lowest_record["combination"] == "Tablet / South"
    assert res.lowest_record["sales"] == 7500.0
    assert res.lowest_record["units"] == 18.0

    assert len(res.secondary_results) == 1
    sec = res.secondary_results[0]
    prod_avgs = {r["product"]: r["sales"] for r in sec["records"]}
    assert prod_avgs["Laptop"] == 16250.0
    assert round(prod_avgs["Phone"], 2) == 9333.33
    assert round(prod_avgs["Tablet"], 2) == 6833.33


def test_3_analysis_agent_preserves_grouped_records(sales_df, repro_query):
    agent = AnalysisAgent()
    task = {"data": sales_df, "request": "group_by", "query": repro_query}
    out = agent.run(task)
    assert out.is_success
    data = out.data
    assert data["request"] == "group_by"
    assert len(data["grouped_records"]) == 6
    assert data["highest_record"]["combination"] == "Laptop / West"
    assert data["lowest_record"]["combination"] == "Tablet / South"
    assert len(data["secondary_results"]) == 1


def test_4_report_agent_incorporates_grouped_results(sales_df, repro_query):
    analysis_agent = AnalysisAgent()
    out = analysis_agent.run({"data": sales_df, "request": "group_by", "query": repro_query})

    report_agent = ReportAgent()
    rep_out = report_agent.run({"agent_outputs": [out.to_dict()], "request": repro_query})
    assert rep_out.is_success
    data = rep_out.data
    assert "grouped_records" in data
    assert len(data["grouped_records"]) == 6
    assert "Laptop / West" in data["report"]
    assert "Tablet / South" in data["report"]


def test_5_dynamic_task_planner_full_plan_and_execution(sales_df, repro_query):
    planner = DynamicTaskPlanner()
    knowledge = SemanticSchemaAgent().build_knowledge(sales_df)
    plan = planner.create_plan(repro_query, sales_df, knowledge)

    assert plan.steps[0].agent_class_name == "AnalysisAgent"
    assert plan.steps[0].action == "group_by"

    out = planner.execute_plan(plan, sales_df)
    assert out.plan_id == plan.plan_id
    step1_out = out.output["step_outputs"][str(plan.steps[0].step_id)]
    assert len(step1_out["grouped_records"]) == 6


def test_6_universal_orchestrator_aggregation(sales_df, repro_query):
    orch = UniversalOrchestrator()
    res = orch.orchestrate(repro_query, sales_df)
    assert res.is_success
    assert "grouped_records" in res.result
    assert len(res.result["grouped_records"]) == 6
    assert res.result["highest_record"]["combination"] == "Laptop / West"
    assert res.result["lowest_record"]["combination"] == "Tablet / South"
    assert len(res.result["secondary_results"]) == 1


def test_7_command_orchestrator_execution(sales_df, repro_query):
    orch = AutonomousCommandOrchestrator()
    res = orch.execute_command(repro_query, sales_df, session_id="test_session")
    assert res.filter_result is not None
    assert len(res.filter_result["grouped_records"]) == 6
    assert res.filter_result["highest_record"]["combination"] == "Laptop / West"
    assert res.filter_result["lowest_record"]["combination"] == "Tablet / South"
    assert len(res.filter_result["secondary_results"]) == 1


def test_8_api_analyze_endpoint(sales_df, repro_query):
    client = app.test_client()
    with open("uploads/basic_sales_test.csv", "rb") as f:
        resp = client.post(
            "/api/analyze",
            data={"command": repro_query, "file": (f, "basic_sales_test.csv")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["type"] == "filter_result"
    assert len(data["grouped_records"]) == 6
    assert data["highest_record"]["combination"] == "Laptop / West"
    assert data["lowest_record"]["combination"] == "Tablet / South"
    assert len(data["secondary_results"]) == 1
    assert data["matching_rows"] == 10
    assert data["total_rows"] == 10


def test_9_chat_agent_command_result(sales_df, repro_query):
    agent = ChatAgent()
    resp = agent.respond(repro_query, dataframe=sales_df)
    assert resp.status == "success"
    assert "Laptop / West" in resp.message
    assert "Tablet / South" in resp.message
    assert resp.command_result is not None
    assert len(resp.command_result["grouped_records"]) == 6
    assert resp.command_result["highest_record"]["combination"] == "Laptop / West"
    assert resp.command_result["lowest_record"]["combination"] == "Tablet / South"
