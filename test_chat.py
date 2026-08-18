"""Test the planner and command parser behavior for chat-like request routing."""
from agent.command_parser import CommandParser
from agent.loader import load_data
from agent.planner import PlannerAgent


def test_planner_agent_runs_single_actions_and_pipeline():
    data = load_data("sample_data.csv")
    planner = PlannerAgent(data)

    summary_out = planner.run_agent({"action": "summary"})
    insights_out = planner.run_agent({"action": "insights"})
    chart_out = planner.run_agent({"action": "chart", "chart_type": "bar", "x": "category"})

    assert summary_out.get("status") == "completed"
    assert insights_out.get("status") == "completed"
    assert chart_out.get("status") == "completed"

    pipe = planner.run_pipeline(data, steps=["clean", "summary", "insights"])
    assert pipe.get("steps") == ["clean", "summary", "insights"]
    assert len(pipe.get("agents", [])) == 3


def test_command_parser_handles_summary_and_chart_requests():
    data = load_data("sample_data.csv")
    parser = CommandParser(data)

    summary_result = parser.parse("summary")
    chart_result = parser.parse("chart x=category")

    assert summary_result["type"] == "summary"
    assert chart_result["type"] == "chart"
