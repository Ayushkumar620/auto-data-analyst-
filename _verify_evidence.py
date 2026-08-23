"""Verification: every agent returns evidence-carrying AgentResult."""
from agent.loader import load_data
from agent.planner import PlannerAgent

data = load_data("sample_data.csv")
p = PlannerAgent(data)

for action, req in [
    ("summary", {"action": "summary"}),
    ("correlation", {"action": "correlation"}),
    ("nulls", {"action": "nulls"}),
    ("chart", {"action": "chart", "chart_type": "auto"}),
    ("clean", {"action": "clean"}),
    ("insights", {"action": "insights"}),
    ("anomalies", {"action": "anomalies"}),
    ("forecast", {"action": "forecast"}),
    ("predict", {"action": "predict"}),
]:
    r = p.run_agent(req)
    ev = r.evidence
    kinds = sorted({e.claim_type.value for e in ev})
    print(f"{action:12s} status={r.status.value:10s} confidence={r.confidence} evidence={len(ev)} claims={kinds}")

pipe = p.run_pipeline(data, steps=["clean", "summary", "insights"])
print("pipeline steps:", pipe.get("steps"))
print("pipeline agents:", len(pipe.get("agents", [])))
rep = pipe["report"]
print("report status:", rep.get("status"), "| evidence:", len(rep.evidence), "| confidence:", rep.confidence)
if rep.to_dict()["evidence"]:
    print("report evidence[0] source:", rep.to_dict()["evidence"][0]["source"])
print("ALL OK")