"""Test the Planner Agent and /api/chat endpoint."""
import io
import app
from agent.loader import load_data
from agent.planner import PlannerAgent

print("=== Planner Agent test ===")
data = load_data("sample_data.csv")
planner = PlannerAgent(data)

# Single agent orchestration
out = planner.run_agent({"action": "summary"})
print("summary agent status:", out.get("status"))

out = planner.run_agent({"action": "insights"})
print("insights agent status:", out.get("status"))

out = planner.run_agent({"action": "chart", "chart_type": "bar", "x": "category"})
print("chart agent status:", out.get("status"))

# Full pipeline
pipe = planner.run_pipeline(data, steps=["clean", "summary", "insights"])
print("pipeline steps:", pipe.get("steps"))
print("pipeline agent statuses:", [o.get("status") for o in pipe.get("agents", [])])

print("\n=== /api/chat endpoint test ===")
client = app.app.test_client()
with open("sample_data.csv", "rb") as f:
    raw = f.read()

# First upload to get data back via analyze
data_post = {"file": (io.BytesIO(raw), "sample_data.csv"), "command": "summary"}
r = client.post("/api/analyze", data=data_post, content_type="multipart/form-data")
j = r.get_json()
print("analyze summary:", r.status_code, j.get("type"))

# Chat with data (pass data payload)
chat_payload = {"message": "total sales", "data": j}
r = client.post("/api/chat", json=chat_payload)
cj = r.get_json()
print("chat 'total sales':", r.status_code, cj.get("type"), "| answer:", cj.get("answer"))

chat_payload = {"message": "insights", "data": j}
r = client.post("/api/chat", json=chat_payload)
cj = r.get_json()
print("chat 'insights':", r.status_code, cj.get("type"), "| answer:", str(cj.get("answer"))[:60])

chat_payload = {"message": "anomalies", "data": j}
r = client.post("/api/chat", json=chat_payload)
cj = r.get_json()
print("chat 'anomalies':", r.status_code, cj.get("type"), "| answer:", str(cj.get("answer"))[:60])

chat_payload = {"message": "how many rows", "data": j}
r = client.post("/api/chat", json=chat_payload)
cj = r.get_json()
print("chat 'how many rows':", r.status_code, cj.get("type"), "| answer:", str(cj.get("answer"))[:60])
