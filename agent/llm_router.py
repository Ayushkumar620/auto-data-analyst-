"""
LLM Router - Uses an LLM (via API key) to understand natural language commands
and convert them into structured JSON task plans for the agent system.
Falls back to the rule-based NLP parser if no API key is configured.
"""
import json
import re
import urllib.request
import urllib.error

from . import config
from .nlp_parser import NLPCommandParser


class LLMRouter:
    """Routes natural language commands to task plans using an LLM."""

    def __init__(self):
        self.api_key = config.get_api_key()
        self.provider = config.get_llm_provider()
        self.base_url = config.get_llm_base_url()
        self.model = config.get_llm_model()
        self.llm_enabled = bool(self.api_key)
        self.nlp = NLPCommandParser()

    def route(self, command, data_summary=None):
        """Convert a natural language command into a structured task plan.

        Returns a dict with 'task' (the structured plan) and 'source' indicating
        whether the LLM or the rule-based parser was used.
        """
        if self.llm_enabled:
            try:
                plan = self._call_llm(command, data_summary)
                if plan:
                    return {"task": plan, "source": "llm"}
            except Exception as e:
                # Fall through to rule-based on any LLM error
                pass

        # Rule-based fallback
        task = self._rule_based_plan(command)
        return {"task": task, "source": "rules"}

    def _rule_based_plan(self, command):
        """Build a task plan using the rule-based NLP parser."""
        intent = self.nlp.parse(command)

# Map intent to a task plan
        action = intent.action

        # If a metric keyword is present (total/sum/average/count/max/min),
        # treat it as an aggregation/insight request
        if intent.metric and action in ("", "summary", "transaction"):
            return {
                "action": "insight",
                "intent": self._intent_to_dict(intent),
                "request": "insight",
            }
        if action == "predict":
            return {
                "action": "predict",
                "target": intent.target or None,
                "request": "predict",
            }
        if action == "chart":
            return {
                "action": "chart",
                "chart_type": intent.chart_type or "auto",
                "x": intent.column or None,
                "y": None,
                "request": "chart",
            }
        if action == "transaction":
            return {
                "action": "insight",
                "intent": self._intent_to_dict(intent),
                "request": "insight",
            }
        if action == "text":
            return {
                "action": "text",
                "request": "text",
            }
        if action == "correlation":
            return {"action": "correlation", "request": "correlation"}
        if action == "nulls":
            return {"action": "nulls", "request": "nulls"}
        if action == "unique":
            return {"action": "unique", "request": "unique"}
        if action == "head":
            return {"action": "head", "request": "head"}
        if action == "summary" and (intent.metric or intent.column or intent.group_by):
            return {
                "action": "insight",
                "intent": self._intent_to_dict(intent),
                "request": "insight",
            }
        if action == "summary":
            return {"action": "summary", "request": "summary"}

        # Default: summary
        return {"action": "summary", "request": "summary"}

    def _intent_to_dict(self, intent):
        return {
            "action": intent.action,
            "metric": intent.metric,
            "amount_type": intent.amount_type,
            "time_filter": intent.time_filter,
            "group_by": intent.group_by,
            "target": intent.target,
            "column": intent.column,
            "chart_type": intent.chart_type,
        }

    def _call_llm(self, command, data_summary):
        """Call the LLM to parse a command into a structured task plan."""
        system_prompt = (
            "You are an AI orchestrator for a data analysis multi-agent system.\n"
            "You convert human commands into a structured JSON task plan. "
            "You must respond with ONLY valid JSON, no other text.\n\n"
            "The task plan must follow exactly this schema:\n"
            '{"action": "summary or describe or stats or nulls or correlation or head '
            'or unique or text or chart or predict or insight", '
            '"request": "summary", "chart_type": "auto or bar or line or pie or scatter '
            'or histogram or box", "x": "column_name_or_null", '
            '"y": "column_name_or_null", "target": "column_name_for_prediction_or_null", '
            '"metric": "total or average or count or maximum or minimum or null", '
            '"group_by": "column_name_or_null"}\n\n'
            "Rules:\n"
            "- For full data overview use action summary\n"
            "- If user asks to sum/aggregate/count a column, use action insight "
            "with metric and column\n"
            "- If user asks 'by X' or 'per X', set group_by\n"
            "- 'predict [column]' -> action predict with target\n"
            "- Any chart request -> action chart with a chart_type\n"
            "- 'how many words' or 'text' -> action text"
        )

        data_desc = ""
        if data_summary:
            data_desc = f"\nAvailable data columns: {json.dumps(data_summary)}"

        user_prompt = f"User command: \"{command}\"{data_desc}\nReturn the JSON task plan."

        data = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 500,
        }).encode("utf-8")

        url = self.base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")

        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        content = body["choices"][0]["message"]["content"]
        return self._extract_json(content)

    def _extract_json(self, content):
        """Extract JSON from LLM response (handles code fences and extra text)."""
        if not content:
            return None
        content = content.strip()
        # Remove markdown code fences
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if fence:
            content = fence.group(1)
        # Find first { to last }
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start:end + 1]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
