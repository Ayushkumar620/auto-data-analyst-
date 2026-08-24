"""LLM Router - Routes natural language commands to structured task plans using Provider-Agnostic LLMs."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
import pandas as pd

from . import config
from .nlp_parser import NLPCommandParser
from backend.app.core.llm_provider import BaseLLMProvider, LLMClientFactory, LLMMessage
from backend.app.core.dynamic_context import DynamicContextAssembler, DynamicPromptContext


class LLMRouter:
    """Routes natural language commands to task plans using provider-agnostic LLMs."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self.api_key = config.get_api_key()
        self.provider_name = config.get_llm_provider()
        self.base_url = config.get_llm_base_url()
        self.model = config.get_llm_model()

        if provider is not None:
            self.llm_provider = provider
        else:
            self.llm_provider = LLMClientFactory.get_provider(
                provider_name=self.provider_name if self.api_key else "mock",
                api_key=self.api_key,
                model=self.model,
                base_url=self.base_url,
            )

        self.llm_enabled = bool(self.api_key or self.provider_name in ("ollama", "mock"))
        self.nlp = NLPCommandParser()
        self.context_assembler = DynamicContextAssembler()

    def route(
        self,
        command: str,
        data_summary: Optional[Dict[str, Any]] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Convert a natural language command into a structured task plan."""
        if self.llm_enabled:
            try:
                plan = self._call_llm(command, data_summary, dataframe)
                if plan:
                    return {"task": plan, "source": "llm", "provider": self.llm_provider.provider_type.value}
            except Exception:
                pass

        # Rule-based fallback
        task = self._rule_based_plan(command)
        return {"task": task, "source": "rules"}

    def _call_llm(
        self,
        command: str,
        data_summary: Optional[Dict[str, Any]] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict[str, Any]]:
        """Assemble dynamic context and invoke the underlying LLM provider."""
        prompt_ctx: DynamicPromptContext = self.context_assembler.assemble(
            query=command,
            dataframe=dataframe,
        )

        response = self.llm_provider.generate(
            messages=prompt_ctx.messages,
            temperature=0.1,
            max_tokens=500,
        )

        return self._extract_json(response.content)

    def _rule_based_plan(self, command: str) -> Dict[str, Any]:
        """Build a task plan using the rule-based NLP parser."""
        intent = self.nlp.parse(command)
        action = intent.action

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
            return {"action": "text", "request": "text"}
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

        return {"action": "unknown", "raw_command": command}

    def _intent_to_dict(self, intent) -> Dict[str, Any]:
        return {
            "metric": intent.metric,
            "column": intent.column,
            "group_by": intent.group_by,
            "amount_type": intent.amount_type,
            "target": intent.target,
            "chart_type": intent.chart_type,
        }

    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response (handles code fences and extra text)."""
        if not content:
            return None
        content = content.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if fence:
            content = fence.group(1)
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
