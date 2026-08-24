"""Provider-Agnostic LLM Layer supporting OpenAI, Anthropic, Ollama, DeepSeek, and Mock."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request


class LLMProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    MOCK = "mock"


@dataclass
class LLMMessage:
    """Individual message in a multi-turn chat sequence."""
    role: str  # 'system', 'user', 'assistant'
    content: str


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    provider: LLMProviderType
    model: str
    token_usage: Dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0
    raw_response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "provider": self.provider.value,
            "model": self.model,
            "token_usage": self.token_usage,
            "duration_ms": round(float(self.duration_ms), 2),
        }


class BaseLLMProvider(ABC):
    """Abstract interface for all LLM providers."""
    provider_type: LLMProviderType

    @abstractmethod
    def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate response given conversation messages."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API Provider (GPT-4o, GPT-4o-mini)."""
    provider_type = LLMProviderType.OPENAI

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, messages: List[LLMMessage], temperature: float = 0.1, max_tokens: int = 1024) -> LLMResponse:
        if not self.api_key:
            raise ValueError("OpenAI API key not configured.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start_t = time.time()
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        duration = (time.time() - start_t) * 1000
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            provider=self.provider_type,
            model=self.model,
            token_usage={"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)},
            duration_ms=duration,
            raw_response=data,
        )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API Provider (Claude 3.5 Sonnet, Claude 3 Haiku)."""
    provider_type = LLMProviderType.ANTHROPIC

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022", base_url: str = "https://api.anthropic.com/v1"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, messages: List[LLMMessage], temperature: float = 0.1, max_tokens: int = 1024) -> LLMResponse:
        if not self.api_key:
            raise ValueError("Anthropic API key not configured.")

        url = f"{self.base_url}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        system_prompt = "\n".join([m.content for m in messages if m.role == "system"])
        user_msgs = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": user_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        start_t = time.time()
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        duration = (time.time() - start_t) * 1000
        content = data["content"][0]["text"] if data.get("content") else ""
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            provider=self.provider_type,
            model=self.model,
            token_usage={"prompt_tokens": usage.get("input_tokens", 0), "completion_tokens": usage.get("output_tokens", 0)},
            duration_ms=duration,
            raw_response=data,
        )


class OllamaProvider(BaseLLMProvider):
    """Local Open-Source LLM Provider (Ollama: Llama3, Mistral, Qwen, DeepSeek-R1)."""
    provider_type = LLMProviderType.OLLAMA

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = (os.environ.get("OLLAMA_BASE_URL") or base_url).rstrip("/")
        self.model = os.environ.get("OLLAMA_MODEL") or model

    def generate(self, messages: List[LLMMessage], temperature: float = 0.1, max_tokens: int = 1024) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        start_t = time.time()
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        duration = (time.time() - start_t) * 1000
        content = data.get("message", {}).get("content", "")

        return LLMResponse(
            content=content,
            provider=self.provider_type,
            model=self.model,
            token_usage={"total_tokens": data.get("eval_count", 0)},
            duration_ms=duration,
            raw_response=data,
        )


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API Provider (DeepSeek-Chat, DeepSeek-Reasoner)."""
    provider_type = LLMProviderType.DEEPSEEK

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, messages: List[LLMMessage], temperature: float = 0.1, max_tokens: int = 1024) -> LLMResponse:
        if not self.api_key:
            raise ValueError("DeepSeek API key not configured.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start_t = time.time()
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        duration = (time.time() - start_t) * 1000
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            provider=self.provider_type,
            model=self.model,
            token_usage={"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)},
            duration_ms=duration,
            raw_response=data,
        )


class MockDeterministicProvider(BaseLLMProvider):
    """Deterministic Mock LLM Provider for offline development and reproducible unit testing."""
    provider_type = LLMProviderType.MOCK

    def __init__(self, model: str = "mock-analyst"):
        self.model = model

    def generate(self, messages: List[LLMMessage], temperature: float = 0.1, max_tokens: int = 1024) -> LLMResponse:
        start_t = time.time()
        user_query = ""
        for m in reversed(messages):
            if m.role == "user":
                user_query = m.content.lower()
                break

        # Generate intelligent deterministic plan response based on user query
        if "predict" in user_query or "model" in user_query or "train" in user_query:
            resp_content = json.dumps({
                "action": "model_selection",
                "reasoning": "User requested predictive modeling. Dispatching to ModelSelectionAgent for candidate benchmarking.",
                "parameters": {"cv_folds": 5},
            })
        elif "ann" in user_query or "neural network" in user_query or "deep learning" in user_query:
            resp_content = json.dumps({
                "action": "ann",
                "reasoning": "User requested artificial neural network training.",
                "parameters": {"layers": [128, 64], "epochs": 200},
            })
        elif "cnn" in user_query or "image" in user_query or "spatial" in user_query:
            resp_content = json.dumps({
                "action": "cnn",
                "reasoning": "User requested convolutional spatial modeling.",
                "parameters": {"epochs": 80},
            })
        elif "forecast" in user_query or "future" in user_query:
            resp_content = json.dumps({
                "action": "forecast",
                "reasoning": "User requested time-series forecasting.",
                "parameters": {"periods": 5},
            })
        elif "clean" in user_query or "missing" in user_query:
            resp_content = json.dumps({
                "action": "clean",
                "reasoning": "User requested data sanitization.",
                "parameters": {"strategy": "auto_impute"},
            })
        else:
            resp_content = json.dumps({
                "action": "insights",
                "reasoning": "User requested dataset analysis. Dispatching to InsightAgent.",
                "parameters": {"type": "structured"},
            })

        duration = (time.time() - start_t) * 1000

        return LLMResponse(
            content=resp_content,
            provider=self.provider_type,
            model=self.model,
            token_usage={"prompt_tokens": len(user_query) // 4, "completion_tokens": len(resp_content) // 4},
            duration_ms=duration,
        )


class LLMClientFactory:
    """Factory to instantiate the appropriate LLM provider based on configuration."""

    @staticmethod
    def get_provider(
        provider_name: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> BaseLLMProvider:
        # Determine provider name from parameter or environment
        p_name = (provider_name or os.environ.get("LLM_PROVIDER", "mock")).lower()

        if p_name == "openai":
            return OpenAIProvider(api_key=api_key, model=model or "gpt-4o-mini", base_url=base_url or "https://api.openai.com/v1")
        elif p_name == "anthropic":
            return AnthropicProvider(api_key=api_key, model=model or "claude-3-5-sonnet-20241022", base_url=base_url or "https://api.anthropic.com/v1")
        elif p_name == "ollama":
            return OllamaProvider(base_url=base_url or "http://localhost:11434", model=model or "llama3")
        elif p_name == "deepseek":
            return DeepSeekProvider(api_key=api_key, model=model or "deepseek-chat", base_url=base_url or "https://api.deepseek.com/v1")
        else:
            return MockDeterministicProvider(model=model or "mock-analyst")
