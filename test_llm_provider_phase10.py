"""Comprehensive test suite for Phase 10: Provider-Agnostic LLM Layer & Dynamic Context Assembly."""
import json
import pandas as pd
import pytest

from agent.llm_router import LLMRouter
from backend.app.core.dynamic_context import DynamicContextAssembler, DynamicPromptContext
from backend.app.core.llm_provider import (
    AnthropicProvider,
    DeepSeekProvider,
    LLMClientFactory,
    LLMMessage,
    LLMProviderType,
    LLMResponse,
    MockDeterministicProvider,
    OllamaProvider,
    OpenAIProvider,
)


# ==============================================================================
# 1. LLM Provider Factory & Provider Instantiation Tests
# ==============================================================================

def test_llm_client_factory():
    """Verify factory instantiation across OpenAI, Anthropic, Ollama, DeepSeek, and Mock."""
    p_mock = LLMClientFactory.get_provider("mock")
    assert isinstance(p_mock, MockDeterministicProvider)
    assert p_mock.provider_type == LLMProviderType.MOCK

    p_openai = LLMClientFactory.get_provider("openai", api_key="sk-test", model="gpt-4o")
    assert isinstance(p_openai, OpenAIProvider)
    assert p_openai.model == "gpt-4o"

    p_claude = LLMClientFactory.get_provider("anthropic", api_key="sk-ant-test", model="claude-3-5-sonnet")
    assert isinstance(p_claude, AnthropicProvider)

    p_ollama = LLMClientFactory.get_provider("ollama", model="llama3")
    assert isinstance(p_ollama, OllamaProvider)

    p_deepseek = LLMClientFactory.get_provider("deepseek", api_key="sk-ds-test")
    assert isinstance(p_deepseek, DeepSeekProvider)


def test_mock_deterministic_provider_responses():
    """Verify that MockDeterministicProvider emits valid structured JSON plans."""
    provider = MockDeterministicProvider()

    # Prediction query
    res_pred = provider.generate([LLMMessage(role="user", content="Build a machine learning model to predict sales")])
    assert isinstance(res_pred, LLMResponse)
    data_pred = json.loads(res_pred.content)
    assert data_pred["action"] == "model_selection"

    # Neural network query
    res_ann = provider.generate([LLMMessage(role="user", content="Train a deep neural network ANN")])
    data_ann = json.loads(res_ann.content)
    assert data_ann["action"] == "ann"

    # CNN query
    res_cnn = provider.generate([LLMMessage(role="user", content="Train a CNN on the image spatial dataset")])
    data_cnn = json.loads(res_cnn.content)
    assert data_cnn["action"] == "cnn"


# ==============================================================================
# 2. Dynamic Context Assembly & Guardrail Tests
# ==============================================================================

def test_dynamic_context_assembly_grounding_guardrails():
    """Verify prompt context assembly, dataset profiling, and anti-hallucination guardrails."""
    assembler = DynamicContextAssembler()

    df = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4"],
        "monthly_revenue": [1200.0, 3400.0, 5600.0, 2100.0],
        "is_active": [1, 1, 0, 1],
    })

    validation_alerts = [
        {"severity": "WARNING", "title": "Class Imbalance", "description": "Minority class is 25%."}
    ]

    agent_findings = [
        {"agent": "ModelSelectionAgent", "output": {"best_model": {"model_name": "Random Forest", "primary_metric_name": "R2", "primary_metric_value": 0.89}}}
    ]

    prompt_ctx: DynamicPromptContext = assembler.assemble(
        query="What is driving monthly revenue?",
        dataframe=df,
        validation_issues=validation_alerts,
        agent_outputs=agent_findings,
    )

    assert isinstance(prompt_ctx, DynamicPromptContext)
    assert len(prompt_ctx.messages) >= 2
    assert prompt_ctx.estimated_tokens > 50

    # Verify hard anti-hallucination guardrails in system prompt
    sys_content = prompt_ctx.system_prompt
    assert "DETERMINISTIC CALCULATIONS ONLY" in sys_content
    assert "NEVER invent, extrapolate, or hallucinate numerical results" in sys_content
    assert "Never present correlation as causation" in sys_content

    # Verify dataset profile and agents list
    assert "4 rows x 3 columns" in sys_content
    assert "monthly_revenue" in sys_content
    assert "ModelSelectionAgent" in sys_content
    assert "Random Forest" in sys_content
    assert "Class Imbalance" in sys_content


# ==============================================================================
# 3. LLMRouter Integration Tests
# ==============================================================================

def test_llm_router_execution():
    """Verify LLMRouter converting natural language to task plans via mock provider."""
    mock_provider = MockDeterministicProvider()
    router = LLMRouter(provider=mock_provider)

    # Route prediction command
    routed = router.route("Predict customer monthly revenue")
    assert routed["source"] == "llm"
    assert routed["task"]["action"] == "model_selection"

    # Route forecasting command
    routed_fc = router.route("Forecast next 6 periods")
    assert routed_fc["source"] == "llm"
    assert routed_fc["task"]["action"] == "forecast"


def test_llm_router_json_extraction():
    """Verify robust extraction of JSON from responses wrapped in markdown code fences."""
    router = LLMRouter()

    raw_fenced = '```json\n{"action": "clean", "parameters": {"strategy": "auto"}}\n```'
    extracted = router._extract_json(raw_fenced)
    assert extracted == {"action": "clean", "parameters": {"strategy": "auto"}}

    raw_text_wrap = 'Here is the requested plan:\n{"action": "summary"}\nHope this helps!'
    extracted_text = router._extract_json(raw_text_wrap)
    assert extracted_text == {"action": "summary"}

