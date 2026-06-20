from __future__ import annotations

import pytest

from experience_graph.llm.client import DEEPSEEK_BASE_URL, DeepSeekLLMClient, build_llm_client, normalize_deepseek_model


def test_deepseek_model_aliases():
    assert normalize_deepseek_model("flash") == "deepseek-v4-flash"
    assert normalize_deepseek_model("v4-pro") == "deepseek-v4-pro"
    assert normalize_deepseek_model("deepseek-v4-flash") == "deepseek-v4-flash"


def test_deepseek_rejects_unknown_model():
    with pytest.raises(ValueError):
        normalize_deepseek_model("deepseek-chat")


def test_build_deepseek_client_from_env(monkeypatch):
    monkeypatch.setenv("EXPERIENCE_GRAPH_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "pro")
    client = build_llm_client()
    assert isinstance(client, DeepSeekLLMClient)
    assert client.model == "deepseek-v4-pro"


def test_deepseek_default_base_url():
    client = DeepSeekLLMClient(api_key="test-key", model="flash")
    assert client.model == "deepseek-v4-flash"
    assert DEEPSEEK_BASE_URL == "https://api.deepseek.com"
