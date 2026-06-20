from __future__ import annotations

import json
import os
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODELS = {"deepseek-v4-flash", "deepseek-v4-pro"}
DEEPSEEK_ALIASES = {
    "flash": "deepseek-v4-flash",
    "v4-flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
    "v4-pro": "deepseek-v4-pro",
}


class LLMClient(Protocol):
    def complete_json(self, messages: list[dict[str, str]]) -> dict:
        ...


class OpenAILLMClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        load_dotenv()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def complete_json(self, messages: list[dict[str, str]]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)


class DeepSeekLLMClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
    ):
        load_dotenv()
        raw_model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.model = normalize_deepseek_model(raw_model)
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url or os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
        )

    def complete_json(self, messages: list[dict[str, str]]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)


class FakeLLMClient:
    def __init__(self, response: dict | None = None):
        self.response = response or {"next_action": {"name": "inspect", "args": {"target": "village"}}, "reason": "fake"}

    def complete_json(self, messages: list[dict[str, str]]) -> dict:
        del messages
        return self.response


def normalize_deepseek_model(model: str) -> str:
    normalized = DEEPSEEK_ALIASES.get(model, model)
    if normalized not in DEEPSEEK_MODELS:
        raise ValueError(f"Unsupported DeepSeek model: {model}. Use one of {sorted(DEEPSEEK_MODELS)}.")
    return normalized


def build_llm_client(provider: str | None = None) -> LLMClient:
    load_dotenv()
    selected = (provider or os.getenv("EXPERIENCE_GRAPH_LLM_PROVIDER", "openai")).lower()
    if selected == "openai":
        return OpenAILLMClient()
    if selected == "deepseek":
        return DeepSeekLLMClient()
    if selected == "fake":
        return FakeLLMClient()
    raise ValueError(f"Unsupported LLM provider: {selected}")
