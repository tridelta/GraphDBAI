from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
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


@dataclass
class LLMUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0
    token_source: str = "none"

    def snapshot(self) -> dict:
        return asdict(self)


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    text = " ".join(message.get("content", "") for message in messages)
    return max(1, len(text) // 4)


def estimate_text_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def usage_delta(before: dict, after: dict) -> dict:
    numeric_keys = [
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_prompt_tokens",
        "estimated_completion_tokens",
    ]
    delta = {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in numeric_keys}
    delta["token_source"] = after.get("token_source", "none")
    return delta


class OpenAILLMClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        load_dotenv()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.usage = LLMUsage()

    def complete_json(self, messages: list[dict[str, str]]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        self._record_usage(messages, content, response)
        return json.loads(content)

    def usage_snapshot(self) -> dict:
        return self.usage.snapshot()

    def _record_usage(self, messages: list[dict[str, str]], content: str, response) -> None:
        self.usage.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None and getattr(usage, "total_tokens", None) is not None:
            self.usage.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.usage.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            self.usage.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
            self.usage.token_source = "api"
            return
        prompt_estimate = estimate_message_tokens(messages)
        completion_estimate = estimate_text_tokens(content)
        self.usage.estimated_prompt_tokens += prompt_estimate
        self.usage.estimated_completion_tokens += completion_estimate
        self.usage.total_tokens += prompt_estimate + completion_estimate
        self.usage.token_source = "estimated"


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
        self.usage = LLMUsage()

    def complete_json(self, messages: list[dict[str, str]]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        self._record_usage(messages, content, response)
        return json.loads(content)

    def usage_snapshot(self) -> dict:
        return self.usage.snapshot()

    def _record_usage(self, messages: list[dict[str, str]], content: str, response) -> None:
        self.usage.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None and getattr(usage, "total_tokens", None) is not None:
            self.usage.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.usage.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            self.usage.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
            self.usage.token_source = "api"
            return
        prompt_estimate = estimate_message_tokens(messages)
        completion_estimate = estimate_text_tokens(content)
        self.usage.estimated_prompt_tokens += prompt_estimate
        self.usage.estimated_completion_tokens += completion_estimate
        self.usage.total_tokens += prompt_estimate + completion_estimate
        self.usage.token_source = "estimated"


class FakeLLMClient:
    def __init__(self, response: dict | None = None):
        self.response = response or {"next_action": {"name": "inspect", "args": {"target": "village"}}, "reason": "fake"}
        self.usage = LLMUsage(token_source="fake")

    def complete_json(self, messages: list[dict[str, str]]) -> dict:
        self.usage.calls += 1
        self.usage.estimated_prompt_tokens += estimate_message_tokens(messages)
        self.usage.estimated_completion_tokens += estimate_text_tokens(json.dumps(self.response))
        self.usage.total_tokens = self.usage.estimated_prompt_tokens + self.usage.estimated_completion_tokens
        return self.response

    def usage_snapshot(self) -> dict:
        return self.usage.snapshot()


def normalize_deepseek_model(model: str) -> str:
    normalized = DEEPSEEK_ALIASES.get(model, model)
    if normalized not in DEEPSEEK_MODELS:
        raise ValueError(f"Unsupported DeepSeek model: {model}. Use one of {sorted(DEEPSEEK_MODELS)}.")
    return normalized


def build_llm_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    load_dotenv()
    selected = (provider or os.getenv("EXPERIENCE_GRAPH_LLM_PROVIDER", "openai")).lower()
    if selected == "openai":
        return OpenAILLMClient(model=model)
    if selected == "deepseek":
        return DeepSeekLLMClient(model=model)
    if selected == "fake":
        return FakeLLMClient()
    raise ValueError(f"Unsupported LLM provider: {selected}")

