from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

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
RETRY_INSTRUCTION = (
    "The previous response was invalid for this task. Return JSON only with a valid next_action object copied "
    "from available_actions, for example {\"next_action\":{\"name\":\"move_to\",\"args\":{\"location\":\"village\"}},"
    "\"reason\":\"short rationale\",\"confidence\":0.7}. Do not return an empty object."
)


class LLMClient(Protocol):
    def complete_json(self, messages: list[dict[str, str]], validator: Callable[[dict[str, Any]], bool] | None = None) -> dict:
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


def parse_json_response(content: str) -> tuple[dict[str, Any], str | None]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return {}, str(exc)
    return parsed if isinstance(parsed, dict) else {}, None if isinstance(parsed, dict) else "response_json_is_not_object"


def response_is_valid(parsed: dict[str, Any], parse_error: str | None, validator: Callable[[dict[str, Any]], bool] | None) -> bool:
    if parse_error is not None or parsed == {}:
        return False
    if validator is not None and not validator(parsed):
        return False
    return True


def retry_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [*messages, {"role": "user", "content": RETRY_INSTRUCTION}]


class OpenAILLMClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        retry_max_tokens: int | None = None,
        retries: int = 0,
    ):
        load_dotenv()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.max_tokens = max_tokens
        self.retry_max_tokens = retry_max_tokens or max_tokens
        self.max_retries = max(0, retries)
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.usage = LLMUsage()
        self.last_trace: dict = {}

    def complete_json(self, messages: list[dict[str, str]], validator: Callable[[dict[str, Any]], bool] | None = None) -> dict:
        return self._complete_json_with_retries(messages, validator)

    def usage_snapshot(self) -> dict:
        return self.usage.snapshot()

    def _complete_json_with_retries(self, messages: list[dict[str, str]], validator: Callable[[dict[str, Any]], bool] | None) -> dict:
        attempts = []
        active_messages = messages
        final_parsed: dict[str, Any] = {}
        for attempt_index in range(self.max_retries + 1):
            attempt_max_tokens = self.max_tokens if attempt_index == 0 else self.retry_max_tokens
            response = self._request(active_messages, attempt_max_tokens)
            content = response.choices[0].message.content or "{}"
            self._record_usage(active_messages, content, response)
            parsed, parse_error = parse_json_response(content)
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            valid = response_is_valid(parsed, parse_error, validator)
            attempts.append(
                {
                    "attempt_index": attempt_index,
                    "model": self.model,
                    "messages": active_messages,
                    "raw_response": content,
                    "response": parsed,
                    "parse_error": parse_error,
                    "finish_reason": finish_reason,
                    "max_tokens": attempt_max_tokens,
                    "valid": valid,
                }
            )
            final_parsed = parsed
            if valid or attempt_index >= self.max_retries:
                break
            active_messages = retry_messages(messages)
        final = attempts[-1]
        self.last_trace = {**final, "attempts": attempts, "retry_count": len(attempts) - 1}
        return final_parsed

    def _request(self, messages: list[dict[str, str]], max_tokens: int | None):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return self.client.chat.completions.create(**kwargs)

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


class DeepSeekLLMClient(OpenAILLMClient):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        retry_max_tokens: int | None = None,
        retries: int = 0,
    ):
        load_dotenv()
        raw_model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.model = normalize_deepseek_model(raw_model)
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("DEEPSEEK_MAX_TOKENS", "4096"))
        self.retry_max_tokens = retry_max_tokens if retry_max_tokens is not None else int(os.getenv("DEEPSEEK_RETRY_MAX_TOKENS", "8192"))
        self.max_retries = max(0, retries)
        self.client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url or os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
        )
        self.usage = LLMUsage()
        self.last_trace: dict = {}

    def _request(self, messages: list[dict[str, str]], max_tokens: int | None):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            max_tokens=max_tokens or self.max_tokens,
        )


class FakeLLMClient:
    def __init__(self, response: dict | None = None):
        self.response = response if response is not None else {"next_action": {"name": "inspect", "args": {"target": "village"}}, "reason": "fake"}
        self.usage = LLMUsage(token_source="fake")
        self.last_trace: dict = {}

    def complete_json(self, messages: list[dict[str, str]], validator: Callable[[dict[str, Any]], bool] | None = None) -> dict:
        self.usage.calls += 1
        self.usage.estimated_prompt_tokens += estimate_message_tokens(messages)
        self.usage.estimated_completion_tokens += estimate_text_tokens(json.dumps(self.response))
        self.usage.total_tokens = self.usage.estimated_prompt_tokens + self.usage.estimated_completion_tokens
        valid = response_is_valid(self.response, None, validator)
        self.last_trace = {
            "model": "fake",
            "messages": messages,
            "raw_response": json.dumps(self.response, ensure_ascii=False),
            "response": self.response,
            "parse_error": None,
            "finish_reason": "fake",
            "max_tokens": None,
            "valid": valid,
            "attempts": [],
            "retry_count": 0,
        }
        return self.response

    def usage_snapshot(self) -> dict:
        return self.usage.snapshot()


def normalize_deepseek_model(model: str) -> str:
    normalized = DEEPSEEK_ALIASES.get(model, model)
    if normalized not in DEEPSEEK_MODELS:
        raise ValueError(f"Unsupported DeepSeek model: {model}. Use one of {sorted(DEEPSEEK_MODELS)}.")
    return normalized


def build_llm_client(
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    retry_max_tokens: int | None = None,
    retries: int = 0,
) -> LLMClient:
    load_dotenv()
    selected = (provider or os.getenv("EXPERIENCE_GRAPH_LLM_PROVIDER", "openai")).lower()
    if selected == "openai":
        return OpenAILLMClient(model=model, max_tokens=max_tokens, retry_max_tokens=retry_max_tokens, retries=retries)
    if selected == "deepseek":
        return DeepSeekLLMClient(model=model, max_tokens=max_tokens, retry_max_tokens=retry_max_tokens, retries=retries)
    if selected == "fake":
        return FakeLLMClient()
    raise ValueError(f"Unsupported LLM provider: {selected}")
