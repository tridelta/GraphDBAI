from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .models import Action, Condition


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return value


def condition_from_dict(data: dict[str, Any]) -> Condition:
    return Condition(
        key=data["key"],
        operator=data.get("operator", "=="),
        value=data.get("value"),
        source=data.get("source", "env"),
        confidence=float(data.get("confidence", 1.0)),
    )


def action_from_dict(data: dict[str, Any]) -> Action:
    return Action(name=data["name"], args=dict(data.get("args", {})))
