from __future__ import annotations

from typing import Any

from experience_graph.core.models import AgentInput, AgentOutput, ExperienceRecord


class BaseAgent:
    def act(self, agent_input: AgentInput) -> AgentOutput:
        raise NotImplementedError

    def update(self, feedback: ExperienceRecord) -> None:
        del feedback


def extract_action_data(payload: dict[str, Any]) -> dict[str, Any] | None:
    action_data = payload.get("next_action") or payload.get("action")
    if isinstance(action_data, dict):
        return action_data
    if "name" in payload:
        return payload
    return None
