from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experience_graph.agents.base import BaseAgent, extract_action_data
from experience_graph.agents.prompts import TEXTCRAFT_ACTION_GUIDE
from experience_graph.agents.memory_utils import append_jsonl, read_jsonl, summarize_experience
from experience_graph.core.models import Action, AgentInput, AgentOutput, ExperienceRecord
from experience_graph.llm.client import LLMClient


class ReflexionAgent(BaseAgent):
    def __init__(self, llm: LLMClient, memory_path: str | Path, max_reflections: int = 8):
        self.llm = llm
        self.memory_path = Path(memory_path)
        self.max_reflections = max_reflections
        self.reflections = read_jsonl(self.memory_path)

    def act(self, agent_input: AgentInput) -> AgentOutput:
        payload = self._ask(agent_input)
        action_data = extract_action_data(payload)
        if not isinstance(action_data, dict) or "name" not in action_data:
            return AgentOutput(action=None, rationale="invalid_llm_output")
        return AgentOutput(
            action=Action(name=action_data["name"], args=dict(action_data.get("args", {}))),
            rationale=payload.get("reason"),
            confidence=payload.get("confidence"),
        )

    def update(self, feedback: ExperienceRecord) -> None:
        reflection = {
            "episode_id": feedback.episode_id,
            "task_id": feedback.task_id,
            "success": feedback.success,
            "failure_reason": feedback.failure_reason,
            "reflection": self._make_reflection(feedback),
            "executed_path": list(feedback.executed_path),
        }
        self.reflections.append(reflection)
        append_jsonl(self.memory_path, reflection)

    def _ask(self, agent_input: AgentInput) -> dict:
        recent = self.reflections[-self.max_reflections :]
        messages = [
            {
                "role": "system",
                "content": "You are a Reflexion TextCraft-MC agent. Use prior natural-language reflections only as flat memory. " + TEXTCRAFT_ACTION_GUIDE,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": agent_input.task.id,
                        "observation": agent_input.observation.to_dict(),
                        "available_actions": [a.__dict__ for a in agent_input.available_actions],
                        "reflections": [item["reflection"] for item in recent],
                        "step_budget_remaining": agent_input.step_budget_remaining,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return self.llm.complete_json(messages)

    def _make_reflection(self, feedback: ExperienceRecord) -> str:
        summary = summarize_experience(feedback)
        if feedback.success:
            return summary + " Reuse this strategy only when the current state satisfies the same visible requirements."
        if feedback.failure_reason:
            return summary + f" Avoid repeating actions that trigger {feedback.failure_reason} unless the missing condition has changed."
        return summary





