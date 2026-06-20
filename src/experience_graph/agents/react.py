from __future__ import annotations

import json

from experience_graph.agents.base import BaseAgent, extract_action_data
from experience_graph.agents.prompts import TEXTCRAFT_ACTION_GUIDE
from experience_graph.core.models import Action, AgentInput, AgentOutput
from experience_graph.llm.client import LLMClient


class ReActAgent(BaseAgent):
    def __init__(self, llm: LLMClient):
        self.llm = llm

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

    def _ask(self, agent_input: AgentInput) -> dict:
        messages = [
            {"role": "system", "content": "You are a TextCraft-MC agent. " + TEXTCRAFT_ACTION_GUIDE},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": agent_input.task.id,
                        "observation": agent_input.observation.to_dict(),
                        "available_actions": [a.__dict__ for a in agent_input.available_actions],
                        "experience_view": agent_input.experience_view.view_id,
                        "step_budget_remaining": agent_input.step_budget_remaining,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return self.llm.complete_json(messages)





