from __future__ import annotations

import json

from experience_graph.agents.react import ReActAgent
from experience_graph.core.models import AgentInput
from experience_graph.llm.client import LLMClient


class ExperienceGraphAgent(ReActAgent):
    def __init__(self, llm: LLMClient):
        super().__init__(llm)

    def _ask(self, agent_input: AgentInput) -> dict:
        view = agent_input.experience_view
        messages = [
            {
                "role": "system",
                "content": "You are an ExperienceGraph TextCraft-MC agent. Choose one next action. Return JSON.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": agent_input.task.id,
                        "observation": agent_input.observation.to_dict(),
                        "available_actions": [a.__dict__ for a in agent_input.available_actions],
                        "candidate_paths": [
                            {
                                "label": c.label,
                                "applicability": c.applicability,
                                "success_rate": c.success_rate,
                                "evidence": c.evidence,
                                "missing_conditions": [m.__dict__ for m in c.missing_conditions],
                            }
                            for c in view.candidate_paths
                        ],
                        "warnings": view.warnings,
                        "step_budget_remaining": agent_input.step_budget_remaining,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return self.llm.complete_json(messages)
