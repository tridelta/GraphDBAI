from __future__ import annotations

import json

from experience_graph.agents.base import extract_action_data
from experience_graph.agents.prompts import TEXTCRAFT_ACTION_GUIDE
from experience_graph.agents.react import ReActAgent
from experience_graph.core.models import Action, AgentInput, AgentOutput
from experience_graph.llm.client import LLMClient


class ExperienceGraphAgent(ReActAgent):
    def __init__(self, llm: LLMClient, propose_new_path: bool = True):
        super().__init__(llm)
        self.propose_new_path = propose_new_path

    def act(self, agent_input: AgentInput) -> AgentOutput:
        payload = self._ask(agent_input)
        action_data = extract_action_data(payload)
        if not isinstance(action_data, dict) or "name" not in action_data:
            return AgentOutput(action=None, rationale="invalid_llm_output")
        plan = self._parse_plan(payload.get("selected_plan") or payload.get("novel_candidate_path"))
        return AgentOutput(
            action=Action(name=action_data["name"], args=dict(action_data.get("args", {}))),
            plan=plan,
            rationale=payload.get("reason"),
            confidence=payload.get("confidence"),
        )

    def _ask(self, agent_input: AgentInput) -> dict:
        view = agent_input.experience_view
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an ExperienceGraph TextCraft-MC agent. "
                    + TEXTCRAFT_ACTION_GUIDE
                    + " If exploration is enabled, first propose one novel candidate path, then compare it with graph candidate paths."
                    + " Return JSON with novel_candidate_path, selected_strategy, selected_plan, next_action, reason, confidence."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": agent_input.task.id,
                        "observation": agent_input.observation.to_dict(),
                        "available_actions": [a.__dict__ for a in agent_input.available_actions],
                        "exploration_enabled": self.propose_new_path,
                        "candidate_paths": [
                            {
                                "path_id": c.path_id,
                                "label": c.label,
                                "applicability": c.applicability,
                                "route_type": c.route_type,
                                "success_rate": c.success_rate,
                                "attempts": c.attempts,
                                "avg_steps": c.avg_steps,
                                "evidence": c.evidence,
                                "state_similarity": c.state_similarity,
                                "retrieval_score": c.retrieval_score,
                                "retrieval_reason": c.retrieval_reason,
                                "matched_conditions": [m.__dict__ for m in c.matched_conditions],
                                "missing_conditions": [m.__dict__ for m in c.missing_conditions],
                                "suggested_probe_actions": [a.label() for a in c.suggested_probe_actions],
                                "common_failures": c.common_failures,
                                "actions_preview": [a.label() for a in c.actions_preview],
                            }
                            for c in view.candidate_paths
                        ],
                        "warnings": view.warnings,
                        "summaries": view.summaries,
                        "step_budget_remaining": agent_input.step_budget_remaining,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return self.llm.complete_json(messages)

    def _parse_plan(self, raw) -> list[Action] | None:
        if not isinstance(raw, list):
            return None
        plan: list[Action] = []
        for item in raw:
            if isinstance(item, dict) and "name" in item:
                plan.append(Action(name=item["name"], args=dict(item.get("args", {}))))
            elif isinstance(item, str):
                plan.append(Action.parse(item))
        return plan or None






