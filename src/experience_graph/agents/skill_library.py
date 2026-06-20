from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experience_graph.agents.base import BaseAgent, extract_action_data
from experience_graph.agents.prompts import TEXTCRAFT_ACTION_GUIDE
from experience_graph.agents.memory_utils import append_jsonl, jaccard, observation_features, read_jsonl
from experience_graph.core.models import Action, AgentInput, AgentOutput, ExperienceRecord
from experience_graph.llm.client import LLMClient


class SkillLibraryAgent(BaseAgent):
    def __init__(self, llm: LLMClient, memory_path: str | Path, top_k: int = 5):
        self.llm = llm
        self.memory_path = Path(memory_path)
        self.top_k = top_k
        self.skills = read_jsonl(self.memory_path)

    def act(self, agent_input: AgentInput) -> AgentOutput:
        payload = self._ask(agent_input)
        action_data = extract_action_data(payload)
        if not isinstance(action_data, dict) or "name" not in action_data:
            return AgentOutput(action=None, rationale="invalid_llm_output")
        plan = None
        if isinstance(payload.get("adapted_plan"), list):
            plan = [Action.parse(str(item)) if not isinstance(item, dict) else Action(name=item.get("name", "unknown"), args=dict(item.get("args", {}))) for item in payload["adapted_plan"]]
        return AgentOutput(
            action=Action(name=action_data["name"], args=dict(action_data.get("args", {}))),
            plan=plan,
            rationale=payload.get("reason"),
            confidence=payload.get("confidence"),
        )

    def update(self, feedback: ExperienceRecord) -> None:
        if not feedback.success or not feedback.executed_path:
            return
        row = {
            "skill_id": f"skill_{len(self.skills):06d}",
            "episode_id": feedback.episode_id,
            "task_id": feedback.task_id,
            "features": sorted(observation_features(feedback.initial_observation)),
            "action_sequence": list(feedback.executed_path),
            "steps": len(feedback.trajectory),
        }
        self.skills.append(row)
        append_jsonl(self.memory_path, row)

    def _ask(self, agent_input: AgentInput) -> dict:
        query_features = observation_features(agent_input.observation)
        retrieved = self._retrieve(query_features, agent_input.task.id)
        messages = [
            {
                "role": "system",
                "content": "You are a SkillLibrary TextCraft-MC agent. Reuse or adapt successful action-sequence skills when applicable. " + TEXTCRAFT_ACTION_GUIDE,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": agent_input.task.id,
                        "observation": agent_input.observation.to_dict(),
                        "available_actions": [a.__dict__ for a in agent_input.available_actions],
                        "retrieved_skills": retrieved,
                        "step_budget_remaining": agent_input.step_budget_remaining,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return self.llm.complete_json(messages, validator=lambda payload: extract_action_data(payload) is not None)

    def _retrieve(self, query_features: set[str], task_id: str) -> list[dict[str, Any]]:
        scored = []
        for row in self.skills:
            if row.get("task_id") != task_id:
                continue
            scored.append((jaccard(query_features, set(row.get("features", []))), row))
        scored.sort(key=lambda item: (item[0], -int(item[1].get("steps", 999))), reverse=True)
        return [
            {
                "skill_id": row.get("skill_id"),
                "similarity": round(score, 3),
                "action_sequence": row.get("action_sequence", []),
                "steps": row.get("steps"),
            }
            for score, row in scored[: self.top_k]
        ]





