from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experience_graph.agents.base import BaseAgent, extract_action_data
from experience_graph.agents.prompts import TEXTCRAFT_ACTION_GUIDE
from experience_graph.agents.memory_utils import append_jsonl, jaccard, observation_features, read_jsonl, summarize_experience
from experience_graph.core.models import Action, AgentInput, AgentOutput, ExperienceRecord
from experience_graph.llm.client import LLMClient


class VectorTrajectoryAgent(BaseAgent):
    def __init__(self, llm: LLMClient, memory_path: str | Path, top_k: int = 5):
        self.llm = llm
        self.memory_path = Path(memory_path)
        self.top_k = top_k
        self.records = read_jsonl(self.memory_path)

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
        row = {
            "episode_id": feedback.episode_id,
            "task_id": feedback.task_id,
            "success": feedback.success,
            "failure_reason": feedback.failure_reason,
            "features": sorted(observation_features(feedback.initial_observation)),
            "trajectory_text": summarize_experience(feedback),
            "executed_path": list(feedback.executed_path),
        }
        self.records.append(row)
        append_jsonl(self.memory_path, row)

    def _ask(self, agent_input: AgentInput) -> dict:
        query_features = observation_features(agent_input.observation)
        retrieved = self._retrieve(query_features, agent_input.task.id)
        messages = [
            {
                "role": "system",
                "content": "You are a VectorTrajectory TextCraft-MC agent. Use retrieved similar trajectories as flat trajectory memory, not as a graph. " + TEXTCRAFT_ACTION_GUIDE,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": agent_input.task.id,
                        "observation": agent_input.observation.to_dict(),
                        "available_actions": [a.__dict__ for a in agent_input.available_actions],
                        "retrieved_trajectories": retrieved,
                        "step_budget_remaining": agent_input.step_budget_remaining,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return self.llm.complete_json(messages)

    def _retrieve(self, query_features: set[str], task_id: str) -> list[dict[str, Any]]:
        scored = []
        for row in self.records:
            if row.get("task_id") != task_id:
                continue
            score = jaccard(query_features, set(row.get("features", [])))
            scored.append((score, row))
        scored.sort(key=lambda item: (item[0], bool(item[1].get("success"))), reverse=True)
        return [
            {
                "episode_id": row.get("episode_id"),
                "similarity": round(score, 3),
                "success": row.get("success"),
                "failure_reason": row.get("failure_reason"),
                "trajectory": row.get("trajectory_text"),
            }
            for score, row in scored[: self.top_k]
        ]





