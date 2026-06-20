from __future__ import annotations

from experience_graph.agents.base import BaseAgent
from experience_graph.core.models import Action, AgentInput, AgentOutput


class ScriptedAgent(BaseAgent):
    def __init__(self, plans_by_case: dict[str, list[str]]):
        self.plans_by_case = plans_by_case
        self.positions: dict[str, int] = {}

    def act(self, agent_input: AgentInput) -> AgentOutput:
        case_id = agent_input.observation.case_id or "default"
        plan = self.plans_by_case.get(case_id, [])
        index = self.positions.get(case_id, 0)
        if index >= len(plan):
            return AgentOutput(action=None, rationale="scripted plan exhausted")
        self.positions[case_id] = index + 1
        return AgentOutput(action=Action.parse(plan[index]), plan=[Action.parse(p) for p in plan], rationale="oracle reference plan")
