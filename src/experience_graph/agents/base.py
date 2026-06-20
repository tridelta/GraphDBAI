from __future__ import annotations

from experience_graph.core.models import AgentInput, AgentOutput, ExperienceRecord


class BaseAgent:
    def act(self, agent_input: AgentInput) -> AgentOutput:
        raise NotImplementedError

    def update(self, feedback: ExperienceRecord) -> None:
        del feedback
