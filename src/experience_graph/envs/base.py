from __future__ import annotations

from typing import Protocol

from experience_graph.core.models import Action, ActionSpec, Condition, Observation, StepResult, TaskSpec


class EnvironmentAdapter(Protocol):
    def reset(self, seed: int = 0, task: TaskSpec | None = None, case_id: str | None = None) -> Observation:
        ...

    def available_actions(self, observation: Observation) -> list[ActionSpec]:
        ...

    def step(self, action: Action) -> StepResult:
        ...

    def is_success(self, observation: Observation, task: TaskSpec) -> bool:
        ...

    def extract_conditions(self, observation: Observation) -> list[Condition]:
        ...
