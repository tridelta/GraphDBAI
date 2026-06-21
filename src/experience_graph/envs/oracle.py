from __future__ import annotations

from dataclasses import dataclass

from experience_graph.core.models import Action, TaskSpec
from experience_graph.envs.textcraft import MyTextCraftAdapter


@dataclass
class OracleResult:
    case_id: str
    solvable: bool
    success: bool
    steps: int
    failure_reason: str | None


class CaseOracle:
    def __init__(self, env: MyTextCraftAdapter):
        self.env = env

    def run_case(self, case_id: str) -> OracleResult:
        case = self.env.cases[case_id]
        observation = self.env.reset(case_id=case_id)
        task = TaskSpec(id=case["task"]["id"])
        failure_reason = None
        for action_text in case["oracle"].get("reference_plan", []):
            result = self.env.step(Action.parse(action_text))
            observation = result.observation
            if not result.ok:
                failure_reason = result.failure_reason
                break
        success = self.env.is_success(observation, task)
        expected_solvable = bool(case["oracle"]["solvable"])
        if not success and failure_reason is None:
            failure_reason = case["oracle"].get("failure_reason")
        return OracleResult(
            case_id=case_id,
            solvable=expected_solvable,
            success=success,
            steps=observation.step_count,
            failure_reason=failure_reason,
        )


