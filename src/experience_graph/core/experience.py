from __future__ import annotations

from experience_graph.core.models import Condition, ExperienceRecord, Observation, StepRecord


class ExperienceBuilder:
    def build(
        self,
        episode_id: str,
        task_id: str,
        initial_observation: Observation,
        trajectory: list[StepRecord],
        success: bool,
        proposed_plan=None,
    ) -> ExperienceRecord:
        final_observation = trajectory[-1].observation_after if trajectory else initial_observation
        discovered: list[Condition] = []
        failure_reason = None
        executed_path: list[str] = []
        for step in trajectory:
            executed_path.append(step.action.label())
            discovered.extend(step.result.revealed_conditions)
            if not success and not step.result.ok:
                failure_reason = step.result.failure_reason
        metrics = {
            "steps": len(trajectory),
            "success": success,
            "failure_reason": failure_reason,
            "discovered_conditions": len(discovered),
        }
        return ExperienceRecord(
            episode_id=episode_id,
            task_id=task_id,
            initial_observation=initial_observation,
            final_observation=final_observation,
            trajectory=trajectory,
            success=success,
            failure_reason=failure_reason,
            discovered_conditions=discovered,
            proposed_plan=proposed_plan,
            executed_path=executed_path,
            metrics=metrics,
        )
