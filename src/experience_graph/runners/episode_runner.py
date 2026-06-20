from __future__ import annotations

from dataclasses import dataclass

from experience_graph.core.experience import ExperienceBuilder
from experience_graph.core.models import AgentInput, ExperienceRecord, StepRecord, TaskSpec
from experience_graph.envs.textcraft import TextCraftAdapter
from experience_graph.evaluation.logger import EvaluationLogger
from experience_graph.graph.organizer import GraphOrganizer
from experience_graph.graph.retriever import GraphRetriever


@dataclass
class EpisodeResult:
    experience: ExperienceRecord
    metrics: dict


class EpisodeRunner:
    def __init__(
        self,
        env: TextCraftAdapter,
        agent,
        organizer: GraphOrganizer,
        retriever: GraphRetriever,
        logger: EvaluationLogger,
        max_steps: int = 30,
    ):
        self.env = env
        self.agent = agent
        self.organizer = organizer
        self.retriever = retriever
        self.logger = logger
        self.max_steps = max_steps
        self.builder = ExperienceBuilder()

    def run_episode(self, episode_id: str, case_id: str) -> EpisodeResult:
        case = self.env.cases[case_id]
        task = TaskSpec(id=case["task"]["id"])
        initial = self.env.reset(task=task, case_id=case_id)
        observation = initial
        trajectory: list[StepRecord] = []
        success = False
        failure_reason = None

        for step_index in range(self.max_steps):
            conditions = self.env.extract_conditions(observation)
            view = self.retriever.retrieve(task, observation, conditions)
            self.logger.write_jsonl("experience_views.jsonl", view)
            agent_input = AgentInput(
                task=task,
                observation=observation,
                available_actions=self.env.available_actions(observation),
                experience_view=view,
                step_budget_remaining=self.max_steps - step_index,
            )
            output = self.agent.act(agent_input)
            if output.action is None:
                failure_reason = case.get("oracle", {}).get("failure_reason") or output.rationale or "no_action"
                break
            before = observation
            result = self.env.step(output.action)
            observation = result.observation
            trajectory.append(
                StepRecord(
                    step_index=step_index,
                    observation_before=before,
                    experience_view_id=view.view_id,
                    agent_thought=output.rationale,
                    action=output.action,
                    result=result,
                    observation_after=observation,
                )
            )
            if result.done:
                success = True
                break
            if not result.ok:
                failure_reason = result.failure_reason
                break
        if not success and failure_reason is None:
            failure_reason = "step_budget_exhausted"

        experience = self.builder.build(episode_id, task.id, initial, trajectory, success)
        if failure_reason and not experience.failure_reason:
            experience.failure_reason = failure_reason
            experience.metrics["failure_reason"] = failure_reason
        update = self.organizer.integrate(experience)
        graph_summary = self.organizer.store.summary()
        metrics = {
            "episode_id": episode_id,
            "case_id": case_id,
            "task": task.id,
            "success": success,
            "steps": len(trajectory),
            "failure_reason": experience.failure_reason,
            "graph_nodes": graph_summary["nodes"],
            "graph_edges": graph_summary["edges"],
            "graph_paths": graph_summary["paths"],
            "added_nodes": update.added_nodes,
            "added_edges": update.added_edges,
            "updated_edges": update.updated_edges,
        }
        experience.metrics.update(metrics)
        self.logger.write_jsonl("episodes.jsonl", experience)
        self.logger.write_jsonl("metrics.jsonl", metrics)
        self.agent.update(experience)
        return EpisodeResult(experience=experience, metrics=metrics)

