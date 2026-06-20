from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from experience_graph.core.experience import ExperienceBuilder
from experience_graph.core.models import AgentInput, ExperienceRecord, StepRecord, TaskSpec
from experience_graph.core.serialization import to_jsonable
from experience_graph.envs.textcraft import TextCraftAdapter
from experience_graph.evaluation.logger import EvaluationLogger
from experience_graph.graph.organizer import GraphOrganizer
from experience_graph.graph.retriever import GraphRetriever
from experience_graph.llm.client import usage_delta


@dataclass
class EpisodeResult:
    experience: ExperienceRecord
    metrics: dict


@dataclass
class RunContext:
    run_id: str | None = None
    agent: str | None = None
    variant: str | None = None
    seed: int | None = None
    difficulty: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "agent": self.agent,
            "variant": self.variant,
            "seed": self.seed,
            "difficulty": self.difficulty,
        }
        payload.update(self.metadata)
        return {key: value for key, value in payload.items() if value is not None}


class EpisodeRunner:
    def __init__(
        self,
        env: TextCraftAdapter,
        agent,
        organizer: GraphOrganizer,
        retriever: GraphRetriever,
        logger: EvaluationLogger,
        max_steps: int = 30,
        run_context: RunContext | None = None,
    ):
        self.env = env
        self.agent = agent
        self.organizer = organizer
        self.retriever = retriever
        self.logger = logger
        self.max_steps = max_steps
        self.builder = ExperienceBuilder()
        self.run_context = run_context or RunContext()

    def run_episode(
        self,
        episode_id: str,
        case_id: str,
        seed: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> EpisodeResult:
        case = self.env.cases[case_id]
        task = TaskSpec(id=case["task"]["id"])
        episode_seed = self.run_context.seed if seed is None else seed
        initial = self.env.reset(seed=episode_seed or 0, task=task, case_id=case_id)
        observation = initial
        trajectory: list[StepRecord] = []
        proposed_plan = None
        success = False
        failure_reason = None
        episode_context = self._episode_context(case, episode_seed, context)
        episode_usage_before = self._usage_snapshot()

        for step_index in range(self.max_steps):
            conditions = self.env.extract_conditions(observation)
            view = self.retriever.retrieve(task, observation, conditions)
            self.logger.write_jsonl("experience_views.jsonl", {**episode_context, "step_index": step_index, "view": view})
            agent_input = AgentInput(
                task=task,
                observation=observation,
                available_actions=self.env.available_actions(observation),
                experience_view=view,
                step_budget_remaining=self.max_steps - step_index,
            )
            usage_before = self._usage_snapshot()
            output = self.agent.act(agent_input)
            usage_after = self._usage_snapshot()
            llm_step_usage = usage_delta(usage_before, usage_after)
            llm_trace = self._last_llm_trace()
            if output.plan is not None:
                proposed_plan = output.plan
            if output.action is None:
                failure_reason = case.get("oracle", {}).get("failure_reason") or output.rationale or "no_action"
                self.logger.write_jsonl(
                    "steps.jsonl",
                    {
                        **episode_context,
                        "episode_id": episode_id,
                        "case_id": case_id,
                        "task": task.id,
                        "step_index": step_index,
                        "experience_view_id": view.view_id,
                        "candidate_paths": len(view.candidate_paths),
                        "experience_view_tokens": view.token_estimate,
                        "action": None,
                        "agent_rationale": output.rationale,
                        "agent_confidence": output.confidence,
                        "ok": False,
                        "done": False,
                        "failure_reason": failure_reason,
                        "llm_usage_delta": llm_step_usage,
                        "llm_model": llm_trace.get("model"),
                        "llm_input_messages": llm_trace.get("messages", []),
                        "llm_output": llm_trace.get("response"),
                        "llm_raw_response": llm_trace.get("raw_response"),
                    },
                )
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
            self.logger.write_jsonl(
                "steps.jsonl",
                {
                    **episode_context,
                    "episode_id": episode_id,
                    "case_id": case_id,
                    "task": task.id,
                    "step_index": step_index,
                    "experience_view_id": view.view_id,
                    "candidate_paths": len(view.candidate_paths),
                    "experience_view_tokens": view.token_estimate,
                    "action": output.action,
                    "agent_rationale": output.rationale,
                    "agent_confidence": output.confidence,
                    "ok": result.ok,
                    "done": result.done,
                    "reward": result.reward,
                    "cost": result.cost,
                    "failure_reason": result.failure_reason,
                    "revealed_conditions": result.revealed_conditions,
                    "state_delta": result.state_delta,
                    "llm_usage_delta": llm_step_usage,
                    "llm_model": llm_trace.get("model"),
                    "llm_input_messages": llm_trace.get("messages", []),
                    "llm_output": llm_trace.get("response"),
                    "llm_raw_response": llm_trace.get("raw_response"),
                },
            )
            if result.done:
                success = True
                break
            if not result.ok:
                failure_reason = result.failure_reason
                break
        if not success and failure_reason is None:
            failure_reason = "step_budget_exhausted"

        experience = self.builder.build(episode_id, task.id, initial, trajectory, success, proposed_plan=proposed_plan)
        if failure_reason and not experience.failure_reason:
            experience.failure_reason = failure_reason
            experience.metrics["failure_reason"] = failure_reason
        update = self.organizer.integrate(experience)
        graph_summary = self.organizer.store.summary()
        episode_usage_after = self._usage_snapshot()
        llm_episode_usage = usage_delta(episode_usage_before, episode_usage_after)
        metrics = {
            **episode_context,
            "episode_id": episode_id,
            "case_id": case_id,
            "task": task.id,
            "success": success,
            "steps": len(trajectory),
            "failure_reason": experience.failure_reason,
            "initial_state_hash": self._stable_hash(initial.to_dict()),
            "final_state_hash": self._stable_hash(experience.final_observation.to_dict()),
            "discovered_conditions": len(experience.discovered_conditions),
            "graph_nodes": graph_summary["nodes"],
            "graph_edges": graph_summary["edges"],
            "graph_paths": graph_summary["paths"],
            "dormant_edges": graph_summary.get("dormant_edges", 0),
            "added_nodes": update.added_nodes,
            "added_edges": update.added_edges,
            "updated_edges": update.updated_edges,
            "added_paths": update.added_paths,
            "llm_usage_delta": llm_episode_usage,
            "llm_usage_cumulative": episode_usage_after,
        }
        experience.metrics.update(metrics)
        self.logger.write_jsonl("episodes.jsonl", experience)
        self.logger.write_jsonl("metrics.jsonl", metrics)
        self.agent.update(experience)
        return EpisodeResult(experience=experience, metrics=metrics)

    def _episode_context(self, case: dict[str, Any], seed: int | None, context: dict[str, Any] | None) -> dict[str, Any]:
        payload = self.run_context.as_dict()
        payload.update(context or {})
        payload["seed"] = seed
        payload["difficulty"] = case.get("difficulty")
        payload["case_title"] = case.get("title")
        payload["oracle_solvable"] = case.get("oracle", {}).get("solvable")
        return payload

    def _last_llm_trace(self) -> dict:
        llm = getattr(self.agent, "llm", None)
        trace = getattr(llm, "last_trace", None)
        return dict(trace) if isinstance(trace, dict) else {}
    def _usage_snapshot(self) -> dict:
        llm = getattr(self.agent, "llm", None)
        snapshot = getattr(llm, "usage_snapshot", None)
        if callable(snapshot):
            return snapshot()
        return {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_prompt_tokens": 0,
            "estimated_completion_tokens": 0,
            "token_source": "none",
        }

    def _stable_hash(self, payload: Any) -> str:
        text = json.dumps(to_jsonable(payload), ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


