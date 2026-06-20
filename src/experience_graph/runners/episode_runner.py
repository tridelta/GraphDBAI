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
        continue_after_env_failure: bool = True,
    ):
        self.env = env
        self.agent = agent
        self.organizer = organizer
        self.retriever = retriever
        self.logger = logger
        self.max_steps = max_steps
        self.builder = ExperienceBuilder()
        self.run_context = run_context or RunContext()
        self.continue_after_env_failure = continue_after_env_failure

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
        action_history: list[str] = []
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
            prompt_diagnostics = self._prompt_diagnostics(llm_trace)
            if output.plan is not None:
                proposed_plan = output.plan
            if output.action is None:
                oracle_failure_reason = case.get("oracle", {}).get("failure_reason")
                failure_reason = output.rationale or "no_action"
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
                        "episode_done": True,
                        "episode_success": False,
                        "episode_terminal_reason": failure_reason,
                        "failure_reason": failure_reason,
                        "oracle_failure_reason": oracle_failure_reason,
                        "repeated_action_count": 0,
                        "repeated_action": False,
                        "prompt_diagnostics": prompt_diagnostics,
                        "llm_usage_delta": llm_step_usage,
                        "llm_model": llm_trace.get("model"),
                        "llm_input_messages": llm_trace.get("messages", []),
                        "llm_output": llm_trace.get("response"),
                        "llm_raw_response": llm_trace.get("raw_response"),
                        "llm_parse_error": llm_trace.get("parse_error"),
                        "llm_finish_reason": llm_trace.get("finish_reason"),
                        "llm_retry_count": llm_trace.get("retry_count", 0),
                        "llm_max_tokens": llm_trace.get("max_tokens"),
                        "llm_attempts": self._trace_attempt_summary(llm_trace),
                    },
                )
                break
            action_label = output.action.label()
            repeated_action_count = self._consecutive_action_count(action_history, action_label) + 1
            action_history.append(action_label)
            before = observation
            result = self.env.step(output.action)
            episode_terminal = result.done or step_index >= self.max_steps - 1 or (
                not result.ok and not self._should_continue_after_failure(result.failure_reason, step_index)
            )
            episode_terminal_reason = None
            if episode_terminal and not result.done:
                episode_terminal_reason = result.failure_reason or "step_budget_exhausted"
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
                    "episode_done": episode_terminal,
                    "episode_success": bool(result.done),
                    "episode_terminal_reason": episode_terminal_reason,
                    "reward": result.reward,
                    "cost": result.cost,
                    "failure_reason": result.failure_reason,
                    "oracle_failure_reason": case.get("oracle", {}).get("failure_reason"),
                    "repeated_action_count": repeated_action_count,
                    "repeated_action": repeated_action_count > 1,
                    "prompt_diagnostics": prompt_diagnostics,
                    "revealed_conditions": result.revealed_conditions,
                    "state_delta": result.state_delta,
                    "llm_usage_delta": llm_step_usage,
                    "llm_model": llm_trace.get("model"),
                    "llm_input_messages": llm_trace.get("messages", []),
                    "llm_output": llm_trace.get("response"),
                    "llm_raw_response": llm_trace.get("raw_response"),
                    "llm_parse_error": llm_trace.get("parse_error"),
                    "llm_finish_reason": llm_trace.get("finish_reason"),
                    "llm_retry_count": llm_trace.get("retry_count", 0),
                    "llm_max_tokens": llm_trace.get("max_tokens"),
                    "llm_attempts": self._trace_attempt_summary(llm_trace),
                },
            )
            if result.done:
                success = True
                failure_reason = None
                break
            if not result.ok:
                failure_reason = result.failure_reason
                if not episode_terminal:
                    continue
                break
        if not success and failure_reason is None:
            failure_reason = "step_budget_exhausted"

        experience = self.builder.build(episode_id, task.id, initial, trajectory, success, proposed_plan=proposed_plan)
        if success:
            experience.failure_reason = None
            experience.metrics["failure_reason"] = None
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
            "oracle_failure_reason": case.get("oracle", {}).get("failure_reason"),
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
        episode_row = to_jsonable(experience)
        episode_row.update(metrics)
        episode_row["metrics"] = metrics
        self.logger.write_jsonl("episodes.jsonl", episode_row)
        self.logger.write_jsonl("metrics.jsonl", metrics)
        self.agent.update(experience)
        return EpisodeResult(experience=experience, metrics=metrics)

    def _should_continue_after_failure(self, failure_reason: str | None, step_index: int) -> bool:
        if not self.continue_after_env_failure:
            return False
        if step_index >= self.max_steps - 1:
            return False
        terminal_reasons = {
            "unknown_action",
            "premature_impossible_report",
            "incorrect_impossible_report",
        }
        return failure_reason not in terminal_reasons

    def _consecutive_action_count(self, action_history: list[str], action_label: str) -> int:
        count = 0
        for previous in reversed(action_history):
            if previous != action_label:
                break
            count += 1
        return count

    def _prompt_diagnostics(self, llm_trace: dict) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "prompt_hidden_facts": False,
            "prompt_ambiguous_keys": [],
            "prompt_candidate_paths": 0,
            "prompt_retrieved_skills": 0,
            "prompt_retrieved_trajectories": 0,
            "prompt_has_experience_view_ref": False,
            "prompt_parse_errors": 0,
        }
        ambiguous_keys: set[str] = set()
        for message in llm_trace.get("messages", []):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if "hidden_facts" in content:
                diagnostics["prompt_hidden_facts"] = True
            try:
                payload = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                diagnostics["prompt_parse_errors"] += 1
                continue
            observation = payload.get("observation", {})
            state = observation.get("state", {}) if isinstance(observation, dict) else {}
            if isinstance(state, dict) and "hidden_facts" in state:
                diagnostics["prompt_hidden_facts"] = True
            ambiguous = state.get("ambiguous") if isinstance(state, dict) else None
            if isinstance(ambiguous, dict):
                ambiguous_keys.update(str(key) for key in ambiguous)
            candidate_paths = payload.get("candidate_paths")
            if isinstance(candidate_paths, list):
                diagnostics["prompt_candidate_paths"] += len(candidate_paths)
            retrieved_skills = payload.get("retrieved_skills")
            if isinstance(retrieved_skills, list):
                diagnostics["prompt_retrieved_skills"] += len(retrieved_skills)
            retrieved_trajectories = payload.get("retrieved_trajectories")
            if isinstance(retrieved_trajectories, list):
                diagnostics["prompt_retrieved_trajectories"] += len(retrieved_trajectories)
            diagnostics["prompt_has_experience_view_ref"] = diagnostics["prompt_has_experience_view_ref"] or "experience_view" in payload
        diagnostics["prompt_ambiguous_keys"] = sorted(ambiguous_keys)
        return diagnostics


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

    def _trace_attempt_summary(self, llm_trace: dict) -> list[dict[str, Any]]:
        attempts = llm_trace.get("attempts")
        if not isinstance(attempts, list):
            return []
        summary = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            raw = attempt.get("raw_response") or ""
            summary.append(
                {
                    "attempt_index": attempt.get("attempt_index"),
                    "finish_reason": attempt.get("finish_reason"),
                    "parse_error": attempt.get("parse_error"),
                    "max_tokens": attempt.get("max_tokens"),
                    "valid": attempt.get("valid"),
                    "raw_response_length": len(raw),
                }
            )
        return summary

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

