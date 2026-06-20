from __future__ import annotations

import hashlib

from experience_graph.core.conditions import missing_conditions
from experience_graph.core.models import CandidatePathView, Condition, ExperienceView, Observation, TaskSpec
from experience_graph.graph.store import JsonGraphStore


class GraphRetriever:
    def __init__(self, store: JsonGraphStore, token_budget: int = 1000, top_k: int = 5):
        self.store = store
        self.token_budget = token_budget
        self.top_k = top_k

    def retrieve(self, task: TaskSpec, observation: Observation, current_conditions: list[Condition]) -> ExperienceView:
        state = observation.state
        candidates: list[CandidatePathView] = []
        for path in self.store.paths.values():
            if path.task_id != task.id:
                continue
            actions = []
            missing = []
            blocked = False
            needs_info = False
            for edge_id in path.edge_ids:
                edge = self.store.edges.get(edge_id)
                if edge is None or edge.status != "active":
                    blocked = True
                    continue
                actions.append(edge.action_template)
                unknown_requirements = [condition for condition in edge.hard_preconditions if condition.operator == "unknown"]
                edge_missing = missing_conditions(edge.hard_preconditions, state)
                missing.extend(edge_missing)
                if unknown_requirements:
                    needs_info = True
                elif edge_missing:
                    blocked = True
            applicability = "blocked" if blocked else "needs_info" if needs_info else "available"
            attempts = path.stats.attempts
            success_rate = (path.stats.successes + 1) / (attempts + 2) if attempts else None
            candidates.append(
                CandidatePathView(
                    path_id=path.id,
                    label=" -> ".join(action.label() for action in actions) or path.id,
                    applicability=applicability,
                    missing_conditions=missing,
                    actions_preview=actions[:5],
                    success_rate=success_rate,
                    attempts=attempts,
                    avg_steps=path.stats.avg_steps,
                    evidence=f"{path.stats.successes}/{path.stats.attempts} successes",
                )
            )
        candidates.sort(key=lambda c: (c.applicability != "available", -(c.success_rate or 0), c.avg_steps or 999))
        candidates = candidates[: self.top_k]
        view = ExperienceView(
            view_id="view_" + hashlib.sha1(f"{task.id}:{observation.case_id}:{observation.step_count}".encode()).hexdigest()[:12],
            relevant_conditions=current_conditions,
            candidate_paths=candidates,
            token_estimate=self._estimate_tokens(candidates),
        )
        if view.token_estimate > self.token_budget:
            view.summaries.append(f"Experience view truncated to top {self.top_k} paths.")
            view.token_estimate = self.token_budget
        return view

    def _estimate_tokens(self, candidates: list[CandidatePathView]) -> int:
        text = " ".join(candidate.label + " " + candidate.evidence for candidate in candidates)
        return max(1, len(text) // 4)


