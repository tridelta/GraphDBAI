from __future__ import annotations

import hashlib
import random

from experience_graph.core.conditions import missing_conditions
from experience_graph.core.models import CandidatePathView, Condition, ExperienceView, Observation, TaskSpec, UNKNOWN, get_path
from experience_graph.graph.relevance import score_path
from experience_graph.graph.store import JsonGraphStore


INFO_CONDITION_KEYS = {
    "environment.village_has_armorer",
    "environment.nearby_mine",
    "environment.nearby_village",
    "environment.mine_depth",
}


class GraphRetriever:
    def __init__(
        self,
        store: JsonGraphStore,
        token_budget: int = 1000,
        top_k: int = 5,
        include_statistics: bool = True,
        ranking_mode: str = "score",
        random_seed: int = 0,
    ):
        self.store = store
        self.token_budget = token_budget
        self.top_k = top_k
        self.include_statistics = include_statistics
        self.ranking_mode = ranking_mode
        self.random = random.Random(random_seed)

    def retrieve(self, task: TaskSpec, observation: Observation, current_conditions: list[Condition]) -> ExperienceView:
        view_id = "view_" + hashlib.sha1(f"{task.id}:{observation.case_id}:{observation.step_count}".encode()).hexdigest()[:12]
        if self.top_k <= 0:
            return ExperienceView(view_id=view_id, relevant_conditions=current_conditions, token_estimate=0)

        state = observation.state
        candidates: list[CandidatePathView] = []
        for path in self.store.paths.values():
            if path.task_id != task.id:
                continue
            actions = []
            missing = []
            hard_preconditions = []
            failure_counts: dict[str, int] = {}
            blocked = False
            needs_info = False
            for edge_id in path.edge_ids:
                edge = self.store.edges.get(edge_id)
                if edge is None or edge.status != "active":
                    blocked = True
                    continue
                actions.append(edge.action_template)
                hard_preconditions.extend(edge.hard_preconditions)
                for reason, count in edge.stats.failure_reasons.items():
                    failure_counts[reason] = failure_counts.get(reason, 0) + count
                unknown_requirements = [condition for condition in edge.hard_preconditions if condition.operator == "unknown"]
                edge_missing = missing_conditions(edge.hard_preconditions, state)
                missing.extend(edge_missing)
                if unknown_requirements or any(self._missing_due_unknown(condition, state) for condition in edge_missing):
                    needs_info = True
                elif edge_missing:
                    blocked = True
            hard_preconditions = self._dedupe_conditions(hard_preconditions)
            missing = self._dedupe_conditions(missing)
            applicability = "blocked" if blocked else "needs_info" if needs_info else "available"
            attempts = path.stats.attempts
            success_rate = None
            evidence = "statistics hidden"
            successes_for_score = path.stats.successes if self.include_statistics else 0
            attempts_for_score = attempts if self.include_statistics else 0
            if self.include_statistics:
                success_rate = (path.stats.successes + 1) / (attempts + 2) if attempts else None
                evidence = f"{path.stats.successes}/{path.stats.attempts} successes"
            relevance = score_path(
                actions=actions,
                hard_preconditions=hard_preconditions,
                state=state,
                successes=successes_for_score,
                attempts=attempts_for_score,
                avg_steps=path.stats.avg_steps if self.include_statistics else None,
                applicability=applicability,
                failure_counts=failure_counts,
            )
            candidates.append(
                CandidatePathView(
                    path_id=path.id,
                    label=" -> ".join(action.label() for action in actions) or path.id,
                    applicability=applicability,
                    missing_conditions=missing,
                    actions_preview=actions[:5],
                    success_rate=success_rate,
                    attempts=attempts if self.include_statistics else 0,
                    avg_steps=path.stats.avg_steps if self.include_statistics else None,
                    evidence=evidence,
                    route_type=relevance.route_type,
                    matched_conditions=relevance.matched_conditions,
                    suggested_probe_actions=relevance.suggested_probe_actions,
                    state_similarity=relevance.state_similarity,
                    retrieval_score=relevance.retrieval_score,
                    retrieval_reason=relevance.retrieval_reason,
                    common_failures=relevance.common_failures,
                )
            )
        if self.ranking_mode == "random":
            self.random.shuffle(candidates)
        else:
            candidates.sort(key=lambda c: (c.applicability == "blocked", -c.retrieval_score, -(c.success_rate or 0), c.avg_steps or 999))
        candidates = candidates[: self.top_k]
        view = ExperienceView(
            view_id=view_id,
            relevant_conditions=current_conditions,
            candidate_paths=candidates,
            token_estimate=self._estimate_tokens(candidates),
        )
        if view.token_estimate > self.token_budget:
            view.summaries.append(f"Experience view truncated to top {self.top_k} paths.")
            view.token_estimate = self.token_budget
        return view

    def _estimate_tokens(self, candidates: list[CandidatePathView]) -> int:
        text = " ".join(candidate.label + " " + candidate.evidence + " " + candidate.retrieval_reason for candidate in candidates)
        return max(1, len(text) // 4) if text else 0

    def _missing_due_unknown(self, condition: Condition, state: dict) -> bool:
        actual = get_path(state, condition.key, default=UNKNOWN)
        return condition.key in INFO_CONDITION_KEYS and actual == UNKNOWN

    def _dedupe_conditions(self, conditions: list[Condition]) -> list[Condition]:
        deduped: list[Condition] = []
        seen: set[str] = set()
        for condition in conditions:
            signature = condition.signature()
            if signature not in seen:
                deduped.append(condition)
                seen.add(signature)
        return deduped
