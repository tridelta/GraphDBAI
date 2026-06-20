from __future__ import annotations

from dataclasses import dataclass

from experience_graph.core.models import Action, Condition, get_path


@dataclass
class PathRelevance:
    route_type: str
    matched_conditions: list[Condition]
    blocking_conditions: list[Condition]
    suggested_probe_actions: list[Action]
    state_similarity: float
    success_score: float
    efficiency_score: float
    info_value: float
    retrieval_score: float
    retrieval_reason: str
    common_failures: list[str]


def classify_route(actions: list[Action]) -> str:
    names = {action.name for action in actions}
    if "trade" in names and "mine" in names:
        return "mixed"
    if "trade" in names:
        return "trading"
    if "mine" in names:
        return "mining"
    if "craft" in names:
        return "crafting"
    if "inspect" in names or "explore" in names:
        return "exploration"
    return "unknown"


def suggested_probe_actions(missing: list[Condition], state: dict) -> list[Action]:
    probes: list[Action] = []
    seen: set[str] = set()
    for condition in missing:
        action: Action | None = None
        if condition.key == "environment.village_has_armorer" and get_path(state, "environment.nearby_village", False) is True:
            action = Action("inspect", {"target": "village"})
        elif condition.key == "environment.nearby_village" and get_path(state, "environment.village_search_available", False) is True:
            action = Action("explore", {"target": "village"})
        elif condition.key == "environment.nearby_mine" and get_path(state, "environment.mine_search_available", False) is True:
            action = Action("explore", {"target": "mine"})
        if action and action.label() not in seen:
            probes.append(action)
            seen.add(action.label())
    return probes


def condition_overlap(conditions: list[Condition], state: dict) -> tuple[list[Condition], list[Condition], float]:
    if not conditions:
        return [], [], 0.0
    matched = [condition for condition in conditions if condition.matches_state(state)]
    missing = [condition for condition in conditions if not condition.matches_state(state)]
    return matched, missing, len(matched) / len(conditions)


def smoothed_success(successes: int, attempts: int) -> float:
    return (successes + 1) / (attempts + 2) if attempts else 0.5


def efficiency_score(avg_steps: float | None) -> float:
    if avg_steps is None or avg_steps <= 0:
        return 0.5
    return max(0.0, min(1.0, 1.0 / avg_steps))


def top_failure_reasons(failures: dict[str, int], limit: int = 3) -> list[str]:
    return [reason for reason, _ in sorted(failures.items(), key=lambda item: item[1], reverse=True)[:limit]]


def build_retrieval_reason(route_type: str, matched: list[Condition], missing: list[Condition], common_failures: list[str]) -> str:
    parts = [f"Route type: {route_type}."]
    if matched:
        parts.append("Matched current state: " + ", ".join(condition.signature() for condition in matched[:4]) + ".")
    if missing:
        parts.append("Missing or unknown conditions: " + ", ".join(condition.signature() for condition in missing[:4]) + ".")
    if common_failures:
        parts.append("Common historical failures: " + ", ".join(common_failures) + ".")
    return " ".join(parts)


def score_path(
    actions: list[Action],
    hard_preconditions: list[Condition],
    state: dict,
    successes: int,
    attempts: int,
    avg_steps: float | None,
    applicability: str,
    failure_counts: dict[str, int],
) -> PathRelevance:
    route_type = classify_route(actions)
    matched, missing, similarity = condition_overlap(hard_preconditions, state)
    success = smoothed_success(successes, attempts)
    efficiency = efficiency_score(avg_steps)
    probes = suggested_probe_actions(missing, state)
    info = 0.7 if applicability == "needs_info" and probes else 0.3 if applicability == "needs_info" else 0.0
    applicability_adjustment = {"available": 0.15, "needs_info": 0.03, "blocked": -0.35}.get(applicability, 0.0)
    retrieval_score = (0.45 * similarity) + (0.35 * success) + (0.10 * info) + (0.10 * efficiency) + applicability_adjustment
    common_failures = top_failure_reasons(failure_counts)
    return PathRelevance(
        route_type=route_type,
        matched_conditions=matched,
        blocking_conditions=missing,
        suggested_probe_actions=probes,
        state_similarity=round(similarity, 4),
        success_score=round(success, 4),
        efficiency_score=round(efficiency, 4),
        info_value=round(info, 4),
        retrieval_score=round(retrieval_score, 4),
        retrieval_reason=build_retrieval_reason(route_type, matched, missing, common_failures),
        common_failures=common_failures,
    )

