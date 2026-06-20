from __future__ import annotations

from .models import Condition


def canonical_signature(conditions: list[Condition]) -> str:
    return "|".join(sorted(condition.canonical().signature() for condition in conditions))


def has_conflict(left: list[Condition], right: list[Condition]) -> bool:
    return any(a.conflicts_with(b) for a in left for b in right)


def conditions_match_state(conditions: list[Condition], state: dict) -> bool:
    return all(condition.matches_state(state) for condition in conditions)


def missing_conditions(conditions: list[Condition], state: dict) -> list[Condition]:
    return [condition for condition in conditions if not condition.matches_state(state)]
