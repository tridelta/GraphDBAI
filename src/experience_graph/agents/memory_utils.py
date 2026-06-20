from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experience_graph.core.models import Action, ExperienceRecord, Observation, get_path, pickaxe_rank
from experience_graph.core.serialization import to_jsonable


FEATURE_KEYS = [
    "inventory.diamond",
    "inventory.emerald",
    "inventory.wood",
    "inventory.crafting_table",
    "inventory.pickaxe",
    "environment.nearby_mine",
    "environment.nearby_village",
    "environment.village_has_armorer",
    "environment.mine_search_available",
    "environment.village_search_available",
    "environment.mine_diamond_capacity",
    "location",
]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def observation_features(observation: Observation) -> set[str]:
    state = observation.state
    features: set[str] = set()
    for key in FEATURE_KEYS:
        value = get_path(state, key, None)
        if value is None:
            continue
        if key == "inventory.diamond":
            features.add(f"{key}:>=24" if value >= 24 else f"{key}:>0" if value > 0 else f"{key}:0")
        elif key == "inventory.emerald":
            features.add(f"{key}:>=40" if value >= 40 else f"{key}:>=10" if value >= 10 else f"{key}:0")
        elif key == "inventory.pickaxe":
            features.add(f"tool.pickaxe_level:{pickaxe_rank(value)}")
        else:
            features.add(f"{key}:{value}")
    return features


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def action_labels(actions: list[Action] | list[str]) -> list[str]:
    labels = []
    for action in actions:
        labels.append(action.label() if isinstance(action, Action) else str(action))
    return labels


def summarize_experience(record: ExperienceRecord) -> str:
    actions = " -> ".join(record.executed_path) or "no action"
    if record.success:
        return f"Episode {record.episode_id} succeeded for {record.task_id} in {len(record.trajectory)} steps: {actions}."
    reason = record.failure_reason or "unknown_failure"
    return f"Episode {record.episode_id} failed for {record.task_id} because {reason}; attempted path: {actions}."
