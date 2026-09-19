from __future__ import annotations

from typing import Any

from experience_graph.core.models import Condition, GraphNode, Observation, TaskSpec, visible_state


IMPORTANT_STATE_KEYS = {
    "task",
    "location",
    "inventory",
    "tools",
    "environment",
    "ambiguous",
}


def observation_text(task: TaskSpec, observation: Observation, current_conditions: list[Condition]) -> str:
    state = visible_state(observation.state)
    parts = [f"task {task.id}", f"case {observation.case_id or 'unknown'}"]
    parts.extend(_flatten_state(state))
    if current_conditions:
        parts.append("conditions " + " ".join(condition.signature() for condition in current_conditions[:12]))
    return " | ".join(parts)


def node_text(node: GraphNode) -> str:
    parts = [node.node_type, node.label, node.id]
    if node.required:
        parts.append("required " + " ".join(condition.signature() for condition in node.required))
    if node.suggested:
        parts.append("suggested " + " ".join(node.suggested[:8]))
    tags = _metadata_terms(node.metadata)
    if tags:
        parts.append("metadata " + " ".join(tags))
    return " | ".join(part for part in parts if part)


def _flatten_state(state: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    for key in sorted(IMPORTANT_STATE_KEYS):
        if key not in state:
            continue
        value = state[key]
        if isinstance(value, dict):
            parts.extend(_flatten_mapping(key, value))
        else:
            parts.append(f"{key}={value}")
    return parts


def _flatten_mapping(prefix: str, value: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    for key, item in sorted(value.items()):
        dotted = f"{prefix}.{key}"
        if isinstance(item, dict):
            parts.extend(_flatten_mapping(dotted, item))
        elif isinstance(item, list):
            parts.append(f"{dotted}=" + ",".join(str(v) for v in item))
        else:
            parts.append(f"{dotted}={item}")
    return parts


def _metadata_terms(metadata: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for key in ("task_id", "action_tags", "resource_tags", "route_type"):
        value = metadata.get(key)
        if isinstance(value, list):
            terms.extend(str(item) for item in value)
        elif value is not None:
            terms.append(str(value))
    return terms
