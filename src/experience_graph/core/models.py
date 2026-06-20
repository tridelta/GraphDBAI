from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


UNKNOWN = "unknown"


@dataclass(frozen=True)
class Condition:
    key: str
    operator: str
    value: Any
    source: str = "env"
    confidence: float = 1.0

    def canonical(self) -> "Condition":
        key = self.key
        operator = self.operator
        value = self.value

        if key == "inventory.crafting_table":
            key = "inventory.crafting_table"
        if key in {"has_enough_diamond", "has_enough_diamonds"}:
            key = "inventory.diamond"
            operator = ">="
            value = 24
        if key == "tool.pickaxe_level" and isinstance(value, str):
            value = pickaxe_rank(value)
        if operator == "unknown":
            value = UNKNOWN
        return Condition(key=key, operator=operator, value=value, source=self.source, confidence=self.confidence)

    def signature(self) -> str:
        c = self.canonical()
        return f"{c.key}:{c.operator}:{stable_value(c.value)}"

    def conflicts_with(self, other: "Condition") -> bool:
        a = self.canonical()
        b = other.canonical()
        if a.key != b.key:
            return False
        if a.operator == "==" and b.operator == "==" and a.value != b.value:
            return True
        if a.operator == "unknown" or b.operator == "unknown":
            return False
        return False

    def matches_state(self, state: dict[str, Any]) -> bool:
        actual = get_path(state, self.key, default=UNKNOWN)
        return compare_value(actual, self.operator, self.value)


@dataclass
class Observation:
    state: dict[str, Any]
    case_id: str | None = None
    step_count: int = 0

    def get(self, dotted_key: str, default: Any = UNKNOWN) -> Any:
        return get_path(self.state, dotted_key, default=default)

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "case_id": self.case_id, "step_count": self.step_count}


@dataclass(frozen=True)
class Action:
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str) -> "Action":
        text = text.strip()
        if "(" not in text or not text.endswith(")"):
            return cls(text, {})
        name, raw_args = text[:-1].split("(", 1)
        args: dict[str, Any] = {}
        parts = [part.strip() for part in raw_args.split(",") if part.strip()]
        if name == "craft" and parts:
            args["item"] = parts[0]
        elif name == "gather" and parts:
            args["resource"] = parts[0]
        elif name == "mine" and parts:
            args["resource"] = parts[0]
        elif name == "move_to" and parts:
            args["location"] = parts[0]
        elif name in {"inspect", "explore"} and parts:
            args["target"] = parts[0]
        elif name == "trade":
            if len(parts) >= 1:
                args["villager"] = parts[0]
            if len(parts) >= 2:
                args["offer"] = parts[1]
            if len(parts) >= 3:
                args["want"] = parts[2]
        else:
            for idx, part in enumerate(parts):
                args[f"arg{idx}"] = part
        return cls(name=name, args=args)

    def label(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(str(v) for v in self.args.values())})"


@dataclass
class ActionSpec:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    id: str
    success_conditions: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    ok: bool
    action: Action
    observation: Observation
    reward: float = 0.0
    cost: float = 1.0
    done: bool = False
    failure_reason: str | None = None
    revealed_conditions: list[Condition] = field(default_factory=list)
    state_delta: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepRecord:
    step_index: int
    observation_before: Observation
    experience_view_id: str | None
    agent_thought: str | None
    action: Action
    result: StepResult
    observation_after: Observation


@dataclass
class ExperienceRecord:
    episode_id: str
    task_id: str
    initial_observation: Observation
    final_observation: Observation
    trajectory: list[StepRecord]
    success: bool
    failure_reason: str | None
    discovered_conditions: list[Condition]
    proposed_plan: list[Action] | None
    executed_path: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeStats:
    attempts: int = 0
    successes: int = 0
    avg_steps_to_goal: float | None = None


@dataclass
class EdgeStats:
    attempts: int = 0
    successes: int = 0
    avg_cost: float | None = None
    failure_reasons: dict[str, int] = field(default_factory=dict)


@dataclass
class PathStats:
    attempts: int = 0
    successes: int = 0
    avg_steps: float | None = None
    avg_cost: float | None = None


@dataclass
class GraphNode:
    id: str
    label: str
    node_type: Literal["start", "checkpoint", "observation", "goal"]
    required: list[Condition]
    suggested: list[str] = field(default_factory=list)
    stats: NodeStats = field(default_factory=NodeStats)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    id: str
    from_node: str
    to_node: str
    action_template: Action
    hard_preconditions: list[Condition] = field(default_factory=list)
    soft_preconditions: list[str] = field(default_factory=list)
    effects: list[Condition] = field(default_factory=list)
    stats: EdgeStats = field(default_factory=EdgeStats)
    status: Literal["active", "dormant", "archived"] = "active"


@dataclass
class PathRecord:
    id: str
    task_id: str
    start_signature: str
    goal_node: str
    edge_ids: list[str]
    stats: PathStats = field(default_factory=PathStats)
    last_used_episode: int = 0


@dataclass
class CandidatePathView:
    path_id: str
    label: str
    applicability: Literal["available", "needs_info", "blocked"]
    missing_conditions: list[Condition] = field(default_factory=list)
    actions_preview: list[Action] = field(default_factory=list)
    success_rate: float | None = None
    attempts: int = 0
    avg_steps: float | None = None
    evidence: str = ""
    route_type: str = "unknown"
    matched_conditions: list[Condition] = field(default_factory=list)
    suggested_probe_actions: list[Action] = field(default_factory=list)
    state_similarity: float = 0.0
    retrieval_score: float = 0.0
    retrieval_reason: str = ""
    common_failures: list[str] = field(default_factory=list)


@dataclass
class ExperienceView:
    view_id: str
    relevant_conditions: list[Condition] = field(default_factory=list)
    candidate_paths: list[CandidatePathView] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    token_estimate: int = 0


@dataclass
class AgentInput:
    task: TaskSpec
    observation: Observation
    available_actions: list[ActionSpec]
    experience_view: ExperienceView
    step_budget_remaining: int


@dataclass
class AgentOutput:
    action: Action | None
    plan: list[Action] | None = None
    rationale: str | None = None
    confidence: float | None = None


class ExternalAgent(Protocol):
    def act(self, agent_input: AgentInput) -> AgentOutput:
        ...

    def update(self, feedback: ExperienceRecord) -> None:
        ...


def pickaxe_rank(value: Any) -> int:
    ranks = {"none": 0, None: 0, "wooden": 1, "stone": 2, "iron": 3, "diamond": 4, "netherite": 5}
    if isinstance(value, int):
        return value
    return ranks.get(value, 0)


def get_path(data: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_path(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def compare_value(actual: Any, operator: str, expected: Any) -> bool:
    if actual == UNKNOWN:
        return operator == "unknown"
    if operator == "unknown":
        return actual == UNKNOWN
    if operator == "==":
        return actual == expected
    if operator == "exists":
        return actual is not None and actual != UNKNOWN
    if operator == "in":
        return actual in expected
    if operator in {">=", ">", "<=", "<"}:
        try:
            actual_num = pickaxe_rank(actual) if isinstance(actual, str) else actual
            expected_num = pickaxe_rank(expected) if isinstance(expected, str) else expected
            if operator == ">=":
                return actual_num >= expected_num
            if operator == ">":
                return actual_num > expected_num
            if operator == "<=":
                return actual_num <= expected_num
            return actual_num < expected_num
        except TypeError:
            return False
    return False


def stable_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(map(str, value)) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{stable_value(v)}" for k, v in sorted(value.items())) + "}"
    return str(value)

