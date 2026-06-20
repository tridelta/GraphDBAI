from __future__ import annotations

import hashlib
from dataclasses import dataclass

from experience_graph.core.conditions import canonical_signature
from experience_graph.core.models import (
    Action,
    Condition,
    EdgeStats,
    ExperienceRecord,
    GraphEdge,
    GraphNode,
    NodeStats,
    PathRecord,
    PathStats,
    get_path,
)
from experience_graph.graph.merge import NodeMerger
from experience_graph.graph.store import JsonGraphStore


FAILURE_PRECONDITIONS = {
    "missing_required_pickaxe": Condition("tool.pickaxe_level", ">=", 3, source="rule"),
    "mine_not_discovered": Condition("environment.nearby_mine", "==", True, source="rule"),
    "village_not_discovered": Condition("environment.nearby_village", "==", True, source="rule"),
    "armorer_unknown": Condition("environment.village_has_armorer", "unknown", True, source="rule"),
    "no_armorer": Condition("environment.village_has_armorer", "==", True, source="rule"),
    "not_enough_emeralds": Condition("inventory.emerald", ">=", 10, source="rule"),
    "insufficient_diamonds": Condition("inventory.diamond", ">=", 24, source="rule"),
    "missing_crafting_table": Condition("inventory.crafting_table", "==", True, source="rule"),
}


@dataclass
class GraphUpdateSummary:
    added_nodes: int = 0
    added_edges: int = 0
    updated_edges: int = 0
    added_paths: int = 0


class GraphOrganizer:
    def __init__(
        self,
        store: JsonGraphStore,
        min_attempts_for_dormant: int = 5,
        dormant_success_threshold: float = 0.1,
        enable_node_merging: bool = True,
        learn_failure_preconditions: bool = True,
    ):
        self.store = store
        self.merger = NodeMerger()
        self.min_attempts_for_dormant = min_attempts_for_dormant
        self.dormant_success_threshold = dormant_success_threshold
        self.enable_node_merging = enable_node_merging
        self.learn_failure_preconditions = learn_failure_preconditions

    def integrate(self, record: ExperienceRecord) -> GraphUpdateSummary:
        summary = GraphUpdateSummary()
        start_node = self._upsert_node(self._node_from_conditions("start", record.initial_observation.case_id or "start", []), record, summary)
        previous_node = start_node
        edge_ids: list[str] = []

        for step in record.trajectory:
            target_conditions = list(step.result.revealed_conditions)
            if self.learn_failure_preconditions and step.result.failure_reason and step.result.failure_reason in FAILURE_PRECONDITIONS:
                target_conditions.append(FAILURE_PRECONDITIONS[step.result.failure_reason])
            if not target_conditions and step.result.ok:
                target_conditions = [Condition("action.result", "==", step.action.label(), source="env")]
            target_node = self._upsert_node(
                self._node_from_conditions("checkpoint", self._label_for_step(step.action, step.result.failure_reason), target_conditions),
                record,
                summary,
            )
            edge_preconditions = self._action_preconditions(step.action, step.observation_before.state, step.result.ok, step.result.failure_reason)
            edge = self._upsert_edge(previous_node.id, target_node.id, step.action, target_conditions, step.result.ok, step.result.failure_reason, summary, edge_preconditions)
            edge_ids.append(edge.id)
            previous_node = target_node

        if record.success:
            goal_node = self._upsert_node(
                self._node_from_conditions("goal", f"{record.task_id}_success", [Condition("task.success", "==", record.task_id, source="env")]),
                record,
                summary,
            )
            previous_node = goal_node

        if edge_ids:
            path_id = "path_" + self._hash([record.task_id, *edge_ids])
            path = self.store.paths.get(path_id)
            if path is None:
                path = PathRecord(
                    id=path_id,
                    task_id=record.task_id,
                    start_signature=record.initial_observation.case_id or "unknown",
                    goal_node=previous_node.id,
                    edge_ids=edge_ids,
                    stats=PathStats(),
                    metadata=self._path_metadata(record),
                )
                summary.added_paths += 1
            path.stats.attempts += 1
            if record.success:
                path.stats.successes += 1
            path.stats.avg_steps = self._running_average(path.stats.avg_steps, len(record.trajectory), path.stats.attempts)
            path.stats.avg_cost = path.stats.avg_steps
            self.store.upsert_path(path)

        self.store.flush()
        return summary

    def _upsert_node(self, node: GraphNode, record: ExperienceRecord, summary: GraphUpdateSummary) -> GraphNode:
        if self.enable_node_merging:
            decision = self.merger.decide(node, list(self.store.nodes.values()))
            if decision.decision == "merged" and decision.target_node_id:
                existing = self.store.nodes[decision.target_node_id]
                existing.stats.attempts += 1
                if record.success:
                    existing.stats.successes += 1
                self.store.append_merge_decision(
                    {
                        "new_node": node.id,
                        "target_node": existing.id,
                        "decision": "merged",
                        "method": decision.method,
                        "rationale": decision.rationale,
                        "episode_id": record.episode_id,
                    }
                )
                return existing
        if not self.enable_node_merging:
            node.id = f"{node.id}_{len(self.store.nodes):06d}"
        node.stats.attempts = 1
        node.stats.successes = 1 if record.success else 0
        self.store.upsert_node(node)
        summary.added_nodes += 1
        if not self.enable_node_merging:
            self.store.append_merge_decision(
                {
                    "new_node": node.id,
                    "target_node": None,
                    "decision": "created",
                    "method": "disabled",
                    "rationale": "node merging disabled for experiment variant",
                    "episode_id": record.episode_id,
                }
            )
        return node

    def _upsert_edge(
        self,
        from_node: str,
        to_node: str,
        action: Action,
        effects: list[Condition],
        ok: bool,
        failure_reason: str | None,
        summary: GraphUpdateSummary,
        inferred_preconditions: list[Condition] | None = None,
    ) -> GraphEdge:
        preconditions = list(inferred_preconditions or [])
        if self.learn_failure_preconditions and failure_reason and failure_reason in FAILURE_PRECONDITIONS:
            preconditions.append(FAILURE_PRECONDITIONS[failure_reason])
        edge_id = "edge_" + self._hash([from_node, to_node, action.label()])
        edge = self.store.edges.get(edge_id)
        if edge is None:
            edge = GraphEdge(
                id=edge_id,
                from_node=from_node,
                to_node=to_node,
                action_template=action,
                hard_preconditions=preconditions,
                effects=effects,
                stats=EdgeStats(),
            )
            summary.added_edges += 1
        else:
            summary.updated_edges += 1
            for condition in preconditions:
                if condition.signature() not in {c.signature() for c in edge.hard_preconditions}:
                    edge.hard_preconditions.append(condition)
        edge.stats.attempts += 1
        if ok:
            edge.stats.successes += 1
        if failure_reason:
            edge.stats.failure_reasons[failure_reason] = edge.stats.failure_reasons.get(failure_reason, 0) + 1
        edge.stats.avg_cost = self._running_average(edge.stats.avg_cost, 1.0, edge.stats.attempts)
        if edge.stats.attempts >= self.min_attempts_for_dormant and edge.stats.successes / edge.stats.attempts < self.dormant_success_threshold:
            edge.status = "dormant"
        self.store.upsert_edge(edge)
        return edge


    def _path_metadata(self, record: ExperienceRecord) -> dict:
        actions = [step.action for step in record.trajectory]
        return {
            "task_id": record.task_id,
            "case_id": record.initial_observation.case_id,
            "action_tags": sorted({action.name for action in actions}),
            "resource_tags": self._resource_tags(actions),
            "domain_tags": ["textcraft_mc"],
        }

    def _resource_tags(self, actions: list[Action]) -> list[str]:
        tags: set[str] = set()
        for action in actions:
            for key in ["item", "resource", "want", "target", "location", "villager"]:
                value = action.args.get(key)
                if value:
                    tags.add(str(value))
        return sorted(tags)

    def _action_preconditions(self, action: Action, state: dict, ok: bool, failure_reason: str | None) -> list[Condition]:
        if not ok:
            return []
        conditions: list[Condition] = []
        if action.name == "craft":
            item = action.args.get("item")
            if item == "crafting_table":
                conditions.append(Condition("inventory.wood", ">=", 4, source="rule"))
            elif item:
                if item == "diamond_set":
                    conditions.append(Condition("inventory.diamond", ">=", 24, source="rule"))
                else:
                    diamond_costs = {"diamond_helmet": 5, "diamond_chestplate": 8, "diamond_leggings": 7, "diamond_boots": 4}
                    if item in diamond_costs:
                        conditions.append(Condition("inventory.diamond", ">=", diamond_costs[item], source="rule"))
                conditions.append(Condition("inventory.crafting_table", "==", True, source="rule"))
        elif action.name == "mine" and action.args.get("resource") == "diamond":
            conditions.append(Condition("environment.nearby_mine", "==", True, source="rule"))
            conditions.append(Condition("tool.pickaxe_level", ">=", 3, source="rule"))
        elif action.name == "move_to":
            location = action.args.get("location")
            if location == "mine":
                conditions.append(Condition("environment.nearby_mine", "==", True, source="rule"))
            elif location == "village":
                conditions.append(Condition("environment.nearby_village", "==", True, source="rule"))
        elif action.name == "inspect":
            target = action.args.get("target")
            if target == "village":
                conditions.append(Condition("environment.nearby_village", "==", True, source="rule"))
            elif target == "mine":
                conditions.append(Condition("environment.nearby_mine", "==", True, source="rule"))
        elif action.name == "explore":
            target = action.args.get("target")
            if target == "village":
                conditions.append(Condition("environment.village_search_available", "==", True, source="rule"))
            elif target == "mine":
                conditions.append(Condition("environment.mine_search_available", "==", True, source="rule"))
        elif action.name == "trade":
            want = action.args.get("want")
            conditions.append(Condition("location", "==", "village", source="rule"))
            conditions.append(Condition("environment.village_has_armorer", "==", True, source="rule"))
            if want == "diamond_set":
                conditions.append(Condition("inventory.emerald", ">=", 40, source="rule"))
            else:
                conditions.append(Condition("inventory.emerald", ">=", 10, source="rule"))
        elif action.name == "gather" and action.args.get("resource") == "wood":
            biome = get_path(state, "environment.biome", None)
            if biome is not None:
                conditions.append(Condition("environment.biome", "in", ["forest", "plains"], source="rule"))
        return self._dedupe_conditions(conditions)

    def _dedupe_conditions(self, conditions: list[Condition]) -> list[Condition]:
        deduped: list[Condition] = []
        seen: set[str] = set()
        for condition in conditions:
            signature = condition.signature()
            if signature not in seen:
                deduped.append(condition)
                seen.add(signature)
        return deduped
    def _node_from_conditions(self, node_type: str, label: str, conditions: list[Condition]) -> GraphNode:
        node_id = "node_" + self._hash([node_type, label, canonical_signature(conditions)])
        return GraphNode(id=node_id, label=label, node_type=node_type, required=conditions, stats=NodeStats())

    def _label_for_step(self, action: Action, failure_reason: str | None) -> str:
        if failure_reason:
            return failure_reason
        return f"after_{action.label()}"

    def _hash(self, values: list[str]) -> str:
        return hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()[:12]

    def _running_average(self, old: float | None, new: float, count: int) -> float:
        if old is None or count <= 1:
            return new
        return ((old * (count - 1)) + new) / count




