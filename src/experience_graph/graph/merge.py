from __future__ import annotations

from dataclasses import dataclass

from experience_graph.core.conditions import canonical_signature, has_conflict
from experience_graph.core.models import GraphNode


@dataclass
class MergeDecision:
    decision: str
    method: str
    target_node_id: str | None
    rationale: str


class NodeMerger:
    def decide(self, new_node: GraphNode, existing_nodes: list[GraphNode]) -> MergeDecision:
        new_sig = canonical_signature(new_node.required)
        for node in existing_nodes:
            if canonical_signature(node.required) == new_sig:
                return MergeDecision("merged", "rule_exact_match", node.id, "same canonical required conditions")
        for node in existing_nodes:
            if has_conflict(new_node.required, node.required):
                return MergeDecision("created", "rule_conflict", None, "conditions conflict")
        return MergeDecision("created", "no_match", None, "no equivalent node found")
