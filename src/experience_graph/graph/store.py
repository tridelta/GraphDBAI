from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experience_graph.core.models import (
    Action,
    Condition,
    EdgeStats,
    GraphEdge,
    GraphNode,
    NodeStats,
    PathRecord,
    PathStats,
)
from experience_graph.core.serialization import action_from_dict, condition_from_dict, to_jsonable


class JsonGraphStore:
    def __init__(self, run_dir: str | Path, load_existing: bool = True):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}
        self.paths: dict[str, PathRecord] = {}
        self.merge_decisions: list[dict[str, Any]] = []
        if load_existing:
            self.load()

    def load(self) -> None:
        self.nodes = {node.id: node for node in self._read_nodes()}
        self.edges = {edge.id: edge for edge in self._read_edges()}
        self.paths = {path.id: path for path in self._read_paths()}
        self.merge_decisions = self._read_jsonl("merge_decisions.jsonl")

    def upsert_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node

    def upsert_edge(self, edge: GraphEdge) -> None:
        self.edges[edge.id] = edge

    def upsert_path(self, path: PathRecord) -> None:
        self.paths[path.id] = path

    def append_merge_decision(self, decision: dict[str, Any]) -> None:
        self.merge_decisions.append(decision)

    def flush(self) -> None:
        self._write_jsonl("graph_nodes.jsonl", self.nodes.values())
        self._write_jsonl("graph_edges.jsonl", self.edges.values())
        self._write_jsonl("path_records.jsonl", self.paths.values())
        self._write_jsonl("merge_decisions.jsonl", self.merge_decisions)

    def summary(self) -> dict[str, int]:
        dormant = sum(1 for edge in self.edges.values() if edge.status == "dormant")
        return {"nodes": len(self.nodes), "edges": len(self.edges), "paths": len(self.paths), "dormant_edges": dormant}

    def _write_jsonl(self, name: str, rows) -> None:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = to_jsonable(row) if not isinstance(row, dict) else to_jsonable(row)
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _read_jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self.run_dir / name
        if not path.exists():
            return []
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _read_nodes(self) -> list[GraphNode]:
        nodes = []
        for row in self._read_jsonl("graph_nodes.jsonl"):
            stats = row.get("stats", {})
            nodes.append(
                GraphNode(
                    id=row["id"],
                    label=row.get("label", row["id"]),
                    node_type=row.get("node_type", "checkpoint"),
                    required=[condition_from_dict(item) for item in row.get("required", [])],
                    suggested=list(row.get("suggested", [])),
                    stats=NodeStats(
                        attempts=int(stats.get("attempts", 0)),
                        successes=int(stats.get("successes", 0)),
                        avg_steps_to_goal=stats.get("avg_steps_to_goal"),
                    ),
                    metadata=dict(row.get("metadata", {})),
                )
            )
        return nodes

    def _read_edges(self) -> list[GraphEdge]:
        edges = []
        for row in self._read_jsonl("graph_edges.jsonl"):
            stats = row.get("stats", {})
            action_data = row.get("action_template", row.get("action", {"name": "unknown", "args": {}}))
            edges.append(
                GraphEdge(
                    id=row["id"],
                    from_node=row["from_node"],
                    to_node=row["to_node"],
                    action_template=action_from_dict(action_data) if isinstance(action_data, dict) else Action.parse(str(action_data)),
                    hard_preconditions=[condition_from_dict(item) for item in row.get("hard_preconditions", [])],
                    soft_preconditions=list(row.get("soft_preconditions", [])),
                    effects=[condition_from_dict(item) for item in row.get("effects", [])],
                    stats=EdgeStats(
                        attempts=int(stats.get("attempts", 0)),
                        successes=int(stats.get("successes", 0)),
                        avg_cost=stats.get("avg_cost"),
                        failure_reasons=dict(stats.get("failure_reasons", {})),
                    ),
                    status=row.get("status", "active"),
                )
            )
        return edges

    def _read_paths(self) -> list[PathRecord]:
        paths = []
        for row in self._read_jsonl("path_records.jsonl"):
            stats = row.get("stats", {})
            paths.append(
                PathRecord(
                    id=row["id"],
                    task_id=row["task_id"],
                    start_signature=row.get("start_signature", "unknown"),
                    goal_node=row.get("goal_node", "unknown"),
                    edge_ids=list(row.get("edge_ids", [])),
                    stats=PathStats(
                        attempts=int(stats.get("attempts", 0)),
                        successes=int(stats.get("successes", 0)),
                        avg_steps=stats.get("avg_steps"),
                        avg_cost=stats.get("avg_cost"),
                    ),
                    last_used_episode=int(row.get("last_used_episode", 0)),
                )
            )
        return paths


def append_jsonl(path: str | Path, row: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
