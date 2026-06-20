from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experience_graph.core.models import GraphEdge, GraphNode, PathRecord
from experience_graph.core.serialization import to_jsonable


class JsonGraphStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}
        self.paths: dict[str, PathRecord] = {}
        self.merge_decisions: list[dict[str, Any]] = []

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


def append_jsonl(path: str | Path, row: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
