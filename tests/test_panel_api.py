from __future__ import annotations

import json

from fastapi.testclient import TestClient

from experience_graph.panel.app import create_app


def test_empty_runs_returns_empty(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.get("/api/runs")
    assert response.status_code == 200
    assert response.json() == []


def test_summary_reads_sample_run(tmp_path):
    run = tmp_path / "run_sample"
    run.mkdir()
    (run / "metrics.jsonl").write_text(json.dumps({"episode_id": "ep", "case_id": "TC", "success": True, "steps": 3}) + "\n", encoding="utf-8")
    (run / "graph_nodes.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "graph_edges.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "path_records.jsonl").write_text("{}\n", encoding="utf-8")
    client = TestClient(create_app(tmp_path))
    response = client.get("/api/runs/run_sample/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["episodes"] == 1
    assert data["success_rate"] == 1
