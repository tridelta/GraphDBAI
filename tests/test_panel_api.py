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


def test_summary_builds_round_summaries(tmp_path):
    run = tmp_path / "run_rounds"
    run.mkdir()
    rows = [
        {"episode_id": "ep0", "episode_index": 0, "case_id": "A", "success": True, "steps": 5, "graph_nodes": 2, "graph_edges": 1, "graph_paths": 1},
        {"episode_id": "ep1", "episode_index": 1, "case_id": "B", "success": False, "steps": 12, "graph_nodes": 3, "graph_edges": 2, "graph_paths": 2},
        {"episode_id": "ep2", "episode_index": 2, "case_id": "A", "success": True, "steps": 4, "graph_nodes": 4, "graph_edges": 3, "graph_paths": 3},
        {"episode_id": "ep3", "episode_index": 3, "case_id": "B", "success": True, "steps": 6, "graph_nodes": 5, "graph_edges": 4, "graph_paths": 4},
    ]
    (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (run / "config.yaml").write_text("case_ids: [A, B, A, B]\n", encoding="utf-8")
    client = TestClient(create_app(tmp_path))
    data = client.get("/api/runs/run_rounds/summary").json()
    assert [row["successes"] for row in data["round_summaries"]] == [1, 2]
    assert data["graph_growth"][-1]["graph_paths"] == 4


def test_experiment_start_requires_external_api_ack(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/experiment/start",
        json={"run_id": "panel_ack_required", "provider": "deepseek", "acknowledge_external_api": False},
    )
    assert response.status_code == 400
    assert "acknowledgement" in response.json()["detail"]


def test_experiment_start_fake_provider_creates_job(tmp_path, monkeypatch):
    captured = {}

    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

    def fake_popen(command, cwd, stdout, stderr):
        captured["command"] = command
        captured["cwd"] = cwd
        stdout.write("started\n")
        return FakeProcess()

    monkeypatch.setattr("experience_graph.panel.app.subprocess.Popen", fake_popen)
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/experiment/start",
        json={"run_id": "panel_fake_run", "provider": "fake", "model": "", "acknowledge_external_api": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["active"] is True
    assert data["pid"] == 12345
    assert "experience_graph.scripts.run_experiment" in captured["command"]
    assert "--case-ids" in captured["command"]
