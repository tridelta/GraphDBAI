from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse


def create_app(run_dir: str | Path = "runs") -> FastAPI:
    base = Path(run_dir)
    app = FastAPI(title="ExperienceGraph Panel")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        if not base.exists():
            return []
        rows = []
        for path in sorted([p for p in base.iterdir() if p.is_dir()], reverse=True):
            rows.append({"run_id": path.name, "summary": summarize_run(path)})
        return rows

    @app.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str) -> dict[str, Any]:
        path = base / run_id
        if not path.exists():
            raise HTTPException(status_code=404, detail="run not found")
        return summarize_run(path)

    @app.get("/api/runs/{run_id}/episodes")
    def run_episodes(run_id: str) -> list[dict[str, Any]]:
        path = base / run_id
        if not path.exists():
            raise HTTPException(status_code=404, detail="run not found")
        return read_jsonl(path / "metrics.jsonl")

    @app.get("/api/runs/{run_id}/graph")
    def run_graph(run_id: str) -> dict[str, Any]:
        path = base / run_id
        if not path.exists():
            raise HTTPException(status_code=404, detail="run not found")
        return {
            "nodes": read_jsonl(path / "graph_nodes.jsonl"),
            "edges": read_jsonl(path / "graph_edges.jsonl"),
            "paths": read_jsonl(path / "path_records.jsonl"),
        }

    return app


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize_run(path: Path) -> dict[str, Any]:
    metrics = read_jsonl(path / "metrics.jsonl")
    graph_nodes = read_jsonl(path / "graph_nodes.jsonl")
    graph_edges = read_jsonl(path / "graph_edges.jsonl")
    graph_paths = read_jsonl(path / "path_records.jsonl")
    successes = sum(1 for row in metrics if row.get("success"))
    steps = [row.get("steps", 0) for row in metrics if row.get("success")]
    failures = [row.get("failure_reason") for row in metrics if row.get("failure_reason")]
    return {
        "run_id": path.name,
        "episodes": len(metrics),
        "successes": successes,
        "success_rate": successes / len(metrics) if metrics else 0,
        "avg_steps_success": sum(steps) / len(steps) if steps else 0,
        "graph_nodes": len(graph_nodes),
        "graph_edges": len(graph_edges),
        "graph_paths": len(graph_paths),
        "latest_failures": failures[-5:],
        "latest_episodes": metrics[-10:],
    }


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ExperienceGraph Panel</title>
  <style>
    body { font-family: Inter, system-ui, sans-serif; margin: 32px; background: #f7f7f4; color: #1d2327; }
    h1 { font-size: 24px; margin-bottom: 20px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0; }
    .card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 14px; }
    .label { color: #667; font-size: 12px; }
    .value { font-size: 24px; margin-top: 6px; }
    table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; }
    th, td { padding: 10px; border-bottom: 1px solid #eee; text-align: left; font-size: 13px; }
    .bar { height: 8px; background: #ddd; border-radius: 99px; overflow: hidden; }
    .bar span { display: block; height: 100%; background: #2f8f83; }
  </style>
</head>
<body>
  <h1>ExperienceGraph Runs</h1>
  <div id="content">Loading...</div>
  <script>
    async function load() {
      const runs = await fetch('/api/runs').then(r => r.json());
      if (!runs.length) { document.getElementById('content').innerHTML = '<p>No runs found.</p>'; return; }
      const latest = runs[0].summary;
      document.getElementById('content').innerHTML = `
        <h2>${latest.run_id}</h2>
        <div class="grid">
          ${card('Episodes', latest.episodes)}
          ${card('Success Rate', (latest.success_rate * 100).toFixed(1) + '%')}
          ${card('Avg Steps', latest.avg_steps_success.toFixed(1))}
          ${card('Graph', latest.graph_nodes + ' nodes / ' + latest.graph_edges + ' edges')}
        </div>
        <div class="card"><div class="label">Progress</div><div class="bar"><span style="width:${Math.min(100, latest.episodes / 12 * 100)}%"></span></div></div>
        <h2>Latest Episodes</h2>
        <table><thead><tr><th>Episode</th><th>Case</th><th>Status</th><th>Steps</th><th>Failure</th></tr></thead>
        <tbody>${latest.latest_episodes.map(e => `<tr><td>${e.episode_id}</td><td>${e.case_id}</td><td>${e.success ? 'success' : 'failed'}</td><td>${e.steps}</td><td>${e.failure_reason || ''}</td></tr>`).join('')}</tbody></table>
      `;
    }
    function card(label, value) { return `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`; }
    load(); setInterval(load, 3000);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Start ExperienceGraph panel.")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(create_app(args.run_dir), host=args.host, port=args.port)
