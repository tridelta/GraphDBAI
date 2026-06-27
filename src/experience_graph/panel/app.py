from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

COMPLEX_TASK_ID = "golden_equipment_chain"
COMPLEX_CASE_IDS = "GEC_004,GEC_005,GEC_006"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
DEFAULT_JOB_DIR = Path("output") / "experiment_jobs"


class ExperimentStartRequest(BaseModel):
    run_id: str = "graph_gec_3x3_deepseek_manual"
    provider: str = "deepseek"
    model: str | None = "deepseek-v4-pro"
    case_ids: str = COMPLEX_CASE_IDS
    episodes: int = 9
    max_steps: int = 14
    max_budget_rmb: float = 20.0
    llm_retries: int = 1
    variant: str = "full"
    top_k: int = 5
    resume: bool = False
    acknowledge_external_api: bool = False


def create_app(run_dir: str | Path = "runs") -> FastAPI:
    base = Path(run_dir)
    app = FastAPI(title="ExperienceGraph Panel")
    app.state.jobs = {}
    app.state.job_dir = DEFAULT_JOB_DIR

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

    @app.get("/api/experiment/defaults")
    def experiment_defaults() -> dict[str, Any]:
        return {
            "task_id": COMPLEX_TASK_ID,
            "case_ids": COMPLEX_CASE_IDS,
            "episodes": 9,
            "max_steps": 14,
            "run_id": "graph_gec_3x3_deepseek_manual",
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "max_budget_rmb": 20,
            "panel_run_dir": str(base),
        }

    @app.post("/api/experiment/start")
    def start_experiment(request: ExperimentStartRequest) -> dict[str, Any]:
        run_id = request.run_id.strip()
        if not RUN_ID_RE.match(run_id):
            raise HTTPException(status_code=400, detail="run_id may only contain letters, numbers, dot, dash, and underscore")
        provider = request.provider.strip().lower()
        if provider not in {"deepseek", "openai", "fake"}:
            raise HTTPException(status_code=400, detail="provider must be deepseek, openai, or fake")
        if provider in {"deepseek", "openai"} and not request.acknowledge_external_api:
            raise HTTPException(status_code=400, detail="external API acknowledgement is required")
        if request.episodes <= 0 or request.max_steps <= 0 or request.top_k < 0 or request.llm_retries < 0:
            raise HTTPException(status_code=400, detail="numeric experiment settings are out of range")
        if is_job_running(app.state.jobs.get(run_id)):
            raise HTTPException(status_code=409, detail="run is already active")
        run_path = base / run_id
        if run_path.exists() and (run_path / "config.yaml").exists() and not request.resume:
            raise HTTPException(status_code=409, detail="run already exists; enable resume or choose another run_id")

        log_dir = Path("output") / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{run_id}.out.log"
        stderr_path = log_dir / f"{run_id}.err.log"
        command = build_experiment_command(request, run_id, str(base), provider)
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=Path.cwd(), stdout=stdout, stderr=stderr)
        app.state.jobs[run_id] = {
            "run_id": run_id,
            "pid": process.pid,
            "process": process,
            "command": command,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_job_metadata(app.state.job_dir, app.state.jobs[run_id])
        return experiment_status_for_job(app.state.jobs[run_id], base)

    @app.get("/api/experiment/status/{run_id}")
    def experiment_status(run_id: str) -> dict[str, Any]:
        job = app.state.jobs.get(run_id)
        if job:
            return experiment_status_for_job(job, base)
        metadata = read_job_metadata(app.state.job_dir, run_id)
        if metadata:
            return experiment_status_for_metadata(metadata, base)
        run_path = base / run_id
        if not run_path.exists():
            raise HTTPException(status_code=404, detail="run not found")
        return {"run_id": run_id, "active": False, "returncode": None, "summary": summarize_run(run_path), "stdout_tail": "", "stderr_tail": ""}

    return app


def build_experiment_command(request: ExperimentStartRequest, run_id: str, run_dir: str, provider: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experience_graph.scripts.run_experiment",
        "--agent",
        "graph",
        "--variant",
        request.variant,
        "--cases",
        "world_cases/task_families",
        "--rules",
        "world_cases/textcraft_rules.yaml",
        "--task-id",
        COMPLEX_TASK_ID,
        "--case-ids",
        request.case_ids,
        "--episodes",
        str(request.episodes),
        "--max-steps",
        str(request.max_steps),
        "--case-schedule",
        "ordered",
        "--run-dir",
        run_dir,
        "--run-id",
        run_id,
        "--top-k",
        str(request.top_k),
        "--llm-provider",
        provider,
        "--llm-retries",
        str(request.llm_retries),
        "--max-budget-rmb",
        str(request.max_budget_rmb),
    ]
    if request.model:
        command.extend(["--llm-model", request.model])
    if request.resume:
        command.append("--resume")
    return command


def is_job_running(job: dict[str, Any] | None) -> bool:
    if not job:
        return False
    process = job.get("process")
    return process is not None and process.poll() is None


def experiment_status_for_job(job: dict[str, Any], base: Path) -> dict[str, Any]:
    process = job["process"]
    returncode = process.poll()
    run_id = job["run_id"]
    run_path = base / run_id
    return {
        "run_id": run_id,
        "active": returncode is None,
        "returncode": returncode,
        "pid": job.get("pid"),
        "command": job.get("command"),
        "started_at": job.get("started_at"),
        "summary": summarize_run(run_path) if run_path.exists() else None,
        "stdout_log": job.get("stdout_log"),
        "stderr_log": job.get("stderr_log"),
        "stdout_tail": read_tail(Path(job["stdout_log"])),
        "stderr_tail": read_tail(Path(job["stderr_log"])),
    }


def experiment_status_for_metadata(metadata: dict[str, Any], base: Path) -> dict[str, Any]:
    run_id = str(metadata["run_id"])
    run_path = base / run_id
    active = pid_is_running(metadata.get("pid"))
    return {
        "run_id": run_id,
        "active": active,
        "returncode": None if active else metadata.get("returncode"),
        "pid": metadata.get("pid"),
        "command": metadata.get("command"),
        "started_at": metadata.get("started_at"),
        "summary": summarize_run(run_path) if run_path.exists() else None,
        "stdout_log": metadata.get("stdout_log"),
        "stderr_log": metadata.get("stderr_log"),
        "stdout_tail": read_tail(Path(metadata.get("stdout_log") or "")),
        "stderr_tail": read_tail(Path(metadata.get("stderr_log") or "")),
    }


def write_job_metadata(job_dir: Path, job: dict[str, Any]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": job["run_id"],
        "pid": job.get("pid"),
        "command": job.get("command"),
        "stdout_log": job.get("stdout_log"),
        "stderr_log": job.get("stderr_log"),
        "started_at": job.get("started_at"),
    }
    (job_dir / f"{job['run_id']}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_job_metadata(job_dir: Path, run_id: str) -> dict[str, Any] | None:
    if not RUN_ID_RE.match(run_id):
        return None
    path = job_dir / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pid_is_running(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    if os.name == "nt":
        command = f"$p = Get-Process -Id {pid_int} -ErrorAction SilentlyContinue; if ($p) {{ exit 0 }} else {{ exit 1 }}"
        result = subprocess.run(["powershell", "-NoProfile", "-Command", command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    try:
        os.kill(pid_int, 0)
    except OSError:
        return False
    return True


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def read_tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def summarize_run(path: Path) -> dict[str, Any]:
    metrics = read_jsonl(path / "metrics.jsonl")
    graph_nodes = read_jsonl(path / "graph_nodes.jsonl")
    graph_edges = read_jsonl(path / "graph_edges.jsonl")
    graph_paths = read_jsonl(path / "path_records.jsonl")
    config = read_yaml(path / "config.yaml")
    successes = sum(1 for row in metrics if row.get("success"))
    steps = [row.get("steps", 0) for row in metrics if row.get("success")]
    failures = [row.get("failure_reason") for row in metrics if row.get("failure_reason")]
    return {
        "run_id": path.name,
        "config": config,
        "episodes": len(metrics),
        "successes": successes,
        "success_rate": successes / len(metrics) if metrics else 0,
        "avg_steps_success": sum(steps) / len(steps) if steps else 0,
        "graph_nodes": len(graph_nodes),
        "graph_edges": len(graph_edges),
        "graph_paths": len(graph_paths),
        "latest_failures": failures[-5:],
        "latest_episodes": metrics[-10:],
        "round_summaries": round_summaries(metrics, config),
        "graph_growth": graph_growth(metrics),
    }


def round_summaries(metrics: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    cycle_len = infer_cycle_length(config.get("case_ids") or [])
    if cycle_len <= 0:
        return []
    rows = []
    for start in range(0, len(metrics), cycle_len):
        group = metrics[start : start + cycle_len]
        if not group:
            continue
        successes = [row for row in group if row.get("success")]
        success_steps = [row.get("steps", 0) for row in successes]
        rows.append(
            {
                "round": len(rows) + 1,
                "episodes": len(group),
                "successes": len(successes),
                "success_rate": len(successes) / len(group),
                "avg_steps_success": sum(success_steps) / len(success_steps) if success_steps else 0,
                "case_ids": [row.get("case_id") for row in group],
                "graph_nodes": group[-1].get("graph_nodes", 0),
                "graph_edges": group[-1].get("graph_edges", 0),
                "graph_paths": group[-1].get("graph_paths", 0),
            }
        )
    return rows


def infer_cycle_length(case_ids: list[str]) -> int:
    if not case_ids:
        return 0
    for size in range(1, len(case_ids) + 1):
        pattern = case_ids[:size]
        if all(case_ids[index] == pattern[index % size] for index in range(len(case_ids))):
            return size
    return len(case_ids)


def graph_growth(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "episode_index": row.get("episode_index"),
            "episode_id": row.get("episode_id"),
            "case_id": row.get("case_id"),
            "success": row.get("success"),
            "steps": row.get("steps"),
            "graph_nodes": row.get("graph_nodes", 0),
            "graph_edges": row.get("graph_edges", 0),
            "graph_paths": row.get("graph_paths", 0),
        }
        for row in metrics
    ]


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ExperienceGraph Panel</title>
  <style>
    body { font-family: Inter, system-ui, sans-serif; margin: 0; background: #f6f6f2; color: #1d2327; }
    header { padding: 18px 24px; border-bottom: 1px solid #d8d8d0; background: #ffffff; display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    main { padding: 20px 24px 28px; }
    h1 { font-size: 22px; margin: 0; }
    h2 { font-size: 17px; margin: 0 0 12px; }
    label { display: grid; gap: 5px; color: #566; font-size: 12px; }
    input, select { border: 1px solid #c9ccc7; border-radius: 6px; padding: 8px 9px; font: inherit; background: #fff; min-width: 0; }
    button { border: 1px solid #1d655f; background: #23766f; color: #fff; border-radius: 6px; padding: 9px 12px; font: inherit; cursor: pointer; }
    button.secondary { background: #fff; color: #1d2327; border-color: #c9ccc7; }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; margin-bottom: 18px; }
    th, td { padding: 10px; border-bottom: 1px solid #eee; text-align: left; font-size: 13px; vertical-align: top; }
    .layout { display: grid; grid-template-columns: minmax(310px, 430px) minmax(0, 1fr); gap: 16px; align-items: start; }
    .panel { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px; }
    .formgrid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .wide { grid-column: 1 / -1; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 12px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 0 0 16px; }
    .card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 14px; }
    .label { color: #667; font-size: 12px; }
    .value { font-size: 23px; margin-top: 6px; }
    .spark { display: flex; align-items: end; gap: 4px; height: 72px; padding-top: 8px; }
    .spark span { flex: 1; min-width: 10px; background: #5373c7; border-radius: 3px 3px 0 0; }
    .muted { color: #667; }
    .ok { color: #137a4f; font-weight: 600; }
    .bad { color: #b33b32; font-weight: 600; }
    .status { font-size: 13px; color: #334; min-height: 20px; }
    .risk { display: flex; gap: 8px; align-items: start; font-size: 13px; color: #3d4548; line-height: 1.35; }
    .risk input { margin-top: 2px; min-width: auto; }
    .log { white-space: pre-wrap; background: #20242a; color: #f2f2ee; border-radius: 8px; padding: 10px; max-height: 220px; overflow: auto; font-size: 12px; }
    .runlist { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    @media (max-width: 920px) { .layout { grid-template-columns: 1fr; } header { align-items: start; flex-direction: column; } }
  </style>
</head>
<body>
  <header>
    <h1>ExperienceGraph Lab</h1>
    <div class="runlist"><select id="runSelect"></select><button class="secondary" onclick="loadSelectedRun()">Open</button></div>
  </header>
  <main class="layout">
    <section class="panel">
      <h2>Experiment</h2>
      <div class="formgrid">
        <label class="wide">Run ID<input id="runId" value="graph_gec_3x3_deepseek_manual"></label>
        <label>Provider<select id="provider"><option>deepseek</option><option>openai</option><option>fake</option></select></label>
        <label>Model<input id="model" value="deepseek-v4-pro"></label>
        <label class="wide">Cases<input id="caseIds" value="GEC_004,GEC_005,GEC_006"></label>
        <label>Episodes<input id="episodes" type="number" value="9" min="1"></label>
        <label>Max Steps<input id="maxSteps" type="number" value="14" min="1"></label>
        <label>Budget RMB<input id="budget" type="number" value="20" min="0" step="1"></label>
        <label>Retries<input id="retries" type="number" value="1" min="0"></label>
        <label>Variant<select id="variant"><option>full</option><option>no_graph_context</option><option>no_exploration</option><option>no_statistics</option><option>random_retrieval</option><option>no_node_merging</option><option>no_failure_preconditions</option></select></label>
        <label>Top K<input id="topK" type="number" value="5" min="0"></label>
        <label class="wide"><span class="risk"><input id="resume" type="checkbox">Resume existing run</span></label>
        <label class="wide"><span class="risk"><input id="ack" type="checkbox">I acknowledge that external providers receive prompts, visible task state, and experience summaries for this run.</span></label>
      </div>
      <div class="actions"><button id="startBtn" onclick="startExperiment()">Start</button><button class="secondary" onclick="refreshStatus()">Refresh</button></div>
      <p id="jobStatus" class="status"></p>
      <h2>Log</h2>
      <div id="log" class="log"></div>
    </section>
    <section>
      <div id="summary">Loading...</div>
    </section>
  </main>
  <script>
    let selectedRunId = '';
    let activeRunId = '';

    async function loadRuns() {
      const runs = await fetch('/api/runs').then(r => r.json());
      const select = document.getElementById('runSelect');
      select.innerHTML = runs.map(r => `<option value="${esc(r.run_id)}">${esc(r.run_id)}</option>`).join('');
      if (!selectedRunId && runs.length) selectedRunId = runs[0].run_id;
      if (selectedRunId) select.value = selectedRunId;
      if (selectedRunId) await renderRun(selectedRunId);
      if (!runs.length) document.getElementById('summary').innerHTML = '<div class="panel muted">No runs found.</div>';
    }

    async function loadSelectedRun() {
      selectedRunId = document.getElementById('runSelect').value;
      await renderRun(selectedRunId);
    }

    async function startExperiment() {
      const body = {
        run_id: value('runId'), provider: value('provider'), model: value('model'), case_ids: value('caseIds'),
        episodes: numberValue('episodes'), max_steps: numberValue('maxSteps'), max_budget_rmb: numberValue('budget'),
        llm_retries: numberValue('retries'), variant: value('variant'), top_k: numberValue('topK'),
        resume: checked('resume'), acknowledge_external_api: checked('ack')
      };
      document.getElementById('startBtn').disabled = true;
      const response = await fetch('/api/experiment/start', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      const data = await response.json();
      document.getElementById('startBtn').disabled = false;
      if (!response.ok) { setStatus(data.detail || 'Failed to start'); return; }
      activeRunId = data.run_id;
      selectedRunId = data.run_id;
      setStatus(data.active ? `Running pid ${data.pid}` : 'Finished');
      renderStatus(data);
      await loadRuns();
    }

    async function refreshStatus() {
      const runId = activeRunId || selectedRunId || value('runId');
      if (!runId) return;
      const response = await fetch(`/api/experiment/status/${encodeURIComponent(runId)}`);
      const data = await response.json();
      if (!response.ok) { setStatus(data.detail || 'No status'); return; }
      renderStatus(data);
      if (data.summary) renderSummary(data.summary);
      await loadRuns();
    }

    function renderStatus(data) {
      setStatus(data.active ? `Running pid ${data.pid}` : `Stopped${data.returncode === null ? '' : ' returncode ' + data.returncode}`);
      document.getElementById('log').textContent = [data.stdout_tail || '', data.stderr_tail || ''].filter(Boolean).join('\n--- stderr ---\n');
      if (data.summary) selectedRunId = data.summary.run_id;
    }

    async function renderRun(runId) {
      if (!runId) return;
      const summary = await fetch(`/api/runs/${encodeURIComponent(runId)}/summary`).then(r => r.json());
      renderSummary(summary);
    }

    function renderSummary(latest) {
      document.getElementById('summary').innerHTML = `
        <div class="grid">
          ${card('Episodes', latest.episodes)}
          ${card('Success Rate', pct(latest.success_rate))}
          ${card('Avg Steps', Number(latest.avg_steps_success || 0).toFixed(1))}
          ${card('Graph', `${latest.graph_nodes} N / ${latest.graph_edges} E / ${latest.graph_paths} P`)}
        </div>
        <div class="panel"><h2>${esc(latest.run_id)}</h2><div class="label">Graph Path Growth</div>${spark(latest.graph_growth.map(e => e.graph_paths))}</div>
        <h2>Rounds</h2>
        <table><thead><tr><th>Round</th><th>Cases</th><th>Success</th><th>Avg Steps</th><th>Graph</th></tr></thead>
        <tbody>${latest.round_summaries.map(r => `<tr><td>${r.round}</td><td>${esc(r.case_ids.join(', '))}</td><td>${r.successes}/${r.episodes} (${pct(r.success_rate)})</td><td>${Number(r.avg_steps_success || 0).toFixed(1)}</td><td>${r.graph_nodes} N / ${r.graph_edges} E / ${r.graph_paths} P</td></tr>`).join('')}</tbody></table>
        <h2>Latest Episodes</h2>
        <table><thead><tr><th>Episode</th><th>Case</th><th>Status</th><th>Steps</th><th>Failure</th></tr></thead>
        <tbody>${latest.latest_episodes.map(e => `<tr><td>${esc(e.episode_id || '')}</td><td>${esc(e.case_id || '')}</td><td class="${e.success ? 'ok' : 'bad'}">${e.success ? 'success' : 'failed'}</td><td>${e.steps || 0}</td><td>${esc(e.failure_reason || '')}</td></tr>`).join('')}</tbody></table>
      `;
    }

    function card(label, value) { return `<div class="card"><div class="label">${esc(label)}</div><div class="value">${value}</div></div>`; }
    function pct(value) { return (Number(value || 0) * 100).toFixed(1) + '%'; }
    function spark(values) {
      if (!values.length) return '<p class="muted">No graph data yet.</p>';
      const max = Math.max(...values, 1);
      return `<div class="spark">${values.map(v => `<span title="${v}" style="height:${Math.max(5, v / max * 64)}px"></span>`).join('')}</div>`;
    }
    function value(id) { return document.getElementById(id).value.trim(); }
    function numberValue(id) { return Number(document.getElementById(id).value); }
    function checked(id) { return document.getElementById(id).checked; }
    function setStatus(text) { document.getElementById('jobStatus').textContent = text; }
    function esc(text) { return String(text).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }

    loadRuns(); setInterval(refreshStatus, 5000);
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


if __name__ == "__main__":
    main()
