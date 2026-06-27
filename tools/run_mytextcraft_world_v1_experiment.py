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


RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
PRESETS = {
    "short10": {
        "cases": "world_cases/mytextcraft_world_v1_short10.yaml",
        "run_id": "mytextcraft_world_v1_short10_graph_full_pro_s1",
    },
    "all": {
        "cases": "world_cases/mytextcraft_world_v1.yaml",
        "run_id": "mytextcraft_world_v1_graph_full_pro_s1",
    },
}


def main() -> None:
    args = parse_args()
    apply_preset_defaults(args)
    validate_args(args)
    experiment_command = build_experiment_command(args)
    panel_command = build_panel_command(args)
    stdout_path, stderr_path, panel_stdout_path, panel_stderr_path = log_paths(args)

    if args.dry_run:
        print("Experiment command:")
        print(" ".join(experiment_command))
        print("Panel command:")
        print(" ".join(panel_command))
        print(f"Panel URL: http://{args.host}:{args.port}/")
        return

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    experiment = start_process(experiment_command, stdout_path, stderr_path, env)
    write_job_hook(args, experiment.pid, experiment_command, stdout_path, stderr_path)

    panel_started = False
    if not args.no_panel:
        if port_is_open(args.host, args.port):
            if args.restart_panel and stop_existing_panel(args.host, args.port):
                start_process(panel_command, panel_stdout_path, panel_stderr_path, env)
                panel_started = True
                print(f"Panel restarted: http://{args.host}:{args.port}/")
            else:
                print(f"Panel already running: http://{args.host}:{args.port}/")
        else:
            start_process(panel_command, panel_stdout_path, panel_stderr_path, env)
            panel_started = True
            print(f"Panel started: http://{args.host}:{args.port}/")

    print(f"Experiment started: run_id={args.run_id}, pid={experiment.pid}")
    print(f"Progress URL: http://{args.host}:{args.port}/")
    print(f"Logs: {stdout_path} / {stderr_path}")
    if not panel_started and args.no_panel:
        print("Panel was not started because --no-panel was set.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch MyTextCraft world-v1 experiment with progress WebUI hook.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="short10")
    parser.add_argument("--cases", default=None, help="Override the cases manifest selected by --preset.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--agent", default="graph", choices=["scripted", "react", "reflexion", "vector_trajectory", "skill_library", "graph"])
    parser.add_argument("--variant", default="full")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "openai", "fake"])
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-budget-rmb", type=float, default=50.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm-retries", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--case-schedule", default="ordered", choices=["ordered", "shuffled_cycle", "random"])
    parser.add_argument("--case-ids", default=None)
    parser.add_argument("--task-id", default="all")
    parser.add_argument("--cross-task-mode", default="all_tasks", choices=["same_task", "cross_task_actions", "all_tasks"])
    parser.add_argument("--include-unsolvable", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    parser.add_argument("--ack-external-api", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--restart-panel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-panel", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def apply_preset_defaults(args: argparse.Namespace) -> None:
    preset = PRESETS[args.preset]
    if args.cases is None:
        args.cases = preset["cases"]
    if args.run_id is None:
        args.run_id = preset["run_id"]


def validate_args(args: argparse.Namespace) -> None:
    if not RUN_ID_RE.match(args.run_id):
        raise SystemExit("run_id may only contain letters, numbers, dot, dash, and underscore.")
    if args.provider in {"deepseek", "openai"} and not args.ack_external_api and not args.dry_run:
        raise SystemExit("--ack-external-api is required for deepseek/openai runs.")
    if args.rounds <= 0 or args.max_steps <= 0 or args.top_k < 0 or args.llm_retries < 0:
        raise SystemExit("rounds, max_steps, top_k, and llm_retries must be valid non-negative settings.")
    run_path = Path(args.run_dir) / args.run_id
    if run_path.exists() and (run_path / "config.yaml").exists() and not args.resume and not args.dry_run:
        raise SystemExit(f"Run already exists: {run_path}. Use --resume or choose a new --run-id.")


def build_experiment_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experience_graph.scripts.run_experiment",
        "--agent",
        args.agent,
        "--variant",
        args.variant,
        "--cases",
        args.cases,
        "--rules",
        "world_cases/textcraft_rules.yaml",
        "--task-id",
        args.task_id,
        "--rounds",
        str(args.rounds),
        "--case-schedule",
        args.case_schedule,
        "--max-steps",
        str(args.max_steps),
        "--seed",
        str(args.seed),
        "--run-dir",
        args.run_dir,
        "--run-id",
        args.run_id,
        "--top-k",
        str(args.top_k),
        "--cross-task-mode",
        args.cross_task_mode,
        "--llm-provider",
        args.provider,
        "--llm-retries",
        str(args.llm_retries),
        "--max-budget-rmb",
        str(args.max_budget_rmb),
    ]
    if args.model:
        command.extend(["--llm-model", args.model])
    if args.case_ids:
        command.extend(["--case-ids", args.case_ids])
    if not args.include_unsolvable:
        command.append("--solvable-only")
    if args.resume:
        command.append("--resume")
    if args.allow_config_mismatch:
        command.append("--allow-config-mismatch")
    return command


def build_panel_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experience_graph.panel.app",
        "--run-dir",
        args.run_dir,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]


def log_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    log_dir = Path("output") / "logs"
    return (
        log_dir / f"{args.run_id}.out.log",
        log_dir / f"{args.run_id}.err.log",
        log_dir / f"{args.run_id}.panel.out.log",
        log_dir / f"{args.run_id}.panel.err.log",
    )


def start_process(command: list[str], stdout_path: Path, stderr_path: Path, env: dict[str, str]) -> subprocess.Popen:
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    stdout = stdout_path.open("a", encoding="utf-8")
    stderr = stderr_path.open("a", encoding="utf-8")
    return subprocess.Popen(command, cwd=Path.cwd(), stdout=stdout, stderr=stderr, env=env, creationflags=creationflags)


def write_job_hook(args: argparse.Namespace, pid: int, command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    job_dir = Path("output") / "experiment_jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": args.run_id,
        "pid": pid,
        "command": command,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (job_dir / f"{args.run_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def stop_existing_panel(host: str, port: int) -> bool:
    if os.name != "nt":
        return False
    command = (
        f"$conn = Get-NetTCPConnection -LocalAddress {host} -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; "
        "if (-not $conn) { exit 1 }; "
        "$proc = Get-CimInstance Win32_Process -Filter \"ProcessId=$($conn.OwningProcess)\"; "
        "if ($proc.CommandLine -notlike '*experience_graph.panel.app*') { exit 2 }; "
        "Stop-Process -Id $conn.OwningProcess -Force; exit 0"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode == 0:
        time.sleep(1)
        return True
    return False


if __name__ == "__main__":
    main()
