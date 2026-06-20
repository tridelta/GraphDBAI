from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


CONDITIONS = [
    ("react", "react", "full"),
    ("reflexion", "reflexion", "full"),
    ("vector_trajectory", "vector_trajectory", "full"),
    ("skill_library", "skill_library", "full"),
    ("graph_full", "graph", "full"),
    ("graph_no_graph_context", "graph", "no_graph_context"),
]
REQUIRED_FILES = [
    "config.yaml",
    "metrics.jsonl",
    "steps.jsonl",
    "experience_views.jsonl",
    "graph_nodes.jsonl",
    "graph_edges.jsonl",
    "path_records.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 real pilot conditions and build analysis artifacts.")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--prefix", default="stage2_pilot_real_s2e_")
    parser.add_argument("--seed", type=int, default=2501)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--max-budget-rmb", type=float, default=50)
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--case-schedule", default="shuffled_cycle")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-retry-max-tokens", type=int, default=8192)
    parser.add_argument("--llm-retries", type=int, default=1)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["EXPERIENCE_GRAPH_LLM_PROVIDER"] = "deepseek"
    env["DEEPSEEK_MODEL"] = args.model
    env["EG_LLM_MAX_TOKENS"] = str(args.llm_max_tokens)
    env["EG_LLM_RETRY_MAX_TOKENS"] = str(args.llm_retry_max_tokens)
    env["EG_LLM_RETRIES"] = str(args.llm_retries)
    return env


def run_id(args: argparse.Namespace, label: str) -> str:
    return f"{args.prefix}{label}_{args.difficulty}_seed{args.seed}"


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def build_command(args: argparse.Namespace, label: str, agent: str, variant: str) -> list[str]:
    rid = run_id(args, label)
    run_path = Path(args.run_dir) / rid
    metrics_count = count_jsonl_rows(run_path / "metrics.jsonl")
    if metrics_count > args.episodes:
        raise SystemExit(
            f"{rid} already has {metrics_count} completed episodes, but this run expects {args.episodes}. "
            "Use a fresh --prefix, or archive/delete that run directory before resuming."
        )
    command = [
        sys.executable,
        "-B",
        "-m",
        "experience_graph.scripts.run_experiment",
        "--agent",
        agent,
        "--variant",
        variant,
        "--difficulty",
        args.difficulty,
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--seed",
        str(args.seed),
        "--case-schedule",
        args.case_schedule,
        "--llm-provider",
        "deepseek",
        "--llm-model",
        args.model,
        "--llm-max-tokens",
        str(args.llm_max_tokens),
        "--llm-retry-max-tokens",
        str(args.llm_retry_max_tokens),
        "--llm-retries",
        str(args.llm_retries),
        "--max-budget-rmb",
        str(args.max_budget_rmb),
        "--run-dir",
        args.run_dir,
        "--run-id",
        rid,
    ]
    if (run_path / "config.yaml").exists():
        print(f"Resuming existing run: {rid}")
        command.append("--resume")
    else:
        print(f"Starting new run: {rid}")
    return command


def print_log(label: str, stdout_path: Path, stderr_path: Path) -> None:
    if stdout_path.exists():
        print(f"--- stdout: {label} ---")
        text = stdout_path.read_text(encoding="utf-8", errors="replace")
        if text:
            print(text.rstrip())
    if stderr_path.exists():
        text = stderr_path.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            print(f"--- stderr: {label} ---")
            print(text.rstrip())


def run_condition(args: argparse.Namespace, condition: tuple[str, str, str], env: dict[str, str], log_dir: Path) -> int:
    label, agent, variant = condition
    command = build_command(args, label, agent, variant)
    stdout_path = log_dir / f"{label}.out.log"
    stderr_path = log_dir / f"{label}.err.log"
    print(f"Launching process: {label}")
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=repo_root(), env=env, stdout=stdout, stderr=stderr, text=True)
    print_log(label, stdout_path, stderr_path)
    return completed.returncode


def run_parallel(args: argparse.Namespace, env: dict[str, str], log_dir: Path) -> None:
    pending = list(CONDITIONS)
    running: list[dict[str, Any]] = []
    failed = False
    while pending or running:
        while pending and len(running) < args.max_workers:
            label, agent, variant = pending.pop(0)
            command = build_command(args, label, agent, variant)
            stdout_path = log_dir / f"{label}.out.log"
            stderr_path = log_dir / f"{label}.err.log"
            print(f"Launching process: {label}")
            stdout = stdout_path.open("w", encoding="utf-8")
            stderr = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(command, cwd=repo_root(), env=env, stdout=stdout, stderr=stderr, text=True)
            running.append({"label": label, "process": process, "stdout": stdout, "stderr": stderr, "stdout_path": stdout_path, "stderr_path": stderr_path})
        time.sleep(0.5)
        still_running = []
        for record in running:
            process = record["process"]
            returncode = process.poll()
            if returncode is None:
                still_running.append(record)
                continue
            record["stdout"].close()
            record["stderr"].close()
            print_log(record["label"], record["stdout_path"], record["stderr_path"])
            if returncode != 0:
                failed = True
                print(f"ERROR: process {record['label']} exited with code {returncode}")
        running = still_running
    if failed:
        raise SystemExit(f"One or more parallel Stage 2 runs failed. See {log_dir} for logs.")


def run_analysis(args: argparse.Namespace, env: dict[str, str]) -> Path:
    analysis_dir = Path(args.run_dir) / f"{args.prefix}analysis"
    command = [
        sys.executable,
        "-B",
        "tools/analyze_stage2.py",
        "--run-dir",
        args.run_dir,
        "--prefix",
        args.prefix,
        "--output-dir",
        str(analysis_dir),
        "--window",
        str(args.window),
    ]
    subprocess.run(command, cwd=repo_root(), env=env, check=True)
    report_path = analysis_dir / "stage2_pilot_report.md"
    command = [
        sys.executable,
        "-B",
        "tools/build_stage2_report.py",
        "--analysis-dir",
        str(analysis_dir),
        "--output",
        str(report_path),
        "--viewer",
        "tools/episode_log_viewer.html",
    ]
    subprocess.run(command, cwd=repo_root(), env=env, check=True)
    return report_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(args: argparse.Namespace) -> None:
    errors = []
    run_dir = Path(args.run_dir)
    for label, agent, variant in CONDITIONS:
        rid = run_id(args, label)
        run_path = run_dir / rid
        missing = [name for name in REQUIRED_FILES if not (run_path / name).exists()]
        if missing:
            errors.append(f"{rid}: missing {missing}")
            continue
        config_text = (run_path / "config.yaml").read_text(encoding="utf-8")
        for needle in [
            f"llm_max_tokens: {args.llm_max_tokens}",
            f"llm_retry_max_tokens: {args.llm_retry_max_tokens}",
            f"llm_retries: {args.llm_retries}",
            f"agent: {agent}",
            f"variant: {variant}",
        ]:
            if needle not in config_text:
                errors.append(f"{rid}: config missing {needle}")
        metrics = read_jsonl(run_path / "metrics.jsonl")
        steps = read_jsonl(run_path / "steps.jsonl")
        hidden = sum(1 for step in steps if (step.get("prompt_diagnostics") or {}).get("prompt_hidden_facts"))
        if len(metrics) != args.episodes:
            errors.append(f"{rid}: expected {args.episodes} metrics rows, got {len(metrics)}")
        if hidden:
            errors.append(f"{rid}: prompt_hidden_facts count is {hidden}")
    fig_dir = run_dir / f"{args.prefix}analysis" / "figures"
    for name in ["learning_curve_medium.png", "graph_growth.png", "token_cost.png"]:
        path = fig_dir / name
        if not path.exists():
            errors.append(f"missing figure {path}")
            continue
        if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            errors.append(f"not a PNG file: {path}")
    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print("-", error)
        raise SystemExit(1)
    print("VALIDATION OK")


def main() -> None:
    args = parse_args()
    root = repo_root()
    os.chdir(root)
    env = build_env(args)
    if not args.analyze_only and not env.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is not set. Set it in this terminal before running the real pilot.")
    log_dir = Path(args.run_dir) / f"{args.prefix}process_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        if args.parallel:
            run_parallel(args, env, log_dir)
        else:
            failed = False
            for condition in CONDITIONS:
                failed = run_condition(args, condition, env, log_dir) != 0 or failed
            if failed:
                raise SystemExit(f"One or more Stage 2 runs failed. See {log_dir} for logs.")
    report_path = run_analysis(args, env)
    if not args.skip_validation:
        validate(args)
    print("Stage 2 real pilot package completed.")
    print(f"Analysis directory: {Path(args.run_dir) / f'{args.prefix}analysis'}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
