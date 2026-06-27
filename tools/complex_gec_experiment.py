from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

CASE_IDS = "GEC_004,GEC_005,GEC_006"
TASK_ID = "golden_equipment_chain"
DEFAULT_RUN_ID = "graph_gec_3x3_deepseek_manual"
DEFAULT_RUN_DIR = "runs"
DEFAULT_CASES = "world_cases/task_families"
DEFAULT_RULES = "world_cases/textcraft_rules.yaml"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def infer_cycle_length(case_ids: list[str]) -> int:
    if not case_ids:
        return 0
    for size in range(1, len(case_ids) + 1):
        pattern = case_ids[:size]
        if all(case_ids[index] == pattern[index % size] for index in range(len(case_ids))):
            return size
    return len(case_ids)


def load_config_case_ids(run_path: Path) -> list[str]:
    config_path = run_path / "config.yaml"
    if not config_path.exists():
        return CASE_IDS.split(",")
    try:
        import yaml
    except ImportError:
        return CASE_IDS.split(",")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return list(config.get("case_ids") or CASE_IDS.split(","))


def build_run_command(args: argparse.Namespace) -> list[str]:
    command = [
        "uv",
        "run",
        "eg-run-experiment",
        "--agent",
        "graph",
        "--variant",
        args.variant,
        "--cases",
        args.cases,
        "--rules",
        args.rules,
        "--task-id",
        TASK_ID,
        "--case-ids",
        args.case_ids,
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--case-schedule",
        "ordered",
        "--run-dir",
        args.run_dir,
        "--run-id",
        args.run_id,
        "--top-k",
        str(args.top_k),
        "--llm-provider",
        args.provider,
        "--llm-retries",
        str(args.llm_retries),
        "--max-budget-rmb",
        str(args.max_budget_rmb),
    ]
    if args.model:
        command.extend(["--llm-model", args.model])
    if args.resume:
        command.append("--resume")
    if args.allow_config_mismatch:
        command.append("--allow-config-mismatch")
    return command


def run_experiment(args: argparse.Namespace) -> int:
    provider = args.provider.lower()
    if provider in {"deepseek", "openai"} and not args.ack_external_api:
        print(
            "External API provider selected. Re-run with --ack-external-api after confirming prompts, task state, "
            "and experience-graph summaries may be sent to the provider.",
            file=sys.stderr,
        )
        return 2
    command = build_run_command(args)
    print("Command:")
    print(" ".join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def summarize_rounds(metrics: list[dict[str, Any]], steps: list[dict[str, Any]], cycle_len: int) -> list[dict[str, Any]]:
    step_rows_by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in steps:
        index = row.get("episode_index")
        if isinstance(index, int):
            step_rows_by_episode.setdefault(index, []).append(row)

    rounds: list[dict[str, Any]] = []
    for start in range(0, len(metrics), cycle_len):
        group = metrics[start : start + cycle_len]
        if not group:
            continue
        successes = [row for row in group if row.get("success")]
        success_steps = [int(row.get("steps", 0) or 0) for row in successes]
        candidate_counts: list[int] = []
        for row in group:
            episode_index = row.get("episode_index")
            for step in step_rows_by_episode.get(episode_index, []):
                candidate_counts.append(int(step.get("candidate_paths", 0) or 0))
        last = group[-1]
        rounds.append(
            {
                "round": len(rounds) + 1,
                "episodes": len(group),
                "case_ids": ",".join(str(row.get("case_id", "")) for row in group),
                "successes": len(successes),
                "success_rate": len(successes) / len(group),
                "avg_steps_success": sum(success_steps) / len(success_steps) if success_steps else 0.0,
                "avg_candidate_paths": sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0,
                "graph_nodes": int(last.get("graph_nodes", 0) or 0),
                "graph_edges": int(last.get("graph_edges", 0) or 0),
                "graph_paths": int(last.get("graph_paths", 0) or 0),
            }
        )
    return rounds


def trend_note(rounds: list[dict[str, Any]]) -> str:
    if len(rounds) < 2:
        return "Not enough completed rounds to judge trend."
    first = rounds[0]
    last = rounds[-1]
    success_delta = last["success_rate"] - first["success_rate"]
    steps_delta = last["avg_steps_success"] - first["avg_steps_success"]
    if success_delta > 0:
        return "Success rate improved across rounds."
    if success_delta == 0 and first["success_rate"] > 0 and steps_delta < 0:
        return "Success rate stayed level and successful episodes used fewer steps."
    if success_delta == 0 and abs(steps_delta) < 1e-9:
        return "No measured efficiency improvement across rounds."
    if success_delta < 0:
        return "Success rate declined across rounds."
    return "Trend is mixed; inspect episode logs before making a claim."


def write_analysis(run_path: Path, rounds: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
    csv_path = run_path / "complex_round_summary.csv"
    json_path = run_path / "complex_round_summary.json"
    report_path = run_path / "complex_experiment_report.md"

    fieldnames = [
        "round",
        "episodes",
        "case_ids",
        "successes",
        "success_rate",
        "avg_steps_success",
        "avg_candidate_paths",
        "graph_nodes",
        "graph_edges",
        "graph_paths",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rounds)
    json_path.write_text(json.dumps(rounds, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    total_successes = sum(1 for row in metrics if row.get("success"))
    lines = [
        "# Complex GEC Experiment Report",
        "",
        "Last updated: 2026-06-21",
        "",
        f"Run directory: `{run_path}`",
        f"Episodes completed: {len(metrics)}",
        f"Overall success: {total_successes}/{len(metrics)}" if metrics else "Overall success: no completed episodes",
        f"Trend note: {trend_note(rounds)}",
        "",
        "| Round | Cases | Success | Avg successful steps | Avg candidate paths | Graph |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rounds:
        lines.append(
            "| {round} | {case_ids} | {successes}/{episodes} ({success_rate:.1%}) | {avg_steps_success:.2f} | {avg_candidate_paths:.2f} | {graph_nodes} N / {graph_edges} E / {graph_paths} P |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation rule:",
            "- Evidence for learning: later rounds improve success rate, or keep success rate while reducing successful steps.",
            "- Weak or no evidence: repeated rounds stay flat, especially when graph paths stop growing after round 1.",
            "- Inspect `steps.jsonl` when failures are `invalid_llm_output` or repeated precondition failures.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")


def analyze_run(args: argparse.Namespace) -> int:
    run_path = Path(args.run_dir) / args.run_id
    metrics = read_jsonl(run_path / "metrics.jsonl")
    steps = read_jsonl(run_path / "steps.jsonl")
    if not metrics:
        print(f"No completed metrics found in {run_path}", file=sys.stderr)
        return 1
    case_ids = load_config_case_ids(run_path)
    cycle_len = infer_cycle_length(case_ids)
    rounds = summarize_rounds(metrics, steps, cycle_len or 3)
    write_analysis(run_path, rounds, metrics)
    for row in rounds:
        print(
            "Round {round}: success {successes}/{episodes}, avg_steps={avg_steps_success:.2f}, "
            "avg_candidates={avg_candidate_paths:.2f}, graph={graph_nodes}N/{graph_edges}E/{graph_paths}P".format(**row)
        )
    print(trend_note(rounds))
    return 0


def panel(args: argparse.Namespace) -> int:
    command = ["uv", "run", "eg-panel", "--run-dir", args.run_dir, "--host", args.host, "--port", str(args.port)]
    print("Open: http://{}:{}/".format(args.host, args.port))
    return subprocess.run(command, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and analyze the complex golden-equipment-chain experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the 3x3 ExperienceGraph experiment.")
    run_parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    run_parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    run_parser.add_argument("--cases", default=DEFAULT_CASES)
    run_parser.add_argument("--rules", default=DEFAULT_RULES)
    run_parser.add_argument("--case-ids", default=CASE_IDS)
    run_parser.add_argument("--episodes", type=int, default=9)
    run_parser.add_argument("--max-steps", type=int, default=14)
    run_parser.add_argument("--provider", default="deepseek", choices=["deepseek", "openai", "fake"])
    run_parser.add_argument("--model", default="deepseek-v4-pro")
    run_parser.add_argument("--variant", default="full")
    run_parser.add_argument("--top-k", type=int, default=5)
    run_parser.add_argument("--llm-retries", type=int, default=1)
    run_parser.add_argument("--max-budget-rmb", type=float, default=20.0)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--allow-config-mismatch", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--ack-external-api", action="store_true")
    run_parser.set_defaults(func=run_experiment)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze round trend for an existing run.")
    analyze_parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    analyze_parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    analyze_parser.set_defaults(func=analyze_run)

    panel_parser = subparsers.add_parser("panel", help="Start the ExperienceGraph panel.")
    panel_parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    panel_parser.add_argument("--host", default="127.0.0.1")
    panel_parser.add_argument("--port", type=int, default=8765)
    panel_parser.set_defaults(func=panel)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
