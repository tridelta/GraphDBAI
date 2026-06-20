from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def action_label(action: Any) -> str:
    if not action:
        return "none"
    if isinstance(action, str):
        return action
    args = action.get("args") or {}
    values = ", ".join(str(v) for v in args.values())
    return f"{action.get('name')}({values})" if values else str(action.get("name"))


def collect_run_details(summary_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        run_path = Path(row["run_path"])
        metrics = read_jsonl(run_path / "metrics.jsonl")
        steps = read_jsonl(run_path / "steps.jsonl")
        budget = read_jsonl(run_path / "budget_progress.jsonl")
        failures = Counter(m.get("failure_reason") or "success" for m in metrics)
        parse_errors = sum(1 for step in steps if (step.get("llm_output") == {} and step.get("llm_raw_response")))
        details[row["run_id"]] = {
            "metrics": metrics,
            "steps": steps,
            "budget": budget,
            "failures": failures,
            "parse_errors": parse_errors,
            "final_cost": budget[-1].get("estimated_cost_rmb", 0) if budget else 0,
            "first_failure_step": next((s for s in steps if s.get("ok") is False or s.get("failure_reason")), None),
        }
    return details


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Stage 2 pilot report from analyzed runs.")
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--viewer", default="tools/episode_log_viewer.html")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    rows = read_csv(analysis_dir / "main_comparison.csv")
    details = collect_run_details(rows)
    total_cost = sum(float(details[row["run_id"]]["final_cost"] or 0) for row in rows)

    lines: list[str] = [
        "# Stage 2 Pilot Report",
        "",
        "This report is generated from real DeepSeek runs. Treat all results as pilot observations, not final paper claims.",
        "",
        "## Setup",
        "",
        "- Provider/model: DeepSeek `deepseek-v4-flash`.",
        "- Difficulty: `medium`.",
        "- Seed: `2501`; all methods use the same shuffled case schedule.",
        "- Episodes per method: 3; max steps per episode: 12.",
        "- JSON mode: `response_format={\"type\": \"json_object\"}` plus an explicit JSON example in the prompt; DeepSeek max output tokens default to 2048.",
        f"- Estimated total cost: {total_cost:.4f} RMB using the configured token-price estimate.",
        "",
        "## Results",
        "",
    ]
    result_rows = []
    for row in sorted(rows, key=lambda r: (r["agent"], r["variant"])):
        detail = details[row["run_id"]]
        result_rows.append([
            row["agent"],
            row["variant"],
            row["episodes_completed"],
            fmt_float(row["success_rate"]),
            fmt_float(row["avg_steps_success"]),
            fmt_float(row["tokens_per_episode"], 1),
            fmt_float(detail["final_cost"], 4),
            f'{row["graph_nodes"]}/{row["graph_edges"]}/{row["graph_paths"]}',
        ])
    lines.extend(markdown_table(["agent", "variant", "episodes", "success_rate", "avg_success_steps", "tokens/episode", "cost_rmb", "nodes/edges/paths"], result_rows))

    lines.extend(["", "## Failure Modes", ""])
    failure_rows = []
    for row in sorted(rows, key=lambda r: r["agent"]):
        detail = details[row["run_id"]]
        failure_text = "; ".join(f"{name}: {count}" for name, count in detail["failures"].items() if name != "success") or "none"
        failure_rows.append([row["agent"], failure_text, detail["parse_errors"]])
    lines.extend(markdown_table(["agent", "episode failure reasons", "json_parse_or_empty_outputs"], failure_rows))

    lines.extend(["", "## Observations", ""])
    graph_row = next((row for row in rows if row["agent"] == "graph"), None)
    if graph_row:
        lines.append(f"- ExperienceGraph completed {graph_row['episodes_completed']} episodes with success_rate={fmt_float(graph_row['success_rate'])}; final graph size was {graph_row['graph_nodes']} nodes, {graph_row['graph_edges']} edges, {graph_row['graph_paths']} paths.")
    lines.extend([
        "- The shared TextCraft action guide materially improved valid action selection compared with the first connectivity run.",
        "- Root-level action JSON such as `{\"name\":..., \"args\":...}` is now accepted as a valid action, reducing avoidable schema failures.",
        "- Malformed or truncated model output is now logged as an invalid step instead of crashing the run.",
        "- Pilot scale is too small for performance claims; it is useful mainly for prompt, logging, budget, and tool readiness checks.",
        "",
        "## Artifacts",
        "",
        f"- Main table: `{analysis_dir / 'main_comparison.csv'}`",
        f"- Token cost table: `{analysis_dir / 'token_cost.csv'}`",
        f"- Graph growth table: `{analysis_dir / 'graph_growth.csv'}`",
        f"- Learning curve: `{analysis_dir / 'figures' / 'learning_curve_medium.svg'}`",
        f"- Graph growth figure: `{analysis_dir / 'figures' / 'graph_growth.svg'}`",
        f"- Episode viewer: `{args.viewer}`. Drag any run's `steps.jsonl` into the page to inspect prompts and decisions.",
        "",
        "## Full Run Readiness",
        "",
        "- Ready: DeepSeek API calls, JSON mode, per-step prompt/output logging, resume config checks, budget progress logging, baseline memory files, graph files, filtered analysis.",
        "- Needs before full run: add one retry for `{}` or malformed JSON, decide whether impossible cases count as task failure or correct refusal, and run at least one real graph ablation if budget allows.",
    ])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
