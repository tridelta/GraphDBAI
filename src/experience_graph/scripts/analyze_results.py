from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ExperienceGraph experiment runs.")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--run-id-prefix", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    runs = load_runs(run_dir, run_id_prefix=args.run_id_prefix)
    summary_rows = build_main_summary(runs)
    write_csv(output_dir / "main_comparison.csv", summary_rows)
    write_csv(output_dir / "token_cost.csv", build_token_cost(summary_rows))
    write_csv(output_dir / "graph_growth.csv", build_graph_growth(runs))
    write_csv(output_dir / "learning_curve_medium.csv", build_learning_curve(runs, args.window, difficulty="medium"))
    write_csv(output_dir / "ablation.csv", build_ablation(summary_rows))
    write_summary_markdown(output_dir / "summary_tables.md", summary_rows, runs)
    write_experiment_writeup(output_dir / "experiment_writeup.md", summary_rows, runs)
    write_learning_curve_png(output_dir / "figures" / "learning_curve_medium.png", build_learning_curve(runs, args.window, difficulty="medium"))
    write_graph_growth_png(output_dir / "figures" / "graph_growth.png", build_graph_growth(runs))
    write_token_cost_png(output_dir / "figures" / "token_cost.png", summary_rows)
    print(f"Analysis written to {output_dir}")


def load_runs(run_dir: Path, run_id_prefix: str | None = None) -> list[dict[str, Any]]:
    runs = []
    for config_path in sorted(run_dir.glob("*/config.yaml")):
        run_path = config_path.parent
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        run_id = str(config.get("run_id") or run_path.name)
        if run_id_prefix and not run_id.startswith(run_id_prefix):
            continue
        metrics = read_jsonl(run_path / "metrics.jsonl")
        steps = read_jsonl(run_path / "steps.jsonl")
        budget = read_jsonl(run_path / "budget_progress.jsonl")
        runs.append({"run_path": run_path, "config": config, "metrics": metrics, "steps": steps, "budget": budget})
    return runs


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


def build_main_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        config = run["config"]
        metrics = run["metrics"]
        if not metrics:
            continue
        successes = [row for row in metrics if row.get("success")]
        final = metrics[-1]
        usage = final.get("llm_usage_cumulative", {})
        total_tokens = int(usage.get("total_tokens") or usage.get("estimated_prompt_tokens", 0) + usage.get("estimated_completion_tokens", 0) or 0)
        total_steps_success = [row.get("steps", 0) for row in successes]
        rows.append(
            {
                "run_id": config.get("run_id"),
                "agent": config.get("agent"),
                "variant": config.get("variant"),
                "difficulty": config.get("difficulty"),
                "seed": config.get("seed"),
                "episodes_requested": config.get("episodes"),
                "episodes_completed": len(metrics),
                "success_rate": safe_div(len(successes), len(metrics)),
                "avg_steps_success": mean(total_steps_success) if total_steps_success else "",
                "total_tokens": total_tokens,
                "tokens_per_episode": safe_div(total_tokens, len(metrics)),
                "tokens_per_success": safe_div(total_tokens, len(successes)) if successes else "",
                "graph_nodes": final.get("graph_nodes", 0),
                "graph_edges": final.get("graph_edges", 0),
                "graph_paths": final.get("graph_paths", 0),
                "dormant_edges": final.get("dormant_edges", 0),
                "llm_provider": config.get("llm_provider"),
                "llm_model": config.get("llm_model"),
                "run_path": str(run["run_path"]),
            }
        )
    return rows


def build_token_cost(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "run_id": row["run_id"],
            "agent": row["agent"],
            "variant": row["variant"],
            "difficulty": row["difficulty"],
            "seed": row["seed"],
            "total_tokens": row["total_tokens"],
            "tokens_per_episode": row["tokens_per_episode"],
            "tokens_per_success": row["tokens_per_success"],
            "llm_model": row["llm_model"],
        }
        for row in summary_rows
    ]


def build_graph_growth(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        config = run["config"]
        for row in run["metrics"]:
            rows.append(
                {
                    "run_id": config.get("run_id"),
                    "agent": config.get("agent"),
                    "variant": config.get("variant"),
                    "difficulty": config.get("difficulty"),
                    "seed": config.get("seed"),
                    "episode_index": row.get("episode_index", episode_index_from_id(row.get("episode_id"))),
                    "graph_nodes": row.get("graph_nodes", 0),
                    "graph_edges": row.get("graph_edges", 0),
                    "graph_paths": row.get("graph_paths", 0),
                    "dormant_edges": row.get("dormant_edges", 0),
                    "added_nodes": row.get("added_nodes", 0),
                    "added_edges": row.get("added_edges", 0),
                    "added_paths": row.get("added_paths", 0),
                }
            )
    return rows


def build_learning_curve(runs: list[dict[str, Any]], window: int, difficulty: str) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        config = run["config"]
        if config.get("difficulty") != difficulty:
            continue
        metrics = run["metrics"]
        for idx, _ in enumerate(metrics):
            start = max(0, idx - window + 1)
            slice_rows = metrics[start : idx + 1]
            rows.append(
                {
                    "run_id": config.get("run_id"),
                    "agent": config.get("agent"),
                    "variant": config.get("variant"),
                    "seed": config.get("seed"),
                    "episode_index": idx,
                    "window": window,
                    "window_success_rate": safe_div(sum(1 for row in slice_rows if row.get("success")), len(slice_rows)),
                }
            )
    return rows


def build_ablation(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in summary_rows if row.get("agent") == "graph"]
    full_by_difficulty = {}
    for row in rows:
        if row.get("variant") == "full":
            full_by_difficulty[row.get("difficulty")] = row.get("success_rate", 0)
    output = []
    for row in rows:
        full = full_by_difficulty.get(row.get("difficulty"), "")
        delta = row["success_rate"] - full if isinstance(full, (int, float)) else ""
        output.append({**row, "delta_vs_full": delta})
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path: Path, rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> None:
    lines = ["# Experiment Summary", "", f"Runs analyzed: {len(runs)}", ""]
    lines.append("## Run-Level Results")
    lines.extend(markdown_table(rows, ["run_id", "agent", "variant", "difficulty", "seed", "episodes_completed", "success_rate", "avg_steps_success", "total_tokens"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_experiment_writeup(path: Path, rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> None:
    lines = [
        "# Experiment Writeup Draft",
        "",
        "This file is generated from current experiment logs and is intended to be merged into `paper/paper_draft.md` after validation.",
        "",
        "## Experimental Setup",
        "",
        f"Analyzed runs: {len(runs)}.",
        "",
        "## Implemented Baselines",
        "",
        "Current code supports ReAct, Reflexion, VectorTrajectory, SkillLibrary, and ExperienceGraph. Interpret results according to completed run scale.",
        "",
        "## Main Results",
        "",
    ]
    lines.extend(markdown_table(rows, ["agent", "variant", "difficulty", "seed", "episodes_completed", "success_rate", "avg_steps_success", "tokens_per_episode"]))
    lines.extend(
        [
            "",
            "## Limitations and Deviations from Original Draft",
            "",
            "- Pilot-scale runs should not replace projected paper claims until the planned seed and episode counts are complete.",
            "- If semantic node merging is not enabled, node-merge quality claims must be restricted to deterministic rule-based merging.",
            "- If TextCraft uses fixed case schedules rather than procedural initial-state sampling, describe seed as controlling case order rather than environment distribution.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]], keys: list[str]) -> list[str]:
    if not rows:
        return ["No rows available."]
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(key, "")) for key in keys) + " |")
    return lines


def write_learning_curve_png(path: Path, rows: list[dict[str, Any]]) -> None:
    series = aggregate_series(rows, "episode_index", "window_success_rate", ["agent", "variant"])
    write_line_plot(path, series, y_label="Window SR", title="Medium Learning Curve")


def write_graph_growth_png(path: Path, rows: list[dict[str, Any]]) -> None:
    graph_rows = [row for row in rows if row.get("agent") == "graph"]
    series = aggregate_series(graph_rows, "episode_index", "graph_nodes", ["variant"])
    write_line_plot(path, series, y_label="Graph Nodes", title="ExperienceGraph Growth")


def write_token_cost_png(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [f"{row.get('agent')}/{row.get('variant')}" for row in rows]
    values = [float(row.get("tokens_per_episode") or 0) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, values, color="#2563eb")
    ax.set_ylabel("Tokens per episode")
    ax.set_title("Token Cost by Run")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def aggregate_series(rows: list[dict[str, Any]], x_key: str, y_key: str, group_keys: list[str]) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        label = "/".join(str(row.get(key)) for key in group_keys)
        x = float(row.get(x_key, 0))
        y = row.get(y_key, "")
        if y == "":
            continue
        grouped[label][x].append(float(y))
    return {label: [(x, mean(values)) for x, values in sorted(points.items())] for label, points in grouped.items()}


def write_line_plot(path: Path, series: dict[str, list[tuple[float, float]]], y_label: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for label, points in series.items():
        if not points:
            continue
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)
    ax.set_xlabel("Episode index")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if series:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def episode_index_from_id(episode_id: str | None) -> int:
    if not episode_id:
        return 0
    try:
        return int(str(episode_id).split("_")[-1])
    except ValueError:
        return 0


def safe_div(left: int | float, right: int | float) -> float:
    return left / right if right else 0.0


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()

