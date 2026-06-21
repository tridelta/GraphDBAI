from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


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


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def fmt_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def fmt_set(values: set[Any]) -> str:
    return ", ".join(str(value) for value in sorted(values, key=str)) or "unknown"


def collect_run_details(summary_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        run_path = Path(row["run_path"])
        metrics = read_jsonl(run_path / "metrics.jsonl")
        steps = read_jsonl(run_path / "steps.jsonl")
        budget = read_jsonl(run_path / "budget_progress.jsonl")
        config = read_yaml(run_path / "config.yaml")
        failures = Counter(m.get("failure_reason") or "success" for m in metrics)
        parse_errors = sum(1 for step in steps if step.get("llm_output") == {} and step.get("llm_raw_response"))
        diagnostics = collect_prompt_diagnostics(steps)
        details[row["run_id"]] = {
            "config": config,
            "metrics": metrics,
            "steps": steps,
            "budget": budget,
            "failures": failures,
            "parse_errors": parse_errors,
            "prompt_diagnostics": diagnostics,
            "final_cost": budget[-1].get("estimated_cost_rmb", 0) if budget else 0,
            "first_failure_step": next((s for s in steps if s.get("ok") is False or s.get("failure_reason")), None),
        }
    return details


def collect_prompt_diagnostics(steps: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    hidden_count = 0
    for step in steps:
        diag = step.get("prompt_diagnostics") or {}
        if diag.get("prompt_hidden_facts"):
            hidden_count += 1
        totals["candidate_paths"] += int(diag.get("prompt_candidate_paths") or 0)
        output = step.get("llm_input_messages") or []
        for message in output:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            try:
                payload = json.loads(message.get("content", ""))
            except (TypeError, json.JSONDecodeError):
                continue
            for candidate in payload.get("candidate_paths", []) if isinstance(payload, dict) else []:
                if isinstance(candidate, dict) and candidate.get("transfer_scope") not in {None, "same_task"}:
                    totals["cross_task_candidate_paths"] += 1
        totals["retrieved_skills"] += int(diag.get("prompt_retrieved_skills") or 0)
        totals["retrieved_trajectories"] += int(diag.get("prompt_retrieved_trajectories") or 0)
        if int(step.get("repeated_action_count") or 0) >= 3:
            totals["repeated_action_alerts"] += 1
        totals["llm_retries"] += int(step.get("llm_retry_count") or 0)
        finish_reason = step.get("llm_finish_reason")
        if finish_reason:
            totals[f"finish_reason:{finish_reason}"] += 1
    step_count = len(steps)
    return {
        "steps": step_count,
        "prompt_hidden_facts": hidden_count,
        "avg_candidate_paths": totals["candidate_paths"] / step_count if step_count else 0.0,
        "avg_cross_task_candidate_paths": totals["cross_task_candidate_paths"] / step_count if step_count else 0.0,
        "avg_retrieved_skills": totals["retrieved_skills"] / step_count if step_count else 0.0,
        "avg_retrieved_trajectories": totals["retrieved_trajectories"] / step_count if step_count else 0.0,
        "repeated_action_alerts": totals["repeated_action_alerts"],
        "llm_retries": totals["llm_retries"],
        "finish_reasons": "; ".join(
            f"{key.split(':', 1)[1]}: {count}" for key, count in sorted(totals.items()) if key.startswith("finish_reason:")
        ),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def setup_summary(details: dict[str, dict[str, Any]]) -> dict[str, str]:
    configs = [detail["config"] for detail in details.values()]
    completed = [len(detail["metrics"]) for detail in details.values()]
    return {
        "providers": fmt_set({config.get("llm_provider") for config in configs}),
        "models": fmt_set({config.get("llm_model") for config in configs}),
        "difficulties": fmt_set({config.get("difficulty") for config in configs}),
        "seeds": fmt_set({config.get("seed") for config in configs}),
        "episodes_requested": fmt_set({config.get("episodes") for config in configs}),
        "episodes_completed": fmt_set(set(completed)),
        "max_steps": fmt_set({config.get("max_steps") for config in configs}),
        "case_schedule": fmt_set({config.get("case_schedule") for config in configs}),
    }


def first_interesting_step(detail: dict[str, Any]) -> dict[str, Any] | None:
    for step in detail["steps"]:
        diag = step.get("prompt_diagnostics") or {}
        if diag.get("prompt_candidate_paths") or diag.get("prompt_retrieved_skills") or diag.get("prompt_retrieved_trajectories"):
            return step
    return detail.get("first_failure_step")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Stage 2 pilot report from analyzed runs.")
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--viewer", default="tools/episode_log_viewer.html")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    rows = read_csv(analysis_dir / "main_comparison.csv")
    details = collect_run_details(rows)
    setup = setup_summary(details)
    total_cost = sum(float(details[row["run_id"]]["final_cost"] or 0) for row in rows)
    providers = {str(detail["config"].get("llm_provider")) for detail in details.values()}
    uses_external_llm = any(provider not in {"fake", "none"} for provider in providers)
    source_note = "real external-LLM run logs" if uses_external_llm else "local smoke/preflight run logs"
    llm_claim_status = "Pilot observation" if uses_external_llm else "Not evaluated in local preflight"

    lines: list[str] = [
        "# Stage 2 Pilot Report",
        "",
        f"This report is generated from {source_note}. Treat all numbers as pilot observations, not final paper claims.",
        "",
        "## Setup",
        "",
        f"- Provider/model: {setup['providers']} / `{setup['models']}`.",
        f"- Difficulty: `{setup['difficulties']}`; seed: `{setup['seeds']}`; case schedule: `{setup['case_schedule']}`.",
        f"- Episodes requested/completed per run: {setup['episodes_requested']} / {setup['episodes_completed']}; max steps: {setup['max_steps']}.",
        f"- Estimated total cost: {total_cost:.4f} RMB using the configured token-price estimate.",
        "- Figures are generated with `matplotlib.pyplot` from the analyzed JSONL/CSV artifacts.",
        "",
        "## Audit Context",
        "",
        "- Earlier `stage2_pilot_real_s2c_*` runs should be treated as affected by prompt leakage and stale ambiguous-state logging.",
        "- This report uses the analyzed run prefix passed to `tools/analyze_stage2.py`; keep these pilot results separate from `[PROJECTED]` paper claims.",
        "",
        "## Main Results",
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

    lines.extend(["", "## Prompt and Retrieval Diagnostics", ""])
    prompt_rows = []
    for row in sorted(rows, key=lambda r: (r["agent"], r["variant"])):
        diag = details[row["run_id"]]["prompt_diagnostics"]
        prompt_rows.append([
            row["agent"],
            diag["prompt_hidden_facts"],
            fmt_float(diag["avg_candidate_paths"]),
            fmt_float(diag["avg_cross_task_candidate_paths"]),
            fmt_float(diag["avg_retrieved_skills"]),
            fmt_float(diag["avg_retrieved_trajectories"]),
            diag["repeated_action_alerts"],
            diag["llm_retries"],
            diag["finish_reasons"],
        ])
    lines.extend(markdown_table(["agent", "hidden_fact_prompts", "avg_candidate_paths", "avg_cross_task_paths", "avg_retrieved_skills", "avg_retrieved_trajectories", "repeated_action_alerts", "llm_retries", "finish_reasons"], prompt_rows))

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
    hidden_total = sum(details[row["run_id"]]["prompt_diagnostics"]["prompt_hidden_facts"] for row in rows)
    lines.append(f"- Prompt hidden-fact detections in this analyzed set: {hidden_total}.")
    lines.append("- ReAct, Reflexion, SkillLibrary, and VectorTrajectory do not consume graph candidate paths unless they are explicitly wired to do so; any runner-built graph metrics should be interpreted separately from agent-consumed graph context.")
    lines.append(f"- The current per-run episode count ({setup['episodes_completed']}) is useful for prompt, logging, budget, and basic behavior checks, but it is not enough for performance claims.")

    interesting = []
    for row in sorted(rows, key=lambda r: (r["agent"], r["variant"])):
        step = first_interesting_step(details[row["run_id"]])
        if step:
            action = step.get("action") or step.get("llm_output") or {}
            interesting.append([row["agent"], step.get("episode_id", ""), step.get("step_index", ""), step.get("failure_reason", ""), json.dumps(action, ensure_ascii=False)[:120]])
    if interesting:
        lines.extend(["", "## Log Inspection Pointers", ""])
        lines.extend(markdown_table(["agent", "episode", "step", "failure_reason", "action_or_output"], interesting[:8]))

    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- Main table: `{analysis_dir / 'main_comparison.csv'}`",
        f"- Token cost table: `{analysis_dir / 'token_cost.csv'}`",
        f"- Analysis summary JSON: `{analysis_dir / 'analysis_summary.json'}`",
        f"- Graph growth table: `{analysis_dir / 'graph_growth.csv'}`",
        f"- Learning curve: `{analysis_dir / 'figures' / 'learning_curve_medium.png'}`",
        f"- Graph growth figure: `{analysis_dir / 'figures' / 'graph_growth.png'}`",
        f"- Token cost figure: `{analysis_dir / 'figures' / 'token_cost.png'}`",
        f"- Episode viewer: `{args.viewer}`. Drag any run's `steps.jsonl` into the page to inspect prompts and decisions.",
        "",
        "## Claim-Evidence Map",
        "",
        "| Claim | Evidence | Status |",
        "| --- | --- | --- |",
        "| Fixed Stage 2 logs no longer expose hidden facts in prompts | `hidden_fact_prompts` table above | Supported for this pilot if the count is 0 |",
        "| Agents can complete the medium MyTextCraft flow with real external LLM calls | `success_rate` and failure table above | " + llm_claim_status + " |",
        "| ExperienceGraph graph context is available to the graph agent | `avg_candidate_paths` plus graph growth artifacts | Pilot observation |",
        "| Pilot numbers replace paper-scale `[PROJECTED]` claims | Only 5 episodes per method | Not supported |",
        "",
        "## Limitations and Deviations from Original Draft",
        "",
        "- This is a Stage 2 pilot, not a 300-episode multi-seed full run.",
        "- Baseline and ablation results should remain separate from paper main tables until the planned scale is complete.",
        "- If semantic node merging is not enabled, node-merge quality claims must stay limited to deterministic rule-based merging.",
    ])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()




