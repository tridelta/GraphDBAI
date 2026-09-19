from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from experience_graph.core.serialization import to_jsonable
from experience_graph.envs.textcraft import MyTextCraftAdapter
from experience_graph.evaluation.logger import EvaluationLogger
from experience_graph.graph.organizer import GraphOrganizer
from experience_graph.graph.store import JsonGraphStore
from experience_graph.runners.episode_runner import EpisodeResult, EpisodeRunner, RunContext
from experience_graph.scripts.run_experiment import (
    AGENTS,
    RETRIEVAL_MODES,
    VARIANTS,
    BudgetTracker,
    append_jsonl,
    build_agent,
    build_case_schedule,
    build_config,
    build_retriever,
    build_run_id,
    build_variant_config,
    copy_warm_start_files,
    filter_case_ids,
    filter_cases,
    read_jsonl,
    resolve_model,
    resolve_warm_start_path,
    validate_resume_config,
)


GRAPH_STATE_FILES = (
    "graph_nodes.jsonl",
    "graph_edges.jsonl",
    "path_records.jsonl",
    "merge_decisions.jsonl",
)
WORKER_LOGS_TO_MERGE = ("steps.jsonl", "experience_views.jsonl")
PARALLEL_PROTOCOL = "round_batch"
SUPPORTED_PARALLEL_AGENTS = {"scripted", "graph"}


@dataclass
class ParallelEpisodeResult:
    round_index: int
    case_order_index: int
    episode_index: int
    case_id: str
    worker_dir: Path
    result: EpisodeResult


def main() -> None:
    load_dotenv()
    args = parse_args()
    validate_args(args)

    run_id = args.run_id or build_run_id(args)
    run_dir = Path(args.run_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    effective_llm_provider = "none" if args.agent == "scripted" else (args.llm_provider or os.getenv("EXPERIENCE_GRAPH_LLM_PROVIDER", "openai"))
    effective_llm_model = resolve_model(effective_llm_provider, args.llm_model)

    env = MyTextCraftAdapter(args.cases, args.rules)
    selected_cases = filter_cases(env.cases, args.difficulty, args.task_id, args.solvable_only)
    selected_cases = filter_case_ids(selected_cases, args.case_ids, env.cases)
    args.episodes = len(selected_cases) * args.rounds
    case_ids = build_case_schedule(selected_cases, args.episodes, args.seed, args.case_schedule)
    variant_config = build_variant_config(args.variant, args.top_k)
    config = build_parallel_config(args, run_id, case_ids, selected_cases, variant_config, effective_llm_provider, effective_llm_model)

    config_path = run_dir / "config.yaml"
    if args.resume and config_path.exists():
        existing = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        validate_resume_config(existing, config, args.allow_config_mismatch)
        case_ids = list(existing.get("case_ids", case_ids))
        config = existing
    elif config_path.exists():
        raise FileExistsError(f"Run directory already has config.yaml. Choose another --run-id: {run_dir}")
    else:
        config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    if args.warm_start_run and not args.resume:
        copy_warm_start_files(resolve_warm_start_path(args.warm_start_run, Path(args.run_dir)), run_dir)

    tmp_root = run_dir / ".parallel_tmp"
    completed_episodes = count_jsonl_rows(run_dir / "metrics.jsonl") if args.resume else 0
    completed_rounds = completed_episodes // len(selected_cases)
    if completed_episodes % len(selected_cases) != 0:
        raise ValueError(
            f"Parallel resume only supports complete rounds. Found {completed_episodes} completed episodes "
            f"for {len(selected_cases)} cases per round."
        )
    if not args.resume and tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        remove_incomplete_tmp_rounds(tmp_root, completed_rounds)

    store = JsonGraphStore(run_dir, load_existing=args.resume or bool(args.warm_start_run))
    organizer = GraphOrganizer(
        store,
        min_attempts_for_dormant=args.min_attempts_for_dormant,
        dormant_success_threshold=args.dormant_success_threshold,
        enable_node_merging=variant_config["enable_node_merging"],
        learn_failure_preconditions=variant_config["learn_failure_preconditions"],
    )
    logger = EvaluationLogger(run_dir)
    budget = BudgetTracker(
        max_budget_rmb=args.max_budget_rmb,
        input_price_per_million=args.input_price_per_million_rmb,
        output_price_per_million=args.output_price_per_million_rmb,
    )
    llm_usage_cumulative = resume_usage_snapshot(run_dir) if args.resume else empty_usage()

    if completed_rounds:
        print(f"Resuming {run_id}: {completed_rounds} completed rounds found.", flush=True)

    for round_index in range(completed_rounds, args.rounds):
        round_start = round_index * len(selected_cases)
        round_case_ids = case_ids[round_start : round_start + len(selected_cases)]
        snapshot_dir = tmp_root / f"round_{round_index:03d}" / "graph_snapshot"
        copy_graph_snapshot(run_dir, snapshot_dir)
        snapshot_summary = JsonGraphStore(snapshot_dir, load_existing=True).summary()
        print(
            f"Round {round_index + 1}/{args.rounds}: starting {len(round_case_ids)} cases with "
            f"graph={snapshot_summary['nodes']}N/{snapshot_summary['edges']}E/{snapshot_summary['paths']}P",
            flush=True,
        )

        round_results = run_round_workers(
            args=args,
            run_id=run_id,
            round_index=round_index,
            round_case_ids=round_case_ids,
            selected_case_count=len(selected_cases),
            snapshot_dir=snapshot_dir,
            tmp_root=tmp_root,
            effective_llm_provider=effective_llm_provider,
            effective_llm_model=effective_llm_model,
            variant_config=variant_config,
        )

        round_successes = 0
        last_cost = budget.estimate(llm_usage_cumulative)
        for item in sorted(round_results, key=lambda result: result.case_order_index):
            merge_worker_logs(item.worker_dir, logger)
            update = organizer.integrate(item.result.experience)
            graph_summary = organizer.store.summary()
            llm_usage_delta = dict(item.result.metrics.get("llm_usage_delta", {}))
            llm_usage_cumulative = add_usage(llm_usage_cumulative, llm_usage_delta)
            metrics = finalize_metrics(item, update, graph_summary, llm_usage_delta, llm_usage_cumulative)
            item.result.experience.metrics.update(metrics)
            episode_row = to_jsonable(item.result.experience)
            episode_row.update(metrics)
            episode_row["metrics"] = metrics
            logger.write_jsonl("episodes.jsonl", episode_row)
            logger.write_jsonl("metrics.jsonl", metrics)
            last_cost = budget.estimate(llm_usage_cumulative)
            append_budget_progress(run_dir, run_id, item, args, last_cost, llm_usage_cumulative)
            round_successes += 1 if metrics["success"] else 0

        graph_summary = organizer.store.summary()
        print(
            f"Round {round_index + 1}/{args.rounds}: integrated {len(round_results)} cases, "
            f"successes={round_successes}, graph={graph_summary['nodes']}N/{graph_summary['edges']}E/{graph_summary['paths']}P, "
            f"cost≈{last_cost:.4f} RMB",
            flush=True,
        )
        if budget.should_stop(last_cost):
            stop_payload = {
                "run_id": run_id,
                "round_index": round_index,
                "estimated_cost_rmb": last_cost,
                "max_budget_rmb": args.max_budget_rmb,
                "reason": "budget_limit_reached_after_parallel_round",
            }
            (run_dir / "budget_stop.json").write_text(json.dumps(stop_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Budget stop written to {run_dir / 'budget_stop.json'}", flush=True)
            break

    print(f"Parallel run written to {run_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ExperienceGraph experiments with round-level parallel batches.")
    parser.add_argument("--agent", choices=AGENTS, default="graph")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="full")
    parser.add_argument("--cases", default="world_cases/mytextcraft_world_v1_short10.yaml")
    parser.add_argument("--rules", default="world_cases/textcraft_rules.yaml")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--difficulty", choices=["all", "easy", "medium", "hard", "impossible"], default="all")
    parser.add_argument("--task-id", default="all")
    parser.add_argument("--solvable-only", action="store_true", help="Exclude cases whose oracle marks them unsolvable.")
    parser.add_argument("--case-ids", default=None, help="Comma-separated case ids to run after task/difficulty filtering.")
    parser.add_argument("--case-schedule", choices=["ordered", "shuffled_cycle", "random"], default="ordered")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cross-task-mode", choices=["same_task", "cross_task_actions", "all_tasks"], default="all_tasks")
    parser.add_argument("--retrieval-mode", choices=RETRIEVAL_MODES, default="graph")
    parser.add_argument("--token-budget", type=int, default=1000)
    parser.add_argument("--min-attempts-for-dormant", type=int, default=5)
    parser.add_argument("--dormant-success-threshold", type=float, default=0.1)
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=int(os.getenv("EG_LLM_MAX_TOKENS", "4096")))
    parser.add_argument("--llm-retry-max-tokens", type=int, default=int(os.getenv("EG_LLM_RETRY_MAX_TOKENS", "8192")))
    parser.add_argument("--llm-retries", type=int, default=int(os.getenv("EG_LLM_RETRIES", "1")))
    parser.add_argument("--max-budget-rmb", type=float, default=None)
    parser.add_argument("--stop-on-env-failure", action="store_true", help="End an episode after the first environment/precondition failure.")
    parser.add_argument("--input-price-per-million-rmb", type=float, default=float(os.getenv("EG_INPUT_PRICE_PER_M_RMB", "1.0")))
    parser.add_argument("--output-price-per-million-rmb", type=float, default=float(os.getenv("EG_OUTPUT_PRICE_PER_M_RMB", "3.0")))
    parser.add_argument("--run-dir", default=os.getenv("EXPERIENCE_GRAPH_RUN_DIR", "runs"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--warm-start-run", default=None, help="Existing run id or path whose graph files seed this new run.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.agent not in SUPPORTED_PARALLEL_AGENTS:
        raise SystemExit(f"Parallel round runner currently supports agents: {sorted(SUPPORTED_PARALLEL_AGENTS)}.")
    if args.rounds <= 0 or args.max_workers <= 0 or args.max_steps <= 0 or args.top_k < 0 or args.llm_retries < 0:
        raise SystemExit("rounds, max-workers, max-steps, top-k, and llm-retries must be valid non-negative settings.")


def build_parallel_config(
    args: argparse.Namespace,
    run_id: str,
    case_ids: list[str],
    selected_cases: list[str],
    variant_config: dict[str, Any],
    llm_provider: str,
    llm_model: str,
) -> dict[str, Any]:
    config = build_config(args, run_id, case_ids, variant_config, llm_provider, llm_model)
    config.update(
        {
            "parallel_protocol": PARALLEL_PROTOCOL,
            "parallel_rounds": True,
            "max_workers": args.max_workers,
            "round_case_count": len(selected_cases),
            "round_case_ids": selected_cases,
            "experience_integration": "after_each_parallel_round",
        }
    )
    return config


def run_round_workers(
    args: argparse.Namespace,
    run_id: str,
    round_index: int,
    round_case_ids: list[str],
    selected_case_count: int,
    snapshot_dir: Path,
    tmp_root: Path,
    effective_llm_provider: str,
    effective_llm_model: str,
    variant_config: dict[str, Any],
) -> list[ParallelEpisodeResult]:
    results: list[ParallelEpisodeResult] = []
    max_workers = min(args.max_workers, len(round_case_ids))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for case_order_index, case_id in enumerate(round_case_ids):
            episode_index = round_index * selected_case_count + case_order_index
            futures.append(
                executor.submit(
                    run_parallel_episode,
                    args,
                    run_id,
                    round_index,
                    case_order_index,
                    selected_case_count,
                    episode_index,
                    case_id,
                    snapshot_dir,
                    tmp_root,
                    effective_llm_provider,
                    effective_llm_model,
                    variant_config,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            status = "success" if item.result.metrics["success"] else f"failed:{item.result.metrics['failure_reason']}"
            print(
                f"Round {round_index + 1}: {item.case_id} finished {status} "
                f"steps={item.result.metrics['steps']} episode_index={item.episode_index}",
                flush=True,
            )
            results.append(item)
    return results


def run_parallel_episode(
    args: argparse.Namespace,
    run_id: str,
    round_index: int,
    case_order_index: int,
    round_case_count: int,
    episode_index: int,
    case_id: str,
    snapshot_dir: Path,
    tmp_root: Path,
    effective_llm_provider: str,
    effective_llm_model: str,
    variant_config: dict[str, Any],
) -> ParallelEpisodeResult:
    worker_dir = tmp_root / f"round_{round_index:03d}" / f"case_{case_order_index:03d}_{safe_name(case_id)}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    env = MyTextCraftAdapter(args.cases, args.rules)
    plans = {item_case_id: case["oracle"].get("reference_plan", []) for item_case_id, case in env.cases.items()}
    agent = build_agent(
        args.agent,
        plans,
        worker_dir,
        effective_llm_provider,
        effective_llm_model,
        variant_config,
        args.llm_max_tokens,
        args.llm_retry_max_tokens,
        args.llm_retries,
    )
    store = JsonGraphStore(snapshot_dir, load_existing=True)
    organizer = GraphOrganizer(
        store,
        min_attempts_for_dormant=args.min_attempts_for_dormant,
        dormant_success_threshold=args.dormant_success_threshold,
        enable_node_merging=variant_config["enable_node_merging"],
        learn_failure_preconditions=variant_config["learn_failure_preconditions"],
    )
    retriever = build_retriever(args, store, variant_config)
    run_context = RunContext(
        run_id=run_id,
        agent=args.agent,
        variant=args.variant,
        seed=args.seed,
        metadata={
            "task_id": args.task_id,
            "case_ids_filter": args.case_ids,
            "case_schedule_mode": args.case_schedule,
            "llm_provider": effective_llm_provider,
            "llm_model": effective_llm_model,
            "retrieval_mode": args.retrieval_mode,
            "parallel_protocol": PARALLEL_PROTOCOL,
            "round_index": round_index,
            "round_number": round_index + 1,
            "case_order_index": case_order_index,
            "round_case_count": round_case_count,
            "worker_log_dir": str(worker_dir),
        },
    )
    runner = EpisodeRunner(
        env,
        agent,
        organizer,
        retriever,
        EvaluationLogger(worker_dir),
        max_steps=args.max_steps,
        run_context=run_context,
        continue_after_env_failure=not args.stop_on_env_failure,
        integrate_experience=False,
        update_agent_memory=False,
    )
    episode_seed = args.seed + episode_index
    result = runner.run_episode(
        episode_id=f"ep_{episode_index:04d}",
        case_id=case_id,
        seed=episode_seed,
        context={"episode_index": episode_index},
    )
    return ParallelEpisodeResult(
        round_index=round_index,
        case_order_index=case_order_index,
        episode_index=episode_index,
        case_id=case_id,
        worker_dir=worker_dir,
        result=result,
    )


def copy_graph_snapshot(source_run: Path, snapshot_dir: Path) -> None:
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in GRAPH_STATE_FILES:
        source = source_run / name
        if source.exists():
            shutil.copyfile(source, snapshot_dir / name)


def merge_worker_logs(worker_dir: Path, logger: EvaluationLogger) -> None:
    for name in WORKER_LOGS_TO_MERGE:
        for row in read_jsonl(worker_dir / name):
            logger.write_jsonl(name, row)


def finalize_metrics(
    item: ParallelEpisodeResult,
    update,
    graph_summary: dict[str, int],
    llm_usage_delta: dict[str, Any],
    llm_usage_cumulative: dict[str, Any],
) -> dict[str, Any]:
    metrics = dict(item.result.metrics)
    metrics.update(
        {
            "round_index": item.round_index,
            "round_number": item.round_index + 1,
            "case_order_index": item.case_order_index,
            "parallel_protocol": PARALLEL_PROTOCOL,
            "graph_nodes": graph_summary["nodes"],
            "graph_edges": graph_summary["edges"],
            "graph_paths": graph_summary["paths"],
            "dormant_edges": graph_summary.get("dormant_edges", 0),
            "added_nodes": update.added_nodes,
            "added_edges": update.added_edges,
            "updated_edges": update.updated_edges,
            "added_paths": update.added_paths,
            "llm_usage_delta": llm_usage_delta,
            "llm_usage_cumulative": llm_usage_cumulative,
        }
    )
    return metrics


def append_budget_progress(
    run_dir: Path,
    run_id: str,
    item: ParallelEpisodeResult,
    args: argparse.Namespace,
    estimated_cost: float,
    llm_usage_cumulative: dict[str, Any],
) -> None:
    append_jsonl(
        run_dir / "budget_progress.jsonl",
        {
            "run_id": run_id,
            "episode_id": item.result.metrics["episode_id"],
            "episode_index": item.episode_index,
            "round_index": item.round_index,
            "case_order_index": item.case_order_index,
            "case_id": item.case_id,
            "parallel_protocol": PARALLEL_PROTOCOL,
            "estimated_cost_rmb": estimated_cost,
            "max_budget_rmb": args.max_budget_rmb,
            "llm_max_tokens": args.llm_max_tokens,
            "llm_retry_max_tokens": args.llm_retry_max_tokens,
            "llm_retries": args.llm_retries,
            "llm_usage_cumulative": llm_usage_cumulative,
        },
    )


def empty_usage() -> dict[str, Any]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_prompt_tokens": 0,
        "estimated_completion_tokens": 0,
        "token_source": "none",
    }


def add_usage(cumulative: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    numeric_keys = [
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "estimated_prompt_tokens",
        "estimated_completion_tokens",
    ]
    updated = dict(cumulative)
    for key in numeric_keys:
        updated[key] = int(updated.get(key, 0) or 0) + int(delta.get(key, 0) or 0)
    token_source = delta.get("token_source")
    if token_source and token_source != "none":
        updated["token_source"] = token_source
    return updated


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def resume_usage_snapshot(run_dir: Path) -> dict[str, Any]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    if not metrics:
        return empty_usage()
    usage = metrics[-1].get("llm_usage_cumulative")
    return dict(usage) if isinstance(usage, dict) else empty_usage()


def remove_incomplete_tmp_rounds(tmp_root: Path, completed_rounds: int) -> None:
    if not tmp_root.exists():
        return
    for child in tmp_root.iterdir():
        if not child.is_dir() or not child.name.startswith("round_"):
            continue
        try:
            round_index = int(child.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if round_index >= completed_rounds:
            shutil.rmtree(child)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


if __name__ == "__main__":
    main()
