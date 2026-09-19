from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from experience_graph.agents.experience_graph_agent import ExperienceGraphAgent
from experience_graph.agents.react import ReActAgent
from experience_graph.agents.reflexion import ReflexionAgent
from experience_graph.agents.scripted import ScriptedAgent
from experience_graph.agents.skill_library import SkillLibraryAgent
from experience_graph.agents.vector_trajectory import VectorTrajectoryAgent
from experience_graph.envs.textcraft import MyTextCraftAdapter
from experience_graph.evaluation.logger import EvaluationLogger
from experience_graph.graph.organizer import GraphOrganizer
from experience_graph.graph.retriever import GraphRetriever
from experience_graph.graph.store import JsonGraphStore
from experience_graph.llm.client import build_llm_client
from experience_graph.runners.episode_runner import EpisodeRunner, RunContext
from experience_graph.semantic_retrieval import SemanticFallbackGraphRetriever


AGENTS = ["scripted", "react", "reflexion", "vector_trajectory", "skill_library", "graph"]
RETRIEVAL_MODES = ["graph", "semantic_fallback"]
VARIANTS = {
    "full",
    "no_exploration",
    "no_statistics",
    "random_retrieval",
    "no_node_merging",
    "no_failure_preconditions",
    "no_graph_context",
}
CONFIG_COMPARE_KEYS = [
    "agent",
    "variant",
    "cases",
    "rules",
    "seed",
    "difficulty",
    "task_id",
    "solvable_only",
    "rounds",
    "case_ids_filter",
    "case_schedule",
    "max_steps",
    "llm_provider",
    "llm_model",
    "llm_max_tokens",
    "llm_retry_max_tokens",
    "llm_retries",
    "continue_after_env_failure",
    "cross_task_mode",
    "retrieval_mode",
    "warm_start_run",
]
EPISODE_SCOPED_LOGS = (
    "steps.jsonl",
    "experience_views.jsonl",
    "budget_progress.jsonl",
)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prune_incomplete_episode_logs(run_dir: Path, completed_episodes: int) -> dict[str, int]:
    pruned: dict[str, int] = {}
    for name in EPISODE_SCOPED_LOGS:
        path = run_dir / name
        rows = read_jsonl(path)
        if not rows:
            continue
        kept = []
        for row in rows:
            episode_index = row.get("episode_index")
            if episode_index is None or int(episode_index) < completed_episodes:
                kept.append(row)
        removed = len(rows) - len(kept)
        if removed:
            write_jsonl(path, kept)
            pruned[name] = removed
    return pruned


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run ExperienceGraph experiments.")
    parser.add_argument("--agent", choices=AGENTS, default="scripted")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="full")
    parser.add_argument("--cases", default="world_cases/textcraft_cases.yaml")
    parser.add_argument("--rules", default="world_cases/textcraft_rules.yaml")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--difficulty", choices=["all", "easy", "medium", "hard", "impossible"], default="all")
    parser.add_argument("--task-id", default="diamond_set")
    parser.add_argument("--solvable-only", action="store_true", help="Exclude cases whose oracle marks them unsolvable.")
    parser.add_argument("--case-ids", default=None, help="Comma-separated case ids to run after task/difficulty filtering.")
    parser.add_argument("--case-schedule", choices=["ordered", "shuffled_cycle", "random"], default="shuffled_cycle")
    parser.add_argument("--rounds", type=int, default=None, help="Run this many full passes over the selected case set.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cross-task-mode", choices=["same_task", "cross_task_actions", "all_tasks"], default="same_task")
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    parser.add_argument("--run-dir", default=os.getenv("EXPERIENCE_GRAPH_RUN_DIR", "runs"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--warm-start-run", default=None, help="Existing run id or path whose graph/memory files seed this new run.")
    args = parser.parse_args()

    run_id = args.run_id or build_run_id(args)
    run_dir = Path(args.run_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    effective_llm_provider = "none" if args.agent == "scripted" else (args.llm_provider or os.getenv("EXPERIENCE_GRAPH_LLM_PROVIDER", "openai"))
    effective_llm_model = resolve_model(effective_llm_provider, args.llm_model)

    env = MyTextCraftAdapter(args.cases, args.rules)
    selected_cases = filter_cases(env.cases, args.difficulty, args.task_id, args.solvable_only)
    selected_cases = filter_case_ids(selected_cases, args.case_ids, env.cases)
    if args.rounds is not None:
        args.episodes = len(selected_cases) * args.rounds
    case_ids = build_case_schedule(selected_cases, args.episodes, args.seed, args.case_schedule)
    variant_config = build_variant_config(args.variant, args.top_k)
    config = build_config(args, run_id, case_ids, variant_config, effective_llm_provider, effective_llm_model)
    config_path = run_dir / "config.yaml"
    if args.resume and config_path.exists():
        existing = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        validate_resume_config(existing, config, args.allow_config_mismatch)
        existing_case_ids = list(existing.get("case_ids", []))
        if args.episodes > len(existing_case_ids):
            case_ids = case_ids[: args.episodes]
            existing["episodes"] = args.episodes
            existing["case_ids"] = case_ids
            config_path.write_text(yaml.safe_dump(existing, sort_keys=False, allow_unicode=True), encoding="utf-8")
        else:
            case_ids = existing_case_ids[: args.episodes]
        config = existing
    elif config_path.exists() and not args.resume:
        raise FileExistsError(f"Run directory already has config.yaml. Use --resume or choose another --run-id: {run_dir}")
    else:
        config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    if args.warm_start_run and not args.resume:
        copy_warm_start_files(resolve_warm_start_path(args.warm_start_run, Path(args.run_dir)), run_dir)

    plans = {case_id: env.cases[case_id]["oracle"].get("reference_plan", []) for case_id in env.cases}
    agent = build_agent(args.agent, plans, run_dir, effective_llm_provider, effective_llm_model, variant_config, args.llm_max_tokens, args.llm_retry_max_tokens, args.llm_retries)
    store = JsonGraphStore(run_dir, load_existing=args.resume or bool(args.warm_start_run))
    organizer = GraphOrganizer(
        store,
        min_attempts_for_dormant=args.min_attempts_for_dormant,
        dormant_success_threshold=args.dormant_success_threshold,
        enable_node_merging=variant_config["enable_node_merging"],
        learn_failure_preconditions=variant_config["learn_failure_preconditions"],
    )
    retriever = build_retriever(args, store, variant_config)
    logger = EvaluationLogger(run_dir)
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
        },
    )
    runner = EpisodeRunner(
        env,
        agent,
        organizer,
        retriever,
        logger,
        max_steps=args.max_steps,
        run_context=run_context,
        continue_after_env_failure=not args.stop_on_env_failure,
    )

    completed = count_completed_episodes(run_dir / "metrics.jsonl") if args.resume else 0
    if args.resume:
        pruned_logs = prune_incomplete_episode_logs(run_dir, completed)
        if pruned_logs:
            print(f"Pruned incomplete episode logs: {pruned_logs}")
    if completed:
        print(f"Resuming {run_id}: {completed} completed episodes found.")
    budget = BudgetTracker(
        max_budget_rmb=args.max_budget_rmb,
        input_price_per_million=args.input_price_per_million_rmb,
        output_price_per_million=args.output_price_per_million_rmb,
    )

    for index, case_id in enumerate(case_ids[completed:], start=completed):
        episode_seed = args.seed + index
        result = runner.run_episode(
            episode_id=f"ep_{index:04d}",
            case_id=case_id,
            seed=episode_seed,
            context={"episode_index": index},
        )
        cost = budget.estimate(result.metrics.get("llm_usage_cumulative", {}))
        append_jsonl(
            run_dir / "budget_progress.jsonl",
            {
                "run_id": run_id,
                "episode_id": result.metrics["episode_id"],
                "episode_index": index,
                "case_id": case_id,
                "estimated_cost_rmb": cost,
                "max_budget_rmb": args.max_budget_rmb,
                "llm_max_tokens": args.llm_max_tokens,
                "llm_retry_max_tokens": args.llm_retry_max_tokens,
                "llm_retries": args.llm_retries,
                "llm_usage_cumulative": result.metrics.get("llm_usage_cumulative", {}),
            },
        )
        status = "success" if result.metrics["success"] else f"failed:{result.metrics['failure_reason']}"
        print(f"{case_id}: {status} steps={result.metrics['steps']} seed={episode_seed} cost≈{cost:.4f} RMB")
        if budget.should_stop(cost):
            stop_payload = {
                "run_id": run_id,
                "episode_index": index,
                "estimated_cost_rmb": cost,
                "max_budget_rmb": args.max_budget_rmb,
                "llm_max_tokens": args.llm_max_tokens,
                "llm_retry_max_tokens": args.llm_retry_max_tokens,
                "llm_retries": args.llm_retries,
                "reason": "budget_limit_reached",
            }
            (run_dir / "budget_stop.json").write_text(json.dumps(stop_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Budget stop written to {run_dir / 'budget_stop.json'}")
            break
    print(f"Run written to {run_dir}")




def resolve_warm_start_path(value: str, run_dir: Path) -> Path:
    path = Path(value)
    if not path.exists():
        path = run_dir / value
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Warm-start run not found: {value}")
    return path


def copy_warm_start_files(source_run: Path, target_run: Path) -> None:
    names = [
        "graph_nodes.jsonl",
        "graph_edges.jsonl",
        "path_records.jsonl",
        "merge_decisions.jsonl",
        "agent_memory_reflexion.jsonl",
        "agent_memory_skill_library.jsonl",
        "agent_memory_vector_trajectory.jsonl",
    ]
    copied = []
    for name in names:
        source = source_run / name
        target = target_run / name
        if not source.exists() or target.exists():
            continue
        shutil.copyfile(source, target)
        copied.append(name)
    print(f"Warm-started {target_run.name} from {source_run}: {copied or 'no reusable files found'}")

def build_run_id(args: argparse.Namespace) -> str:
    timestamp = time.strftime("run_%Y%m%d_%H%M%S")
    retrieval_suffix = "" if args.retrieval_mode == "graph" else f"_{args.retrieval_mode}"
    return f"{timestamp}_{args.agent}_{args.variant}{retrieval_suffix}_seed{args.seed}"


def resolve_model(provider: str, requested_model: str | None) -> str:
    if requested_model:
        return requested_model
    if provider == "deepseek":
        return os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    if provider == "fake":
        return "fake"
    return "none"


def filter_cases(cases: dict[str, dict[str, Any]], difficulty: str, task_id: str, solvable_only: bool = False) -> list[str]:
    case_ids = []
    for case_id, case in cases.items():
        if difficulty != "all" and case.get("difficulty") != difficulty:
            continue
        if task_id != "all" and case.get("task", {}).get("id") != task_id:
            continue
        if solvable_only and case.get("oracle", {}).get("solvable") is False:
            continue
        case_ids.append(case_id)
    if not case_ids:
        raise ValueError(f"No cases found for difficulty={difficulty}, task_id={task_id}, solvable_only={solvable_only}")
    return case_ids



def filter_case_ids(selected_cases: list[str], case_ids_arg: str | None, cases: dict[str, dict[str, Any]]) -> list[str]:
    if not case_ids_arg:
        return selected_cases
    requested = [case_id.strip() for case_id in case_ids_arg.split(",") if case_id.strip()]
    if not requested:
        return selected_cases
    selected = set(selected_cases)
    missing = [case_id for case_id in requested if case_id not in cases]
    outside_filter = [case_id for case_id in requested if case_id in cases and case_id not in selected]
    if missing:
        raise ValueError(f"Unknown case ids: {missing}")
    if outside_filter:
        raise ValueError(f"Case ids do not match difficulty/task filters: {outside_filter}")
    return requested

def build_case_schedule(case_ids: list[str], episodes: int, seed: int, mode: str) -> list[str]:
    if episodes <= 0:
        return []
    rng = random.Random(seed)
    if mode == "random":
        return [rng.choice(case_ids) for _ in range(episodes)]
    if mode == "ordered":
        return [case_ids[index % len(case_ids)] for index in range(episodes)]
    if mode == "shuffled_cycle":
        schedule: list[str] = []
        while len(schedule) < episodes:
            batch = list(case_ids)
            rng.shuffle(batch)
            schedule.extend(batch)
        return schedule[:episodes]
    raise ValueError(f"Unsupported case schedule: {mode}")


def build_variant_config(variant: str, top_k: int) -> dict[str, Any]:
    config: dict[str, Any] = {
        "include_statistics": True,
        "ranking_mode": "score",
        "enable_node_merging": True,
        "learn_failure_preconditions": True,
        "propose_new_path": True,
        "top_k": top_k,
    }
    if variant == "no_exploration":
        config["propose_new_path"] = False
    elif variant == "no_statistics":
        config["include_statistics"] = False
    elif variant == "random_retrieval":
        config["ranking_mode"] = "random"
    elif variant == "no_node_merging":
        config["enable_node_merging"] = False
    elif variant == "no_failure_preconditions":
        config["learn_failure_preconditions"] = False
    elif variant == "no_graph_context":
        config["top_k"] = 0
    elif variant != "full":
        raise ValueError(f"Unsupported variant: {variant}")
    return config


def build_config(args: argparse.Namespace, run_id: str, case_ids: list[str], variant_config: dict[str, Any], llm_provider: str, llm_model: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "agent": args.agent,
        "variant": args.variant,
        "cases": args.cases,
        "rules": args.rules,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "task_id": args.task_id,
        "solvable_only": args.solvable_only,
        "rounds": args.rounds,
        "case_ids_filter": args.case_ids,
        "case_schedule": args.case_schedule,
        "case_ids": case_ids,
        "cross_task_mode": args.cross_task_mode,
        "retrieval_mode": args.retrieval_mode,
        "top_k": variant_config["top_k"],
        "requested_top_k": args.top_k,
        "token_budget": args.token_budget,
        "min_attempts_for_dormant": args.min_attempts_for_dormant,
        "dormant_success_threshold": args.dormant_success_threshold,
        "variant_config": variant_config,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "max_budget_rmb": args.max_budget_rmb,
        "llm_max_tokens": args.llm_max_tokens,
        "llm_retry_max_tokens": args.llm_retry_max_tokens,
        "llm_retries": args.llm_retries,
        "continue_after_env_failure": not args.stop_on_env_failure,
        "input_price_per_million_rmb": args.input_price_per_million_rmb,
        "output_price_per_million_rmb": args.output_price_per_million_rmb,
        "supported_agents": AGENTS,
        "warm_start_run": args.warm_start_run,
    }


def build_retriever(args: argparse.Namespace, store: JsonGraphStore, variant_config: dict[str, Any]) -> GraphRetriever:
    retriever_cls = GraphRetriever
    if args.retrieval_mode == "semantic_fallback":
        retriever_cls = SemanticFallbackGraphRetriever
    return retriever_cls(
        store,
        token_budget=args.token_budget,
        top_k=variant_config["top_k"],
        include_statistics=variant_config["include_statistics"],
        ranking_mode=variant_config["ranking_mode"],
        random_seed=args.seed,
        cross_task_mode=args.cross_task_mode,
    )


def validate_resume_config(existing: dict[str, Any], current: dict[str, Any], allow_mismatch: bool) -> None:
    if allow_mismatch:
        return
    mismatches = []
    for key in CONFIG_COMPARE_KEYS:
        if existing.get(key) != current.get(key):
            mismatches.append(f"{key}: existing={existing.get(key)!r}, current={current.get(key)!r}")
    if mismatches:
        raise ValueError("Config mismatch during resume. " + "; ".join(mismatches))


def count_completed_episodes(metrics_path: Path) -> int:
    if not metrics_path.exists():
        return 0
    count = 0
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


class BudgetTracker:
    def __init__(self, max_budget_rmb: float | None, input_price_per_million: float, output_price_per_million: float):
        self.max_budget_rmb = max_budget_rmb
        self.input_price_per_million = input_price_per_million
        self.output_price_per_million = output_price_per_million

    def estimate(self, usage: dict[str, Any]) -> float:
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("estimated_prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or usage.get("estimated_completion_tokens") or 0)
        return (prompt_tokens / 1_000_000 * self.input_price_per_million) + (completion_tokens / 1_000_000 * self.output_price_per_million)

    def should_stop(self, cost_rmb: float) -> bool:
        return self.max_budget_rmb is not None and cost_rmb >= self.max_budget_rmb


def build_agent(
    agent_type: str,
    plans: dict[str, list[str]],
    run_dir: Path,
    llm_provider: str,
    llm_model: str,
    variant_config: dict[str, Any],
    llm_max_tokens: int | None = None,
    llm_retry_max_tokens: int | None = None,
    llm_retries: int = 0,
):
    if agent_type == "scripted":
        return ScriptedAgent(plans)
    llm = build_llm_client(
        llm_provider,
        model=llm_model,
        max_tokens=llm_max_tokens,
        retry_max_tokens=llm_retry_max_tokens,
        retries=llm_retries,
    )
    if agent_type == "react":
        return ReActAgent(llm)
    if agent_type == "reflexion":
        return ReflexionAgent(llm, run_dir / "agent_memory_reflexion.jsonl")
    if agent_type == "vector_trajectory":
        return VectorTrajectoryAgent(llm, run_dir / "agent_memory_vector_trajectory.jsonl")
    if agent_type == "skill_library":
        return SkillLibraryAgent(llm, run_dir / "agent_memory_skill_library.jsonl")
    if agent_type == "graph":
        return ExperienceGraphAgent(llm, propose_new_path=variant_config["propose_new_path"])
    raise ValueError(f"Unsupported agent: {agent_type}")


if __name__ == "__main__":
    main()



