from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from experience_graph.agents.experience_graph_agent import ExperienceGraphAgent
from experience_graph.agents.react import ReActAgent
from experience_graph.agents.scripted import ScriptedAgent
from experience_graph.envs.textcraft import TextCraftAdapter
from experience_graph.evaluation.logger import EvaluationLogger
from experience_graph.graph.organizer import GraphOrganizer
from experience_graph.graph.retriever import GraphRetriever
from experience_graph.graph.store import JsonGraphStore
from experience_graph.llm.client import build_llm_client
from experience_graph.runners.episode_runner import EpisodeRunner, RunContext


VARIANTS = {
    "full",
    "no_statistics",
    "random_retrieval",
    "no_node_merging",
    "no_failure_preconditions",
    "no_graph_context",
}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run ExperienceGraph experiments.")
    parser.add_argument("--agent", choices=["scripted", "react", "graph"], default="scripted")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="full")
    parser.add_argument("--cases", default="world_cases/textcraft_cases.yaml")
    parser.add_argument("--rules", default="world_cases/textcraft_rules.yaml")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--difficulty", choices=["all", "easy", "medium", "hard", "impossible"], default="all")
    parser.add_argument("--task-id", default="diamond_set")
    parser.add_argument("--case-schedule", choices=["ordered", "shuffled_cycle", "random"], default="shuffled_cycle")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=1000)
    parser.add_argument("--min-attempts-for-dormant", type=int, default=5)
    parser.add_argument("--dormant-success-threshold", type=float, default=0.1)
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--run-dir", default=os.getenv("EXPERIENCE_GRAPH_RUN_DIR", "runs"))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or build_run_id(args)
    run_dir = Path(args.run_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = TextCraftAdapter(args.cases, args.rules)
    selected_cases = filter_cases(env.cases, args.difficulty, args.task_id)
    case_ids = build_case_schedule(selected_cases, args.episodes, args.seed, args.case_schedule)
    plans = {case_id: env.cases[case_id]["oracle"].get("reference_plan", []) for case_id in env.cases}
    agent = build_agent(args.agent, plans, args.llm_provider)
    effective_llm_provider = "none" if args.agent == "scripted" else (args.llm_provider or os.getenv("EXPERIENCE_GRAPH_LLM_PROVIDER", "openai"))
    store = JsonGraphStore(run_dir)
    variant_config = build_variant_config(args.variant, args.top_k)
    organizer = GraphOrganizer(
        store,
        min_attempts_for_dormant=args.min_attempts_for_dormant,
        dormant_success_threshold=args.dormant_success_threshold,
        enable_node_merging=variant_config["enable_node_merging"],
        learn_failure_preconditions=variant_config["learn_failure_preconditions"],
    )
    retriever = GraphRetriever(
        store,
        token_budget=args.token_budget,
        top_k=variant_config["top_k"],
        include_statistics=variant_config["include_statistics"],
        ranking_mode=variant_config["ranking_mode"],
        random_seed=args.seed,
    )
    logger = EvaluationLogger(run_dir)
    run_context = RunContext(
        run_id=run_id,
        agent=args.agent,
        variant=args.variant,
        seed=args.seed,
        metadata={
            "task_id": args.task_id,
            "case_schedule_mode": args.case_schedule,
            "llm_provider": effective_llm_provider,
        },
    )
    runner = EpisodeRunner(env, agent, organizer, retriever, logger, max_steps=args.max_steps, run_context=run_context)

    config = {
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
        "case_schedule": args.case_schedule,
        "case_ids": case_ids,
        "top_k": variant_config["top_k"],
        "requested_top_k": args.top_k,
        "token_budget": args.token_budget,
        "min_attempts_for_dormant": args.min_attempts_for_dormant,
        "dormant_success_threshold": args.dormant_success_threshold,
        "variant_config": variant_config,
        "llm_provider": effective_llm_provider,
        "supported_agents": ["scripted", "react", "graph"],
        "planned_baselines_not_implemented": ["reflexion", "vector_trajectory", "skill_library"],
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    for index, case_id in enumerate(case_ids):
        episode_seed = args.seed + index
        result = runner.run_episode(
            episode_id=f"ep_{index:04d}",
            case_id=case_id,
            seed=episode_seed,
            context={"episode_index": index},
        )
        status = "success" if result.metrics["success"] else f"failed:{result.metrics['failure_reason']}"
        print(f"{case_id}: {status} steps={result.metrics['steps']} seed={episode_seed}")
    print(f"Run written to {run_dir}")


def build_run_id(args: argparse.Namespace) -> str:
    timestamp = time.strftime("run_%Y%m%d_%H%M%S")
    return f"{timestamp}_{args.agent}_{args.variant}_seed{args.seed}"


def filter_cases(cases: dict[str, dict[str, Any]], difficulty: str, task_id: str) -> list[str]:
    case_ids = []
    for case_id, case in cases.items():
        if difficulty != "all" and case.get("difficulty") != difficulty:
            continue
        if case.get("task", {}).get("id") != task_id:
            continue
        case_ids.append(case_id)
    if not case_ids:
        raise ValueError(f"No cases found for difficulty={difficulty}, task_id={task_id}")
    return case_ids


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
        "top_k": top_k,
    }
    if variant == "no_statistics":
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


def build_agent(agent_type: str, plans: dict[str, list[str]], llm_provider: str | None = None):
    if agent_type == "scripted":
        return ScriptedAgent(plans)
    llm = build_llm_client(llm_provider)
    if agent_type == "react":
        return ReActAgent(llm)
    if agent_type == "graph":
        return ExperienceGraphAgent(llm)
    raise ValueError(f"Unsupported agent: {agent_type}")


if __name__ == "__main__":
    main()


