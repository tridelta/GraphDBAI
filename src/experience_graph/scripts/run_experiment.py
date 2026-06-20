from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

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
from experience_graph.runners.episode_runner import EpisodeRunner


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run ExperienceGraph experiments.")
    parser.add_argument("--agent", choices=["scripted", "react", "graph"], default="scripted")
    parser.add_argument("--cases", default="world_cases/textcraft_cases.yaml")
    parser.add_argument("--rules", default="world_cases/textcraft_rules.yaml")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--run-dir", default=os.getenv("EXPERIENCE_GRAPH_RUN_DIR", "runs"))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = TextCraftAdapter(args.cases, args.rules)
    case_ids = list(env.cases.keys())[: args.episodes]
    plans = {case_id: env.cases[case_id]["oracle"].get("reference_plan", []) for case_id in env.cases}
    agent = build_agent(args.agent, plans)
    store = JsonGraphStore(run_dir)
    organizer = GraphOrganizer(store)
    retriever = GraphRetriever(store)
    logger = EvaluationLogger(run_dir)
    runner = EpisodeRunner(env, agent, organizer, retriever, logger, max_steps=args.max_steps)

    config = {
        "agent": args.agent,
        "cases": args.cases,
        "rules": args.rules,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "run_id": run_id,
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    for index, case_id in enumerate(case_ids):
        result = runner.run_episode(episode_id=f"ep_{index:04d}", case_id=case_id)
        status = "success" if result.metrics["success"] else f"failed:{result.metrics['failure_reason']}"
        print(f"{case_id}: {status} steps={result.metrics['steps']}")
    print(f"Run written to {run_dir}")


def build_agent(agent_type: str, plans: dict[str, list[str]]):
    if agent_type == "scripted":
        return ScriptedAgent(plans)
    llm = build_llm_client()
    if agent_type == "react":
        return ReActAgent(llm)
    if agent_type == "graph":
        return ExperienceGraphAgent(llm)
    raise ValueError(f"Unsupported agent: {agent_type}")


if __name__ == "__main__":
    main()

