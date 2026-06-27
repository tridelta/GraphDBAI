from __future__ import annotations

import json
import pytest
from experience_graph.agents.react import ReActAgent
from experience_graph.agents.scripted import ScriptedAgent
from experience_graph.evaluation.logger import EvaluationLogger
from experience_graph.graph.organizer import GraphOrganizer
from experience_graph.graph.retriever import GraphRetriever
from experience_graph.graph.store import JsonGraphStore
from experience_graph.llm.client import FakeLLMClient
from experience_graph.runners.episode_runner import EpisodeRunner
from experience_graph.scripts.run_experiment import build_case_schedule, filter_case_ids, filter_cases, prune_incomplete_episode_logs


def test_scripted_runner_writes_jsonl(tmp_path, textcraft_env):
    plans = {case_id: case["oracle"].get("reference_plan", []) for case_id, case in textcraft_env.cases.items()}
    store = JsonGraphStore(tmp_path)
    runner = EpisodeRunner(
        textcraft_env,
        ScriptedAgent(plans),
        GraphOrganizer(store),
        GraphRetriever(store),
        EvaluationLogger(tmp_path),
        max_steps=30,
    )
    result = runner.run_episode("ep_0001", "TC_CRAFT_001")
    assert result.metrics["success"] is True
    assert result.metrics["steps"] == 4
    assert (tmp_path / "episodes.jsonl").exists()
    assert (tmp_path / "metrics.jsonl").exists()
    assert "graph_nodes" in result.metrics


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_runner(tmp_path, textcraft_env, agent, max_steps=3):
    store = JsonGraphStore(tmp_path)
    return EpisodeRunner(
        textcraft_env,
        agent,
        GraphOrganizer(store),
        GraphRetriever(store),
        EvaluationLogger(tmp_path),
        max_steps=max_steps,
    )


def test_runner_keeps_invalid_llm_output_failure(tmp_path, textcraft_env):
    runner = make_runner(tmp_path, textcraft_env, ReActAgent(FakeLLMClient(response={})), max_steps=12)
    result = runner.run_episode("ep_invalid", "TC_TRADE_003")

    assert result.metrics["success"] is False
    assert result.metrics["failure_reason"] == "invalid_llm_output"
    assert result.metrics["oracle_failure_reason"] == "no_armorer_and_no_mine"
    steps = read_jsonl(tmp_path / "steps.jsonl")
    assert steps[0]["failure_reason"] == "invalid_llm_output"
    assert steps[0]["prompt_diagnostics"]["prompt_hidden_facts"] is False


def test_runner_logs_repeated_actions_and_episode_top_fields(tmp_path, textcraft_env):
    llm = FakeLLMClient(response={"next_action": {"name": "inspect", "args": {"target": "village"}}, "reason": "fake"})
    runner = make_runner(tmp_path, textcraft_env, ReActAgent(llm), max_steps=3)
    result = runner.run_episode("ep_repeat", "TC_TRADE_003", context={"episode_index": 0})

    assert result.metrics["failure_reason"] == "step_budget_exhausted"
    steps = read_jsonl(tmp_path / "steps.jsonl")
    assert [row["repeated_action_count"] for row in steps] == [1, 2, 3]
    assert steps[-1]["repeated_action"] is True
    assert all(not row["prompt_diagnostics"]["prompt_hidden_facts"] for row in steps)
    episodes = read_jsonl(tmp_path / "episodes.jsonl")
    assert episodes[0]["case_id"] == "TC_TRADE_003"
    assert "graph_nodes" in episodes[0]


def test_prune_incomplete_episode_logs_removes_rows_after_completed(tmp_path):
    for name in ["steps.jsonl", "experience_views.jsonl", "budget_progress.jsonl"]:
        path = tmp_path / name
        path.write_text(
            "\n".join(
                [
                    json.dumps({"episode_index": 0, "step_index": 0}),
                    json.dumps({"episode_index": 1, "step_index": 0}),
                ]
            ) + "\n",
            encoding="utf-8",
        )

    pruned = prune_incomplete_episode_logs(tmp_path, completed_episodes=1)

    assert pruned == {"steps.jsonl": 1, "experience_views.jsonl": 1, "budget_progress.jsonl": 1}
    assert read_jsonl(tmp_path / "steps.jsonl") == [{"episode_index": 0, "step_index": 0}]


def test_filter_case_ids_preserves_requested_order():
    cases = {"A": {}, "B": {}, "C": {}}
    selected = filter_case_ids(["A", "B", "C"], "C,A", cases)
    assert selected == ["C", "A"]


def test_filter_cases_can_exclude_unsolvable_cases():
    cases = {
        "A": {"difficulty": "easy", "task": {"id": "x"}, "oracle": {"solvable": True}},
        "B": {"difficulty": "hard", "task": {"id": "x"}, "oracle": {"solvable": False}},
        "C": {"difficulty": "hard", "task": {"id": "y"}, "oracle": {"solvable": True}},
    }

    assert filter_cases(cases, "all", "all", solvable_only=True) == ["A", "C"]
    assert filter_cases(cases, "hard", "x", solvable_only=False) == ["B"]
    with pytest.raises(ValueError):
        filter_cases(cases, "hard", "x", solvable_only=True)


def test_ordered_round_schedule_repeats_full_case_set():
    assert build_case_schedule(["A", "B", "C"], episodes=6, seed=1, mode="ordered") == ["A", "B", "C", "A", "B", "C"]


def test_scripted_runner_reuses_plan_for_repeated_case(tmp_path, textcraft_env):
    plans = {case_id: case["oracle"].get("reference_plan", []) for case_id, case in textcraft_env.cases.items()}
    store = JsonGraphStore(tmp_path)
    runner = EpisodeRunner(
        textcraft_env,
        ScriptedAgent(plans),
        GraphOrganizer(store),
        GraphRetriever(store),
        EvaluationLogger(tmp_path),
        max_steps=30,
    )
    first = runner.run_episode("ep_0001", "TC_CRAFT_001")
    second = runner.run_episode("ep_0002", "TC_CRAFT_001")
    assert first.metrics["success"] is True
    assert second.metrics["success"] is True
