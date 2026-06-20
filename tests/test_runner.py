from __future__ import annotations

from experience_graph.agents.scripted import ScriptedAgent
from experience_graph.evaluation.logger import EvaluationLogger
from experience_graph.graph.organizer import GraphOrganizer
from experience_graph.graph.retriever import GraphRetriever
from experience_graph.graph.store import JsonGraphStore
from experience_graph.runners.episode_runner import EpisodeRunner


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
