from __future__ import annotations

from experience_graph.core.models import Action, Condition, EdgeStats, GraphEdge, PathRecord, PathStats, TaskSpec
from experience_graph.graph.retriever import GraphRetriever
from experience_graph.graph.store import JsonGraphStore


def test_retriever_applicability_states(tmp_path, textcraft_env):
    store = JsonGraphStore(tmp_path)
    active = GraphEdge(
        id="e1",
        from_node="a",
        to_node="b",
        action_template=Action.parse("trade(armorer, emerald, diamond_set)"),
        hard_preconditions=[Condition("environment.village_has_armorer", "==", True), Condition("inventory.emerald", ">=", 40)],
        stats=EdgeStats(attempts=2, successes=2),
    )
    unknown = GraphEdge(
        id="e2",
        from_node="a",
        to_node="c",
        action_template=Action.parse("inspect(village)"),
        hard_preconditions=[Condition("environment.village_has_armorer", "unknown", True)],
        stats=EdgeStats(attempts=1, successes=1),
    )
    blocked = GraphEdge(
        id="e3",
        from_node="a",
        to_node="d",
        action_template=Action.parse("mine(diamond)"),
        hard_preconditions=[Condition("tool.pickaxe_level", ">=", 3)],
        stats=EdgeStats(attempts=1, successes=0),
    )
    for edge in [active, unknown, blocked]:
        store.upsert_edge(edge)
    store.upsert_path(PathRecord("p1", "diamond_set", "s", "b", ["e1"], PathStats(2, 2, 2, 2)))
    store.upsert_path(PathRecord("p2", "diamond_set", "s", "c", ["e2"], PathStats(1, 1, 1, 1)))
    store.upsert_path(PathRecord("p3", "diamond_set", "s", "d", ["e3"], PathStats(1, 0, 1, 1)))
    obs = textcraft_env.reset(case_id="TC_TRADE_002")
    view = GraphRetriever(store, top_k=3, token_budget=20).retrieve(TaskSpec("diamond_set"), obs, [])
    states = {candidate.path_id: candidate.applicability for candidate in view.candidate_paths}
    assert states["p2"] == "needs_info"
    assert states["p3"] == "blocked"

    obs = textcraft_env.reset(case_id="TC_TRADE_001")
    view = GraphRetriever(store, top_k=1, token_budget=1).retrieve(TaskSpec("diamond_set"), obs, [])
    assert len(view.candidate_paths) == 1
    assert view.token_estimate <= 1
