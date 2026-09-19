from __future__ import annotations

from types import SimpleNamespace

from experience_graph.core.models import Action, Condition, EdgeStats, GraphEdge, GraphNode, Observation, PathRecord, PathStats, TaskSpec
from experience_graph.graph.store import JsonGraphStore
from experience_graph.scripts.run_experiment import build_retriever
from experience_graph.semantic_retrieval import SemanticFallbackGraphRetriever


def test_semantic_fallback_boosts_paths_with_nearby_node_anchors(tmp_path):
    store = JsonGraphStore(tmp_path)
    store.upsert_node(
        GraphNode(
            id="mine_anchor",
            label="mine route with gold ore and apple crafting",
            node_type="checkpoint",
            required=[Condition("inventory.gold_ingot", ">=", 8)],
            metadata={"resource_tags": ["mine", "gold", "ore", "apple"]},
        )
    )
    store.upsert_node(
        GraphNode(
            id="village_anchor",
            label="village trade route with emeralds",
            node_type="checkpoint",
            required=[Condition("inventory.emerald", ">=", 40)],
            metadata={"resource_tags": ["village", "trade", "emerald"]},
        )
    )
    store.upsert_edge(
        GraphEdge(
            id="e_mine",
            from_node="mine_anchor",
            to_node="goal",
            action_template=Action.parse("mine(gold_ore)"),
            hard_preconditions=[Condition("environment.nearby_mine", "==", True)],
            stats=EdgeStats(attempts=2, successes=2),
        )
    )
    store.upsert_edge(
        GraphEdge(
            id="e_village",
            from_node="village_anchor",
            to_node="goal",
            action_template=Action.parse("trade(armorer, emerald, gold_ingot)"),
            hard_preconditions=[Condition("environment.nearby_village", "==", True)],
            stats=EdgeStats(attempts=2, successes=2),
        )
    )
    store.upsert_path(PathRecord("p_village", "golden_equipment_chain", "s", "goal", ["e_village"], PathStats(2, 2, 3, 3)))
    store.upsert_path(PathRecord("p_mine", "golden_equipment_chain", "s", "goal", ["e_mine"], PathStats(2, 2, 3, 3)))

    observation = Observation(
        state={
            "location": "mine",
            "inventory": {"apple": 1, "gold_ore": 3, "gold_ingot": 0},
            "environment": {"nearby_mine": True, "nearby_village": False},
        },
        case_id="GEC_TEST",
        step_count=0,
    )
    view = SemanticFallbackGraphRetriever(store, top_k=2, semantic_min_score=0.01).retrieve(
        TaskSpec("golden_equipment_chain"),
        observation,
        [],
    )

    assert view.candidate_paths[0].path_id == "p_mine"
    assert "Semantic fallback anchors" in view.summaries[-1]
    assert "Semantic fallback anchor" in view.candidate_paths[0].retrieval_reason


def test_run_experiment_builds_semantic_retriever_only_when_requested(tmp_path):
    store = JsonGraphStore(tmp_path)
    base_args = {
        "token_budget": 1000,
        "seed": 1,
        "cross_task_mode": "all_tasks",
    }
    variant_config = {
        "top_k": 5,
        "include_statistics": True,
        "ranking_mode": "score",
    }

    graph = build_retriever(SimpleNamespace(**base_args, retrieval_mode="graph"), store, variant_config)
    semantic = build_retriever(SimpleNamespace(**base_args, retrieval_mode="semantic_fallback"), store, variant_config)

    assert not isinstance(graph, SemanticFallbackGraphRetriever)
    assert isinstance(semantic, SemanticFallbackGraphRetriever)
