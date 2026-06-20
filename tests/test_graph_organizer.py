from __future__ import annotations

from experience_graph.core.experience import ExperienceBuilder
from experience_graph.core.models import Action, Condition, GraphNode, StepRecord
from experience_graph.graph.merge import NodeMerger
from experience_graph.graph.organizer import GraphOrganizer
from experience_graph.graph.store import JsonGraphStore


def test_exact_condition_merge():
    merger = NodeMerger()
    a = GraphNode("a", "a", "checkpoint", [Condition("inventory.diamond", ">=", 24)])
    b = GraphNode("b", "b", "checkpoint", [Condition("has_enough_diamond", "==", True)])
    decision = merger.decide(b, [a])
    assert decision.decision == "merged"


def test_conflict_condition_does_not_merge():
    merger = NodeMerger()
    a = GraphNode("a", "a", "checkpoint", [Condition("environment.village_has_armorer", "==", True)])
    b = GraphNode("b", "b", "checkpoint", [Condition("environment.village_has_armorer", "==", False)])
    decision = merger.decide(b, [a])
    assert decision.decision == "created"
    assert decision.method == "rule_conflict"


def test_failure_reason_becomes_precondition(tmp_path, textcraft_env):
    initial = textcraft_env.reset(case_id="TC_MINE_002")
    textcraft_env.step(Action.parse("move_to(mine)"))
    before = textcraft_env._observation()
    result = textcraft_env.step(Action.parse("mine(diamond)"))
    step = StepRecord(0, before, None, None, Action.parse("mine(diamond)"), result, result.observation)
    record = ExperienceBuilder().build("ep", "diamond_set", initial, [step], success=False)
    store = JsonGraphStore(tmp_path)
    GraphOrganizer(store).integrate(record)
    preconditions = [condition for edge in store.edges.values() for condition in edge.hard_preconditions]
    assert any(c.key == "tool.pickaxe_level" and c.operator == ">=" and c.value == 3 for c in preconditions)


def test_failed_edge_can_become_dormant(tmp_path, textcraft_env):
    store = JsonGraphStore(tmp_path)
    organizer = GraphOrganizer(store, min_attempts_for_dormant=1, dormant_success_threshold=0.5)
    initial = textcraft_env.reset(case_id="TC_MINE_002")
    textcraft_env.step(Action.parse("move_to(mine)"))
    before = textcraft_env._observation()
    result = textcraft_env.step(Action.parse("mine(diamond)"))
    step = StepRecord(0, before, None, None, Action.parse("mine(diamond)"), result, result.observation)
    record = ExperienceBuilder().build("ep", "diamond_set", initial, [step], success=False)
    organizer.integrate(record)
    assert any(edge.status == "dormant" for edge in store.edges.values())
