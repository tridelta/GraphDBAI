from __future__ import annotations

from pathlib import Path

import yaml

from experience_graph.envs.oracle import CaseOracle
from experience_graph.envs.textcraft import MyTextCraftAdapter, TextCraftAdapter


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "world_cases" / "textcraft_cases.yaml"
RULES = ROOT / "world_cases" / "textcraft_rules.yaml"
TASK_FAMILIES = ROOT / "world_cases" / "task_families"
WORLD_SUITE = ROOT / "world_cases" / "mytextcraft_world_v1.yaml"


def action_label(spec) -> str:
    return f"{spec.name}({', '.join(str(value) for value in spec.args.values())})" if spec.args else spec.name


def test_legacy_textcraft_adapter_alias():
    assert TextCraftAdapter is MyTextCraftAdapter


def test_world_cases_load():
    data = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    assert len(data["cases"]) == 12
    for case in data["cases"]:
        assert case["id"]
        assert "initial_state" in case
        assert "oracle" in case
        assert "expected_experience" in case


def test_rule_ids_unique():
    data = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    ids = [rule["id"] for rule in data["rules"]]
    assert len(ids) == len(set(ids))


def test_task_family_cases_load_as_active_suite():
    env = MyTextCraftAdapter(TASK_FAMILIES, RULES)
    assert len(env.cases) == 102
    assert set(env.actions_by_task) == {
        "cake",
        "cure_zombie_villager",
        "diamond_set",
        "enchant_pickaxe",
        "eye_of_ender",
        "fire_resistance_potion",
        "golden_apple",
        "golden_equipment_chain",
        "nether_portal",
    }
    for case in env.cases.values():
        assert case["task"]["id"]
        assert case["task"].get("success_conditions")
        assert all(isinstance(action, str) for action in case["oracle"].get("reference_plan", []))


def test_world_suite_manifest_loads_all_task_families():
    env = MyTextCraftAdapter(WORLD_SUITE, RULES)
    assert env.suite_metadata["name"] == "mytextcraft_world_v1_all_tasks"
    assert len(env.cases) == 102
    assert env.default_state["location"] == "base"
    assert "gold_ingot" in env.default_state["inventory"]
    assert "nearby_fortress" in env.default_state["environment"]


def test_world_suite_reset_uses_shared_state_schema():
    env = MyTextCraftAdapter(WORLD_SUITE, RULES)
    inventory_keys = set(env.default_state["inventory"])
    environment_keys = set(env.default_state["environment"])
    for case_id in env.cases:
        obs = env.reset(case_id=case_id)
        assert set(obs.state["inventory"]) == inventory_keys
        assert set(obs.state["environment"]) == environment_keys
        assert "location" in obs.state
        assert "ambiguous" in obs.state
        assert env.state is not None
        assert "hidden_facts" in env.state
        assert "hidden_facts" not in obs.state


def test_task_family_oracle_plans_match_expected_solvability():
    env = MyTextCraftAdapter(TASK_FAMILIES, RULES)
    oracle = CaseOracle(env)
    mismatches = []
    for case_id, case in env.cases.items():
        result = oracle.run_case(case_id)
        expected = bool(case["oracle"]["solvable"])
        if result.success != expected:
            mismatches.append((case_id, case["task"]["id"], expected, result.success, result.failure_reason))
    assert mismatches == []


def test_task_family_oracle_actions_are_declared():
    env = MyTextCraftAdapter(TASK_FAMILIES, RULES)
    missing = []
    for case_id, case in env.cases.items():
        declared = {action_label(spec) for spec in env.actions_by_task[case["task"]["id"]]}
        for action_text in case["oracle"].get("reference_plan", []) or []:
            if action_text.startswith("report_impossible"):
                continue
            if action_text not in declared:
                missing.append((case_id, case["task"]["id"], action_text))
    assert missing == []


def test_available_actions_are_state_relevant_for_chain_cases():
    env = MyTextCraftAdapter(TASK_FAMILIES, RULES)
    obs = env.reset(case_id="GEC_004")
    labels = {action_label(spec) for spec in env.available_actions(obs)}
    assert "inspect(chest)" not in labels
    assert "loot(chest, golden_apple)" not in labels
    assert "explore(mine)" in labels

    obs = env.reset(case_id="GEC_006")
    labels = {action_label(spec) for spec in env.available_actions(obs)}
    assert "inspect(chest)" in labels
