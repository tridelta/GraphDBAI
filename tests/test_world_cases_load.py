from __future__ import annotations

from pathlib import Path

import yaml

from experience_graph.envs.oracle import CaseOracle
from experience_graph.envs.textcraft import MyTextCraftAdapter, TextCraftAdapter


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "world_cases" / "textcraft_cases.yaml"
RULES = ROOT / "world_cases" / "textcraft_rules.yaml"
TASK_FAMILIES = ROOT / "world_cases" / "task_families"


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
    assert len(env.cases) == 96
    assert set(env.actions_by_task) == {
        "cake",
        "cure_zombie_villager",
        "diamond_set",
        "enchant_pickaxe",
        "eye_of_ender",
        "fire_resistance_potion",
        "golden_apple",
        "nether_portal",
    }
    for case in env.cases.values():
        assert case["task"]["id"]
        assert case["task"].get("success_conditions")
        assert all(isinstance(action, str) for action in case["oracle"].get("reference_plan", []))


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
