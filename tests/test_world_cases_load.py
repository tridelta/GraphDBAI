from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "world_cases" / "textcraft_cases.yaml"
RULES = ROOT / "world_cases" / "textcraft_rules.yaml"


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


