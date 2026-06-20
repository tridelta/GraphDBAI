from __future__ import annotations

from pathlib import Path

import pytest

from experience_graph.envs.textcraft import TextCraftAdapter


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "world_cases" / "textcraft_cases.yaml"
RULES = ROOT / "world_cases" / "textcraft_rules.yaml"


@pytest.fixture
def textcraft_env() -> TextCraftAdapter:
    return TextCraftAdapter(CASES, RULES)
