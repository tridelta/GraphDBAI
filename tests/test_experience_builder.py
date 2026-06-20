from __future__ import annotations

from experience_graph.core.experience import ExperienceBuilder
from experience_graph.core.models import Action, Observation, StepRecord


def test_experience_builder_records_failure_and_discovery(textcraft_env):
    initial = textcraft_env.reset(case_id="TC_TRADE_002")
    before = initial
    result = textcraft_env.step(Action.parse("move_to(village)"))
    after = result.observation
    step0 = StepRecord(0, before, None, None, Action.parse("move_to(village)"), result, after)
    before = after
    result = textcraft_env.step(Action.parse("inspect(village)"))
    step1 = StepRecord(1, before, None, None, Action.parse("inspect(village)"), result, result.observation)
    record = ExperienceBuilder().build("ep", "diamond_set", initial, [step0, step1], success=False)
    assert record.trajectory[0].step_index == 0
    assert record.discovered_conditions
    assert record.discovered_conditions[0].key == "environment.village_has_armorer"


def test_experience_builder_failure_reason(textcraft_env):
    initial = textcraft_env.reset(case_id="TC_MINE_002")
    textcraft_env.step(Action.parse("move_to(mine)"))
    before = textcraft_env._observation()
    result = textcraft_env.step(Action.parse("mine(diamond)"))
    step = StepRecord(0, before, None, None, Action.parse("mine(diamond)"), result, result.observation)
    record = ExperienceBuilder().build("ep", "diamond_set", initial, [step], success=False)
    assert record.failure_reason == "missing_required_pickaxe"
