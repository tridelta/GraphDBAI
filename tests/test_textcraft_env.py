from __future__ import annotations

from experience_graph.core.models import Action, TaskSpec


def run_plan(env, case_id: str):
    case = env.cases[case_id]
    obs = env.reset(case_id=case_id)
    results = []
    for action_text in case["oracle"].get("reference_plan", []):
        result = env.step(Action.parse(action_text))
        results.append(result)
        obs = result.observation
        if not result.ok:
            break
    return obs, results


def test_crafting_success(textcraft_env):
    obs, _ = run_plan(textcraft_env, "TC_CRAFT_001")
    assert textcraft_env.is_success(obs, TaskSpec("diamond_set"))


def test_missing_table_path_can_create_table(textcraft_env):
    obs, _ = run_plan(textcraft_env, "TC_CRAFT_002")
    assert textcraft_env.is_success(obs, TaskSpec("diamond_set"))
    assert obs.get("inventory.crafting_table") is True


def test_insufficient_diamonds_failure(textcraft_env):
    textcraft_env.reset(case_id="TC_CRAFT_003")
    result = textcraft_env.step(Action.parse("craft(diamond_set)"))
    assert not result.ok
    assert result.failure_reason == "insufficient_diamonds"


def test_missing_pickaxe_failure(textcraft_env):
    textcraft_env.reset(case_id="TC_MINE_002")
    textcraft_env.step(Action.parse("move_to(mine)"))
    result = textcraft_env.step(Action.parse("mine(diamond)"))
    assert not result.ok
    assert result.failure_reason == "missing_required_pickaxe"


def test_inspect_reveals_armorer(textcraft_env):
    textcraft_env.reset(case_id="TC_TRADE_002")
    textcraft_env.step(Action.parse("move_to(village)"))
    result = textcraft_env.step(Action.parse("inspect(village)"))
    assert result.ok
    assert result.revealed_conditions[0].key == "environment.village_has_armorer"
    assert result.revealed_conditions[0].value is True


def test_impossible_case_remains_unsolved(textcraft_env):
    obs, _ = run_plan(textcraft_env, "TC_IMPOSSIBLE_001")
    assert not textcraft_env.is_success(obs, TaskSpec("diamond_set"))


def test_observation_hides_hidden_facts(textcraft_env):
    obs = textcraft_env.reset(case_id="TC_TRADE_002")
    state = obs.to_dict()["state"]
    assert "hidden_facts" not in state
    assert state["environment"]["village_has_armorer"] == "unknown"
    assert state["ambiguous"] == {"village_has_armorer": "unknown"}


def test_inspect_clears_revealed_ambiguous_flag(textcraft_env):
    textcraft_env.reset(case_id="TC_TRADE_003")
    textcraft_env.step(Action.parse("move_to(village)"))
    result = textcraft_env.step(Action.parse("inspect(village)"))
    state = result.observation.to_dict()["state"]
    assert state["environment"]["village_has_armorer"] is False
    assert state["ambiguous"] == {}
    assert "hidden_facts" not in state
