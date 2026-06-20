from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from experience_graph.core.models import (
    UNKNOWN,
    Action,
    ActionSpec,
    Condition,
    Observation,
    StepResult,
    TaskSpec,
    get_path,
    pickaxe_rank,
    set_path,
)


ARMOR_COSTS = {
    "diamond_helmet": 5,
    "diamond_chestplate": 8,
    "diamond_leggings": 7,
    "diamond_boots": 4,
}
ARMOR_PIECES = list(ARMOR_COSTS)


class TextCraftAdapter:
    def __init__(self, cases_path: str | Path, rules_path: str | Path):
        self.cases_path = Path(cases_path)
        self.rules_path = Path(rules_path)
        self.cases_data = yaml.safe_load(self.cases_path.read_text(encoding="utf-8"))
        self.rules_data = yaml.safe_load(self.rules_path.read_text(encoding="utf-8"))
        self.cases = {case["id"]: case for case in self.cases_data["cases"]}
        self.rules = self.rules_data["rules"]
        self.current_case: dict[str, Any] | None = None
        self.state: dict[str, Any] | None = None
        self.step_count = 0

    def reset(self, seed: int = 0, task: TaskSpec | None = None, case_id: str | None = None) -> Observation:
        del seed, task
        if case_id is None:
            case_id = next(iter(self.cases))
        if case_id not in self.cases:
            raise KeyError(f"Unknown TextCraft case: {case_id}")
        self.current_case = self.cases[case_id]
        self.state = copy.deepcopy(self.current_case["initial_state"])
        self.step_count = 0
        return self._observation()

    def available_actions(self, observation: Observation) -> list[ActionSpec]:
        del observation
        specs = [
            ActionSpec("craft", {"item": item}) for item in [*ARMOR_PIECES, "crafting_table"]
        ]
        specs.extend(
            [
                ActionSpec("gather", {"resource": "wood"}),
                ActionSpec("mine", {"resource": "diamond"}),
                ActionSpec("move_to", {"location": "mine"}),
                ActionSpec("move_to", {"location": "village"}),
                ActionSpec("inspect", {"target": "village"}),
                ActionSpec("inspect", {"target": "mine"}),
                ActionSpec("explore", {"target": "mine"}),
                ActionSpec("explore", {"target": "village"}),
                ActionSpec("trade", {"villager": "armorer", "offer": "emerald", "want": "diamond_set"}),
                ActionSpec("trade", {"villager": "armorer", "offer": "emerald", "want": "diamond_boots"}),
                ActionSpec("trade", {"villager": "armorer", "offer": "emerald", "want": "diamond_chestplate"}),
            ]
        )
        return specs

    def step(self, action: Action) -> StepResult:
        if self.state is None:
            raise RuntimeError("TextCraftAdapter.reset must be called before step.")
        before = copy.deepcopy(self.state)
        revealed: list[Condition] = []
        ok = True
        failure_reason: str | None = None
        state_delta: dict[str, Any] = {}

        if action.name == "craft":
            ok, failure_reason, state_delta = self._craft(action.args.get("item"))
        elif action.name == "gather":
            ok, failure_reason, state_delta = self._gather(action.args.get("resource"))
        elif action.name == "mine":
            ok, failure_reason, state_delta = self._mine(action.args.get("resource"))
        elif action.name == "move_to":
            ok, failure_reason, state_delta = self._move_to(action.args.get("location"))
        elif action.name == "inspect":
            ok, failure_reason, state_delta, revealed = self._inspect(action.args.get("target"))
        elif action.name == "explore":
            ok, failure_reason, state_delta, revealed = self._explore(action.args.get("target"))
        elif action.name == "trade":
            ok, failure_reason, state_delta = self._trade(action.args.get("want"))
        else:
            ok = False
            failure_reason = "unknown_action"

        self.step_count += 1
        done = self.is_success(self._observation(), TaskSpec(id="diamond_set"))
        reward = 1.0 if done else 0.0
        if not ok:
            self.state = before
        return StepResult(
            ok=ok,
            action=action,
            observation=self._observation(),
            reward=reward,
            cost=1.0,
            done=done,
            failure_reason=failure_reason,
            revealed_conditions=revealed,
            state_delta=state_delta,
        )

    def is_success(self, observation: Observation, task: TaskSpec) -> bool:
        del task
        return all(bool(observation.get(f"inventory.{piece}", False)) for piece in ARMOR_PIECES)

    def extract_conditions(self, observation: Observation) -> list[Condition]:
        state = observation.state
        conditions: list[Condition] = []
        diamond = get_path(state, "inventory.diamond", 0)
        emerald = get_path(state, "inventory.emerald", 0)
        if diamond >= 24:
            conditions.append(Condition("inventory.diamond", ">=", 24))
        elif diamond > 0:
            conditions.append(Condition("inventory.diamond", ">=", diamond))
        if emerald >= 40:
            conditions.append(Condition("inventory.emerald", ">=", 40))
        elif emerald >= 10:
            conditions.append(Condition("inventory.emerald", ">=", 10))
        if get_path(state, "inventory.crafting_table", False):
            conditions.append(Condition("inventory.crafting_table", "==", True))
        pickaxe_level = pickaxe_rank(get_path(state, "inventory.pickaxe", "none"))
        conditions.append(Condition("tool.pickaxe_level", "==", pickaxe_level))
        for key in [
            "environment.nearby_mine",
            "environment.nearby_village",
            "environment.village_has_armorer",
            "environment.mine_search_available",
            "environment.village_search_available",
            "environment.mine_diamond_capacity",
        ]:
            value = get_path(state, key, None)
            if value is not None:
                operator = "unknown" if value == UNKNOWN else "=="
                conditions.append(Condition(key, operator, value))
        return conditions

    def _observation(self) -> Observation:
        assert self.state is not None
        assert self.current_case is not None
        return Observation(copy.deepcopy(self.state), case_id=self.current_case["id"], step_count=self.step_count)

    def _craft(self, item: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if item == "crafting_table":
            if get_path(self.state, "inventory.wood", 0) < 4:
                return False, "insufficient_wood", {}
            self._add("inventory.wood", -4)
            set_path(self.state, "inventory.crafting_table", True)
            return True, None, {"inventory.wood": -4, "inventory.crafting_table": True}
        if item == "diamond_set":
            if get_path(self.state, "inventory.diamond", 0) < 24:
                return False, "insufficient_diamonds", {}
            if not get_path(self.state, "inventory.crafting_table", False):
                return False, "missing_crafting_table", {}
            self._add("inventory.diamond", -24)
            for piece in ARMOR_PIECES:
                set_path(self.state, f"inventory.{piece}", True)
            return True, None, {"inventory.diamond": -24, "inventory.diamond_set": True}
        if item not in ARMOR_COSTS:
            return False, "unknown_craft_item", {}
        if not get_path(self.state, "inventory.crafting_table", False):
            return False, "missing_crafting_table", {}
        cost = ARMOR_COSTS[item]
        if get_path(self.state, "inventory.diamond", 0) < cost:
            return False, "insufficient_diamonds", {}
        self._add("inventory.diamond", -cost)
        set_path(self.state, f"inventory.{item}", True)
        return True, None, {"inventory.diamond": -cost, f"inventory.{item}": True}

    def _gather(self, resource: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        if resource != "wood":
            return False, "unknown_resource", {}
        if get_path(self.state or {}, "environment.biome", "") not in {"forest", "plains"}:
            return False, "resource_unavailable", {}
        self._add("inventory.wood", 4)
        return True, None, {"inventory.wood": 4}

    def _mine(self, resource: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if resource != "diamond":
            return False, "unknown_resource", {}
        if get_path(self.state, "environment.nearby_mine", False) != True:
            return False, "mine_not_discovered", {}
        if pickaxe_rank(get_path(self.state, "inventory.pickaxe", "none")) < 3:
            return False, "missing_required_pickaxe", {}
        capacity = get_path(self.state, "environment.mine_diamond_capacity", None)
        mined = get_path(self.state, "environment.mine_diamonds_mined", 0)
        if isinstance(capacity, int) and mined >= capacity:
            return False, "mine_depleted", {}
        amount = 8
        if isinstance(capacity, int):
            amount = min(amount, capacity - mined)
        self._add("inventory.diamond", amount)
        self._add("environment.mine_diamonds_mined", amount)
        return True, None, {"inventory.diamond": amount}

    def _move_to(self, location: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if location == "mine" and get_path(self.state, "environment.nearby_mine", False) != True:
            return False, "mine_not_discovered", {}
        if location == "village" and get_path(self.state, "environment.nearby_village", False) != True:
            return False, "village_not_discovered", {}
        if location not in {"mine", "village", "base"}:
            return False, "unknown_location", {}
        self.state["location"] = location
        return True, None, {"location": location}

    def _inspect(self, target: str | None) -> tuple[bool, str | None, dict[str, Any], list[Condition]]:
        assert self.state is not None
        if target == "village":
            if get_path(self.state, "environment.nearby_village", False) != True:
                return False, "village_not_discovered", {}, []
            value = self._hidden("village_has_armorer", default=get_path(self.state, "environment.village_has_armorer", UNKNOWN))
            set_path(self.state, "environment.village_has_armorer", value)
            condition = Condition("environment.village_has_armorer", "==", value, source="inspect")
            return True, None, {"environment.village_has_armorer": value}, [condition]
        if target == "mine":
            if get_path(self.state, "environment.nearby_mine", False) != True:
                return False, "mine_not_discovered", {}, []
            value = self._hidden("mine_depth", default=get_path(self.state, "environment.mine_depth", "shallow"))
            set_path(self.state, "environment.mine_depth", value)
            condition = Condition("environment.mine_depth", "==", value, source="inspect")
            return True, None, {"environment.mine_depth": value}, [condition]
        return False, "unknown_inspect_target", {}, []

    def _explore(self, target: str | None) -> tuple[bool, str | None, dict[str, Any], list[Condition]]:
        assert self.state is not None
        if target == "mine":
            if not get_path(self.state, "environment.mine_search_available", False):
                return False, "mine_search_unavailable", {}, []
            value = self._hidden("nearby_mine", default=False)
            set_path(self.state, "environment.nearby_mine", value)
            revealed = [Condition("environment.nearby_mine", "==", value, source="explore")]
            depth = self._hidden("mine_depth", default=None)
            delta = {"environment.nearby_mine": value}
            if depth is not None:
                set_path(self.state, "environment.mine_depth", depth)
                delta["environment.mine_depth"] = depth
                revealed.append(Condition("environment.mine_depth", "==", depth, source="explore"))
            return True, None, delta, revealed
        if target == "village":
            if not get_path(self.state, "environment.village_search_available", False):
                return False, "village_search_unavailable", {}, []
            value = self._hidden("nearby_village", default=False)
            set_path(self.state, "environment.nearby_village", value)
            return True, None, {"environment.nearby_village": value}, [Condition("environment.nearby_village", "==", value, source="explore")]
        return False, "unknown_explore_target", {}, []

    def _trade(self, want: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if get_path(self.state, "location", "base") != "village":
            return False, "not_at_village", {}
        armorer = get_path(self.state, "environment.village_has_armorer", UNKNOWN)
        if armorer == UNKNOWN:
            return False, "armorer_unknown", {}
        if armorer is not True:
            return False, "no_armorer", {}
        if want == "diamond_set":
            if get_path(self.state, "inventory.emerald", 0) < 40:
                return False, "not_enough_emeralds", {}
            self._add("inventory.emerald", -40)
            for piece in ARMOR_PIECES:
                set_path(self.state, f"inventory.{piece}", True)
            return True, None, {"inventory.emerald": -40, "inventory.diamond_set": True}
        if want in ARMOR_PIECES:
            if get_path(self.state, "inventory.emerald", 0) < 10:
                return False, "not_enough_emeralds", {}
            self._add("inventory.emerald", -10)
            set_path(self.state, f"inventory.{want}", True)
            return True, None, {"inventory.emerald": -10, f"inventory.{want}": True}
        if want == "diamond_armor_piece":
            missing = [piece for piece in ARMOR_PIECES if not get_path(self.state, f"inventory.{piece}", False)]
            if not missing:
                return False, "no_missing_armor_piece", {}
            return self._trade(missing[0])
        return False, "unknown_trade_item", {}

    def _add(self, dotted_key: str, amount: int) -> None:
        assert self.state is not None
        current = get_path(self.state, dotted_key, 0)
        set_path(self.state, dotted_key, current + amount)

    def _hidden(self, key: str, default: Any) -> Any:
        assert self.current_case is not None
        return self.current_case.get("hidden_facts", {}).get(key, self.state.get("hidden_facts", {}).get(key, default))

