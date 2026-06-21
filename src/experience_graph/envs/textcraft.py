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
    compare_value,
    get_path,
    pickaxe_rank,
    set_path,
    visible_state,
)


ARMOR_COSTS = {
    "diamond_helmet": 5,
    "diamond_chestplate": 8,
    "diamond_leggings": 7,
    "diamond_boots": 4,
}
ARMOR_PIECES = list(ARMOR_COSTS)


TASK_SUCCESS_CONDITIONS = {
    "diamond_set": {
        "inventory.diamond_helmet": True,
        "inventory.diamond_chestplate": True,
        "inventory.diamond_leggings": True,
        "inventory.diamond_boots": True,
    },
    "golden_apple": {"inventory.golden_apple": True},
    "nether_portal": {"environment.portal_lit": True},
    "enchant_pickaxe": {"inventory.enchanted_pickaxe": True},
    "fire_resistance_potion": {"inventory.fire_resistance_potion": True},
    "cure_zombie_villager": {"environment.zombie_villager_cured": True},
    "cake": {"inventory.cake": True},
    "eye_of_ender": {"inventory.eye_of_ender": True},
}

LOCATION_REQUIREMENTS = {
    "base": None,
    "mine": "environment.nearby_mine",
    "village": "environment.nearby_village",
    "orchard": "environment.nearby_orchard",
    "dungeon": "environment.nearby_dungeon",
    "ruined_portal": "environment.nearby_ruined_portal",
    "lava_pool": "environment.nearby_lava_pool",
    "fortress": "environment.nearby_fortress",
    "bastion": "environment.nearby_bastion",
    "basalt_delta": "environment.nearby_magma_cube",
    "mob_area": "environment.nearby_mob_area",
    "river": "environment.nearby_river",
    "pasture": "environment.nearby_cow",
    "igloo": "environment.nearby_igloo",
}

EXPLORE_TARGETS = {
    "mine": ("environment.mine_search_available", "environment.nearby_mine"),
    "village": ("environment.village_search_available", "environment.nearby_village"),
    "dungeon": ("environment.dungeon_search_available", "environment.nearby_dungeon"),
    "ruined_portal": ("environment.portal_search_available", "environment.nearby_ruined_portal"),
    "lava_pool": ("environment.lava_search_available", "environment.nearby_lava_pool"),
    "fortress": ("environment.nether_search_available", "environment.nearby_fortress"),
    "igloo": ("environment.snow_biome_search_available", "environment.nearby_igloo"),
    "pasture": ("environment.pasture_search_available", "environment.nearby_cow"),
    "chicken": ("environment.chicken_search_available", "environment.nearby_chicken"),
}

INSPECT_TARGET_KEYS = {
    "village": [
        "environment.village_has_armorer",
        "environment.village_has_librarian",
        "environment.village_has_cleric",
        "environment.village_has_farmer",
        "environment.zombie_villager_present",
    ],
    "chest": [
        "environment.chest_has_golden_apple",
        "environment.chest_has_fire_resistance_potion",
    ],
    "basement": [
        "environment.basement_has_cure_supplies",
        "environment.igloo_has_weakness_potion",
    ],
    "farmer": ["environment.farmer_has_wheat_trade"],
    "portal_frame": [
        "environment.ruined_portal_missing_obsidian",
        "environment.ruined_portal_repairable",
    ],
    "fortress": ["environment.fortress_has_nether_wart"],
    "mine": ["environment.mine_depth"],
}


class TextCraftAdapter:
    def __init__(self, cases_path: str | Path, rules_path: str | Path):
        self.cases_path = Path(cases_path)
        self.rules_path = Path(rules_path)
        self.rules_data = yaml.safe_load(self.rules_path.read_text(encoding="utf-8"))
        self.cases_data = self._load_cases(self.cases_path)
        self.cases = {case["id"]: case for case in self.cases_data["cases"]}
        self.rules = self.rules_data["rules"]
        self.actions_by_task = self._build_actions_by_task()
        self.current_case: dict[str, Any] | None = None
        self.state: dict[str, Any] | None = None
        self.step_count = 0

    def _load_cases(self, path: Path) -> dict[str, Any]:
        sources = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
        cases: list[dict[str, Any]] = []
        for source in sources:
            data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
            if "cases" not in data:
                raise ValueError(f"TextCraft cases file has no cases list: {source}")
            family = data.get("family")
            for raw_case in data["cases"]:
                case = copy.deepcopy(raw_case)
                if family:
                    case.setdefault("family", family)
                task = dict(case.get("task") or {})
                task_id = task.get("id") or family
                if not task_id:
                    raise ValueError(f"TextCraft case has no task id: {case.get('id')} in {source}")
                task["id"] = task_id
                rule_task = (self.rules_data.get("tasks", {}) or {}).get(task_id, {})
                task.setdefault("success_conditions", rule_task.get("success_conditions") or TASK_SUCCESS_CONDITIONS.get(task_id, {}))
                case["task"] = task
                for key in ["visible_manual_refs", "required_engine_actions", "goal"]:
                    if key in data:
                        case.setdefault(key, copy.deepcopy(data[key]))
                case.setdefault("source_file", str(source))
                cases.append(case)
        ids = [case["id"] for case in cases]
        if len(ids) != len(set(ids)):
            duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
            raise ValueError(f"Duplicate TextCraft case ids: {duplicates}")
        return {"cases": cases}

    def _build_actions_by_task(self) -> dict[str, list[ActionSpec]]:
        by_task: dict[str, dict[str, ActionSpec]] = {}
        for case in self.cases_data["cases"]:
            task_id = case["task"]["id"]
            templates = by_task.setdefault(task_id, {})
            for action_text in case.get("oracle", {}).get("reference_plan", []) or []:
                action = Action.parse(action_text)
                templates[action.label()] = ActionSpec(action.name, dict(action.args))
        return {task_id: list(templates.values()) for task_id, templates in by_task.items()}

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
        task_id = self._task_id_for_observation(observation)
        specs = list(self.actions_by_task.get(task_id, []))
        if not any(spec.name == "report_impossible" for spec in specs):
            specs.append(ActionSpec("report_impossible", {"reason": "no_viable_plan"}))
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
            ok, failure_reason, state_delta = self._trade(action.args.get("villager"), action.args.get("want"))
        elif action.name == "smelt":
            ok, failure_reason, state_delta = self._smelt(action.args.get("arg0") or action.args.get("resource"))
        elif action.name == "loot":
            ok, failure_reason, state_delta = self._loot(action.args.get("arg0"), action.args.get("arg1"))
        elif action.name == "brew":
            ok, failure_reason, state_delta = self._brew(action.args.get("arg0"))
        elif action.name == "fill":
            ok, failure_reason, state_delta = self._fill(action.args.get("arg0"))
        elif action.name == "place":
            ok, failure_reason, state_delta = self._place(action.args.get("arg0"))
        elif action.name == "cast":
            ok, failure_reason, state_delta = self._cast(action.args.get("arg0"))
        elif action.name == "ignite":
            ok, failure_reason, state_delta = self._ignite(action.args.get("arg0"))
        elif action.name == "enchant":
            ok, failure_reason, state_delta = self._enchant(action.args.get("arg0"))
        elif action.name == "fight":
            ok, failure_reason, state_delta = self._fight(action.args.get("arg0"))
        elif action.name == "use_anvil":
            ok, failure_reason, state_delta = self._use_anvil(action.args.get("arg0"), action.args.get("arg1"))
        elif action.name == "milk":
            ok, failure_reason, state_delta = self._milk(action.args.get("arg0"))
        elif action.name == "plant":
            ok, failure_reason, state_delta = self._plant(action.args.get("arg0"))
        elif action.name == "harvest":
            ok, failure_reason, state_delta = self._harvest(action.args.get("arg0"))
        elif action.name == "throw":
            ok, failure_reason, state_delta = self._throw(action.args.get("arg0"))
        elif action.name == "use":
            ok, failure_reason, state_delta = self._use(action.args.get("arg0"), action.args.get("arg1"))
        elif action.name == "wait":
            ok, failure_reason, state_delta = self._wait(action.args.get("arg0"))
        elif action.name == "report_impossible":
            ok, failure_reason, state_delta = self._report_impossible(action.args.get("reason"))
        else:
            ok = False
            failure_reason = "unknown_action"

        self.step_count += 1
        done = self.is_success(self._observation(), TaskSpec(id=self._current_task_id())) or (ok and action.name == "report_impossible")
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
        task_id = task.id or self._task_id_for_observation(observation)
        conditions = task.success_conditions or self._success_conditions(task_id)
        return all(self._matches_expected(observation.state, key, expected) for key, expected in conditions.items())

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
        inventory = state.get("inventory", {})
        if isinstance(inventory, dict):
            for key, value in inventory.items():
                dotted = f"inventory.{key}"
                if isinstance(value, bool) and value:
                    conditions.append(Condition(dotted, "==", True))
                elif isinstance(value, int) and value > 0:
                    conditions.append(Condition(dotted, ">=", value))
        pickaxe_level = pickaxe_rank(get_path(state, "inventory.pickaxe", "none"))
        conditions.append(Condition("tool.pickaxe_level", "==", pickaxe_level))
        environment = state.get("environment", {})
        if isinstance(environment, dict):
            for key, value in environment.items():
                if value is None:
                    continue
                operator = "unknown" if value == UNKNOWN else "=="
                conditions.append(Condition(f"environment.{key}", operator, value))
        deduped: dict[str, Condition] = {}
        for condition in conditions:
            deduped.setdefault(condition.signature(), condition)
        return list(deduped.values())

    def _observation(self) -> Observation:
        assert self.state is not None
        assert self.current_case is not None
        return Observation(visible_state(self.state), case_id=self.current_case["id"], step_count=self.step_count)

    def _task_id_for_observation(self, observation: Observation) -> str:
        if observation.case_id and observation.case_id in self.cases:
            return self.cases[observation.case_id]["task"]["id"]
        return self._current_task_id()

    def _current_task_id(self) -> str:
        if self.current_case is not None:
            return self.current_case["task"]["id"]
        return next(iter(self.actions_by_task), "diamond_set")

    def _success_conditions(self, task_id: str) -> dict[str, Any]:
        if self.current_case is not None and self.current_case["task"].get("id") == task_id:
            return self.current_case["task"].get("success_conditions") or TASK_SUCCESS_CONDITIONS.get(task_id, {})
        return TASK_SUCCESS_CONDITIONS.get(task_id, {})

    def _matches_expected(self, state: dict[str, Any], key: str, expected: Any) -> bool:
        actual = get_path(state, key, UNKNOWN)
        if isinstance(expected, str):
            for operator in [">=", "<=", ">", "<"]:
                if expected.startswith(operator):
                    raw = expected[len(operator):]
                    try:
                        value: Any = int(raw)
                    except ValueError:
                        value = raw
                    return compare_value(actual, operator, value)
        return actual == expected


    def _clear_ambiguous(self, dotted_key: str) -> None:
        assert self.state is not None
        ambiguous = self.state.get("ambiguous")
        if not isinstance(ambiguous, dict):
            return
        leaf_key = dotted_key.split(".", 1)[1] if dotted_key.startswith("environment.") else dotted_key
        ambiguous.pop(dotted_key, None)
        ambiguous.pop(leaf_key, None)

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
        if item in ARMOR_COSTS:
            if not get_path(self.state, "inventory.crafting_table", False):
                return False, "missing_crafting_table", {}
            cost = ARMOR_COSTS[item]
            if get_path(self.state, "inventory.diamond", 0) < cost:
                return False, "insufficient_diamonds", {}
            self._add("inventory.diamond", -cost)
            set_path(self.state, f"inventory.{item}", True)
            return True, None, {"inventory.diamond": -cost, f"inventory.{item}": True}
        if item == "golden_apple":
            if get_path(self.state, "inventory.apple", 0) < 1 or get_path(self.state, "inventory.gold_ingot", 0) < 8:
                return False, "missing_golden_apple_ingredients", {}
            self._add("inventory.apple", -1)
            self._add("inventory.gold_ingot", -8)
            set_path(self.state, "inventory.golden_apple", True)
            return True, None, {"inventory.apple": -1, "inventory.gold_ingot": -8, "inventory.golden_apple": True}
        if item == "flint_and_steel":
            if get_path(self.state, "inventory.flint", 0) < 1 or get_path(self.state, "inventory.iron_ingot", 0) < 1:
                return False, "missing_flint_and_iron", {}
            self._add("inventory.flint", -1)
            self._add("inventory.iron_ingot", -1)
            set_path(self.state, "inventory.flint_and_steel", True)
            return True, None, {"inventory.flint": -1, "inventory.iron_ingot": -1, "inventory.flint_and_steel": True}
        if item == "blaze_powder":
            if get_path(self.state, "inventory.blaze_rod", 0) < 1:
                return False, "missing_blaze_rod", {}
            self._add("inventory.blaze_rod", -1)
            self._add("inventory.blaze_powder", 2)
            return True, None, {"inventory.blaze_rod": -1, "inventory.blaze_powder": 2}
        if item == "magma_cream":
            if get_path(self.state, "inventory.slimeball", 0) < 1 or get_path(self.state, "inventory.blaze_powder", 0) < 1:
                return False, "missing_magma_cream_ingredients", {}
            self._add("inventory.slimeball", -1)
            self._add("inventory.blaze_powder", -1)
            self._add("inventory.magma_cream", 1)
            return True, None, {"inventory.slimeball": -1, "inventory.blaze_powder": -1, "inventory.magma_cream": 1}
        if item == "sugar":
            if get_path(self.state, "inventory.sugar_cane", 0) < 1:
                return False, "missing_sugar_cane", {}
            self._add("inventory.sugar_cane", -1)
            self._add("inventory.sugar", 2)
            return True, None, {"inventory.sugar_cane": -1, "inventory.sugar": 2}
        if item == "cake":
            if get_path(self.state, "inventory.milk_bucket", 0) < 3 or get_path(self.state, "inventory.sugar", 0) < 2 or get_path(self.state, "inventory.wheat", 0) < 3 or get_path(self.state, "inventory.egg", 0) < 1:
                return False, "missing_cake_ingredients", {}
            self._add("inventory.milk_bucket", -3)
            self._add("inventory.sugar", -2)
            self._add("inventory.wheat", -3)
            self._add("inventory.egg", -1)
            set_path(self.state, "inventory.cake", True)
            return True, None, {"inventory.cake": True}
        if item == "enchanting_table":
            for key, amount in [("inventory.book", 1), ("inventory.diamond", 2), ("inventory.obsidian", 4)]:
                if get_path(self.state, key, 0) < amount:
                    return False, "missing_enchanting_table_ingredients", {}
            self._add("inventory.book", -1)
            self._add("inventory.diamond", -2)
            self._add("inventory.obsidian", -4)
            set_path(self.state, "inventory.enchanting_table", True)
            return True, None, {"inventory.enchanting_table": True}
        if item == "bookshelf":
            if get_path(self.state, "inventory.book", 0) < 3 or get_path(self.state, "inventory.wood", 0) < 6:
                return False, "missing_bookshelf_ingredients", {}
            self._add("inventory.book", -3)
            self._add("inventory.wood", -6)
            self._add("inventory.bookshelf", 1)
            return True, None, {"inventory.bookshelf": 1}
        if item == "splash_potion_of_weakness":
            if get_path(self.state, "inventory.potion_of_weakness", 0) < 1 or get_path(self.state, "inventory.gunpowder", 0) < 1:
                return False, "missing_splash_potion_ingredients", {}
            self._add("inventory.potion_of_weakness", -1)
            self._add("inventory.gunpowder", -1)
            self._add("inventory.splash_potion_of_weakness", 1)
            return True, None, {"inventory.splash_potion_of_weakness": 1}
        if item == "eye_of_ender":
            if get_path(self.state, "inventory.ender_pearl", 0) < 1 or get_path(self.state, "inventory.blaze_powder", 0) < 1:
                return False, "missing_eye_of_ender_ingredients", {}
            self._add("inventory.ender_pearl", -1)
            self._add("inventory.blaze_powder", -1)
            set_path(self.state, "inventory.eye_of_ender", True)
            return True, None, {"inventory.eye_of_ender": True}
        return False, "unknown_craft_item", {}

    def _gather(self, resource: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if resource == "wood":
            if get_path(self.state, "environment.biome", "") not in {"forest", "plains"}:
                return False, "resource_unavailable", {}
            self._add("inventory.wood", 4)
            return True, None, {"inventory.wood": 4}
        if resource == "apple":
            if get_path(self.state, "environment.nearby_orchard", False) is not True and get_path(self.state, "environment.biome", "") != "forest":
                return False, "apple_source_unavailable", {}
            self._add("inventory.apple", 1)
            return True, None, {"inventory.apple": 1}
        if resource == "sugar_cane":
            if get_path(self.state, "environment.sugar_cane_available", False) is not True and get_path(self.state, "environment.nearby_river", False) is not True:
                return False, "sugar_cane_unavailable", {}
            self._add("inventory.sugar_cane", 1)
            return True, None, {"inventory.sugar_cane": 1}
        if resource == "egg":
            if get_path(self.state, "environment.nearby_chicken", False) is not True and get_path(self.state, "environment.egg_available", False) is not True:
                return False, "egg_source_unavailable", {}
            self._add("inventory.egg", 1)
            return True, None, {"inventory.egg": 1}
        if resource == "flint":
            if get_path(self.state, "environment.nearby_gravel", False) is not True:
                return False, "flint_source_unavailable", {}
            self._add("inventory.flint", 1)
            return True, None, {"inventory.flint": 1}
        if resource == "nether_wart":
            if get_path(self.state, "environment.fortress_has_nether_wart", False) is not True:
                return False, "nether_wart_unavailable", {}
            self._add("inventory.nether_wart", 1)
            return True, None, {"inventory.nether_wart": 1}
        return False, "unknown_resource", {}

    def _mine(self, resource: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if resource == "diamond":
            if get_path(self.state, "environment.nearby_mine", False) is not True:
                return False, "mine_not_discovered", {}
            if pickaxe_rank(get_path(self.state, "inventory.pickaxe", "none")) < 3:
                return False, "missing_required_pickaxe", {}
            capacity = get_path(self.state, "environment.mine_diamond_capacity", None)
            mined = get_path(self.state, "environment.mine_diamonds_mined", 0)
            if isinstance(capacity, int) and mined >= capacity:
                return False, "mine_depleted", {}
            amount = min(8, capacity - mined) if isinstance(capacity, int) else 8
            self._add("inventory.diamond", amount)
            self._add("environment.mine_diamonds_mined", amount)
            return True, None, {"inventory.diamond": amount}
        if resource == "gold_ore":
            if get_path(self.state, "inventory.gold_ore", 0) > 0 and pickaxe_rank(get_path(self.state, "inventory.pickaxe", "none")) < 3:
                return True, None, {}
            if get_path(self.state, "environment.nearby_mine", False) is not True or get_path(self.state, "environment.mine_has_gold", True) is not True:
                return False, "gold_mine_unavailable", {}
            if pickaxe_rank(get_path(self.state, "inventory.pickaxe", "none")) < 3:
                return False, "missing_required_pickaxe", {}
            capacity = get_path(self.state, "environment.mine_gold_capacity", None)
            mined = get_path(self.state, "environment.mine_gold_mined", 0)
            amount = min(8, capacity - mined) if isinstance(capacity, int) else 8
            if amount <= 0:
                return False, "mine_depleted", {}
            self._add("inventory.gold_ore", amount)
            self._add("environment.mine_gold_mined", amount)
            return True, None, {"inventory.gold_ore": amount}
        if resource == "lapis":
            if get_path(self.state, "environment.nearby_mine", False) is not True or get_path(self.state, "environment.mine_has_lapis", True) is not True:
                return False, "lapis_unavailable", {}
            self._add("inventory.lapis", 3)
            return True, None, {"inventory.lapis": 3}
        if resource == "obsidian":
            if get_path(self.state, "environment.cast_obsidian", False) is True:
                self._add("inventory.obsidian", 10)
                set_path(self.state, "environment.cast_obsidian", False)
                return True, None, {"inventory.obsidian": 10}
            if get_path(self.state, "environment.nearby_mine", False) is not True or get_path(self.state, "environment.mine_has_obsidian", True) is not True:
                return False, "obsidian_unavailable", {}
            if pickaxe_rank(get_path(self.state, "inventory.pickaxe", "none")) < 3:
                return False, "missing_required_pickaxe", {}
            self._add("inventory.obsidian", 4)
            return True, None, {"inventory.obsidian": 4}
        if resource == "iron_ore":
            self._add("inventory.iron_ore", 1)
            return True, None, {"inventory.iron_ore": 1}
        return False, "unknown_resource", {}

    def _move_to(self, location: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if location not in LOCATION_REQUIREMENTS:
            return False, "unknown_location", {}
        required = LOCATION_REQUIREMENTS[location]
        if required is not None and get_path(self.state, required, False) is not True:
            return False, f"{location}_not_discovered", {}
        self.state["location"] = location
        return True, None, {"location": location}

    def _inspect(self, target: str | None) -> tuple[bool, str | None, dict[str, Any], list[Condition]]:
        assert self.state is not None
        keys = INSPECT_TARGET_KEYS.get(target or "")
        if not keys:
            return False, "unknown_inspect_target", {}, []
        delta: dict[str, Any] = {}
        revealed: list[Condition] = []
        for key in keys:
            current = get_path(self.state, key, None)
            hidden_key = key.split(".", 1)[1] if key.startswith("environment.") else key
            hidden = self._hidden(hidden_key, default=current)
            if current == UNKNOWN or hidden is not None:
                set_path(self.state, key, hidden)
                self._clear_ambiguous(key)
                delta[key] = hidden
                revealed.append(Condition(key, "==", hidden, source="inspect"))
        return True, None, delta, revealed

    def _explore(self, target: str | None) -> tuple[bool, str | None, dict[str, Any], list[Condition]]:
        assert self.state is not None
        if target not in EXPLORE_TARGETS:
            return False, "unknown_explore_target", {}, []
        search_key, reveal_key = EXPLORE_TARGETS[target]
        if not get_path(self.state, search_key, False):
            return False, f"{target}_search_unavailable", {}, []
        hidden_key = reveal_key.split(".", 1)[1]
        value = self._hidden(hidden_key, default=False)
        set_path(self.state, reveal_key, value)
        set_path(self.state, search_key, False)
        self._clear_ambiguous(reveal_key)
        delta = {reveal_key: value, search_key: False}
        revealed = [Condition(reveal_key, "==", value, source="explore")]
        hidden_facts = dict((self.current_case or {}).get("hidden_facts", {}))
        hidden_facts.update(self.state.get("hidden_facts", {}))
        for hidden_name, hidden_value in hidden_facts.items():
            dotted = f"environment.{hidden_name}"
            if get_path(self.state, dotted, None) in {None, UNKNOWN}:
                set_path(self.state, dotted, hidden_value)
                delta[dotted] = hidden_value
                self._clear_ambiguous(dotted)
                revealed.append(Condition(dotted, "==", hidden_value, source="explore"))
        return True, None, delta, revealed

    def _report_impossible(self, reason: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if self._needs_more_information():
            return False, "premature_impossible_report", {}
        oracle_solvable = (self.current_case or {}).get("oracle", {}).get("solvable")
        if oracle_solvable is True:
            return False, "incorrect_impossible_report", {}
        return True, None, {"task.reported_impossible": reason or "no_viable_plan"}

    def _needs_more_information(self) -> bool:
        assert self.state is not None
        ambiguous = self.state.get("ambiguous")
        if isinstance(ambiguous, dict) and ambiguous:
            return True
        environment = self.state.get("environment", {})
        if isinstance(environment, dict):
            return any(value == UNKNOWN for value in environment.values())
        return False

    def _trade(self, villager: str | None, want: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if get_path(self.state, "location", "base") != "village":
            return False, "not_at_village", {}
        villager_key = f"environment.village_has_{villager}" if villager else ""
        if villager_key and get_path(self.state, villager_key, True) == UNKNOWN:
            return False, f"{villager}_unknown", {}
        if villager_key and get_path(self.state, villager_key, True) is not True:
            return False, f"no_{villager}", {}
        costs = {
            "diamond_set": 40,
            "diamond_helmet": 10,
            "diamond_chestplate": 10,
            "diamond_leggings": 10,
            "diamond_boots": 10,
            "apple": 2,
            "gold_ingot": 4,
            "wheat": 2,
            "sugar": 2,
            "enchanted_book": 12,
            "ender_pearl": 4,
        }
        cost = costs.get(want or "", 1)
        if get_path(self.state, "inventory.emerald", 0) < cost:
            return False, "not_enough_emeralds", {}
        self._add("inventory.emerald", -cost)
        if want == "diamond_set":
            for piece in ARMOR_PIECES:
                set_path(self.state, f"inventory.{piece}", True)
            return True, None, {"inventory.emerald": -cost, "inventory.diamond_set": True}
        if want in ARMOR_PIECES:
            set_path(self.state, f"inventory.{want}", True)
            return True, None, {"inventory.emerald": -cost, f"inventory.{want}": True}
        trade_yields = {"gold_ingot": 4, "wheat": 3, "sugar": 2}
        if want in {"apple", "gold_ingot", "wheat", "sugar", "enchanted_book", "ender_pearl"}:
            amount = trade_yields.get(want, 1)
            self._add(f"inventory.{want}", amount)
            return True, None, {"inventory.emerald": -cost, f"inventory.{want}": amount}
        return False, "unknown_trade_item", {}

    def _smelt(self, resource: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        if resource == "gold_ore":
            amount = get_path(self.state or {}, "inventory.gold_ore", 0)
            if amount <= 0 or not get_path(self.state or {}, "inventory.furnace", False):
                return False, "cannot_smelt_gold", {}
            self._add("inventory.gold_ore", -amount)
            self._add("inventory.gold_ingot", amount)
            return True, None, {"inventory.gold_ore": -amount, "inventory.gold_ingot": amount}
        if resource == "iron_ore":
            amount = get_path(self.state or {}, "inventory.iron_ore", 0)
            if amount <= 0 or not get_path(self.state or {}, "inventory.furnace", False):
                return False, "cannot_smelt_iron", {}
            self._add("inventory.iron_ore", -amount)
            self._add("inventory.iron_ingot", amount)
            return True, None, {"inventory.iron_ore": -amount, "inventory.iron_ingot": amount}
        return False, "unknown_smelt_resource", {}

    def _loot(self, target: str | None, item: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if target not in {"chest", "stand"} or not item:
            return False, "unknown_loot_target", {}
        if item == "golden_apple" and get_path(self.state, "environment.chest_has_golden_apple", True) is not True and get_path(self.state, "environment.basement_has_cure_supplies", True) is not True:
            return False, "loot_unavailable", {}
        if item == "fire_resistance_potion" and get_path(self.state, "environment.chest_has_fire_resistance_potion", True) is not True:
            return False, "loot_unavailable", {}
        if item == "splash_potion_of_weakness" and get_path(self.state, "environment.basement_has_cure_supplies", True) is not True and get_path(self.state, "environment.igloo_has_weakness_potion", True) is not True:
            return False, "loot_unavailable", {}
        if item in {"golden_apple", "splash_potion_of_weakness"}:
            self._add(f"inventory.{item}", 1)
            return True, None, {f"inventory.{item}": 1}
        if item == "fire_resistance_potion":
            set_path(self.state, "inventory.fire_resistance_potion", True)
            return True, None, {"inventory.fire_resistance_potion": True}
        return False, "unknown_loot_item", {}

    def _brew(self, item: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if not get_path(self.state, "inventory.brewing_stand", False):
            return False, "missing_brewing_stand", {}
        if item == "awkward_potion":
            if get_path(self.state, "inventory.water_bottle", 0) < 1 or get_path(self.state, "inventory.nether_wart", 0) < 1:
                return False, "missing_awkward_potion_ingredients", {}
            self._add("inventory.water_bottle", -1)
            self._add("inventory.nether_wart", -1)
            self._add("inventory.awkward_potion", 1)
            return True, None, {"inventory.awkward_potion": 1}
        if item == "fire_resistance_potion":
            if get_path(self.state, "inventory.awkward_potion", 0) < 1 or get_path(self.state, "inventory.magma_cream", 0) < 1:
                return False, "missing_fire_resistance_ingredients", {}
            self._add("inventory.awkward_potion", -1)
            self._add("inventory.magma_cream", -1)
            set_path(self.state, "inventory.fire_resistance_potion", True)
            return True, None, {"inventory.fire_resistance_potion": True}
        if item == "potion_of_weakness":
            if get_path(self.state, "inventory.water_bottle", 0) < 1 or get_path(self.state, "inventory.fermented_spider_eye", 0) < 1:
                return False, "missing_weakness_ingredients", {}
            self._add("inventory.water_bottle", -1)
            self._add("inventory.fermented_spider_eye", -1)
            self._add("inventory.potion_of_weakness", 1)
            return True, None, {"inventory.potion_of_weakness": 1}
        return False, "unknown_brew_item", {}

    def _fill(self, item: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        if item != "water_bottle":
            return False, "unknown_fill_item", {}
        if get_path(self.state or {}, "environment.nearby_water", False) is not True:
            return False, "water_unavailable", {}
        if get_path(self.state or {}, "inventory.empty_bottle", 0) < 1:
            return False, "missing_empty_bottle", {}
        self._add("inventory.empty_bottle", -1)
        self._add("inventory.water_bottle", 1)
        return True, None, {"inventory.empty_bottle": -1, "inventory.water_bottle": 1}

    def _place(self, item: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if item == "obsidian_portal_frame":
            if get_path(self.state, "inventory.obsidian", 0) < 10:
                return False, "missing_obsidian", {}
            self._add("inventory.obsidian", -10)
            set_path(self.state, "environment.portal_frame", True)
            return True, None, {"inventory.obsidian": -10, "environment.portal_frame": True}
        if item == "obsidian":
            if get_path(self.state, "inventory.obsidian", 0) < 1:
                return False, "missing_obsidian", {}
            self._add("inventory.obsidian", -1)
            missing = get_path(self.state, "environment.ruined_portal_missing_obsidian", 1)
            if isinstance(missing, int):
                missing -= 1
                set_path(self.state, "environment.ruined_portal_missing_obsidian", missing)
                if missing <= 0:
                    set_path(self.state, "environment.portal_frame", True)
            return True, None, {"inventory.obsidian": -1, "environment.ruined_portal_missing_obsidian": missing}
        if item == "bookshelf":
            if get_path(self.state, "inventory.bookshelf", 0) < 1:
                return False, "missing_bookshelf", {}
            self._add("inventory.bookshelf", -1)
            set_path(self.state, "environment.bookshelf_placed", True)
            return True, None, {"inventory.bookshelf": -1, "environment.bookshelf_placed": True}
        if item == "enchanting_table":
            if not get_path(self.state, "inventory.enchanting_table", False):
                return False, "missing_enchanting_table", {}
            set_path(self.state, "environment.enchanting_table_placed", True)
            return True, None, {"environment.enchanting_table_placed": True}
        return False, "unknown_place_item", {}

    def _cast(self, item: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if get_path(self.state, "environment.nearby_lava_pool", False) is not True or get_path(self.state, "inventory.water_bucket", False) is not True:
            return False, "casting_unavailable", {}
        if item == "obsidian":
            set_path(self.state, "environment.cast_obsidian", True)
            return True, None, {"environment.cast_obsidian": True}
        if item == "portal_frame":
            set_path(self.state, "environment.portal_frame", True)
            return True, None, {"environment.portal_frame": True}
        return False, "unknown_cast_item", {}

    def _ignite(self, item: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if item != "portal":
            return False, "unknown_ignite_item", {}
        if get_path(self.state, "inventory.flint_and_steel", False) is not True:
            return False, "missing_igniter", {}
        missing = get_path(self.state, "environment.ruined_portal_missing_obsidian", 0)
        if get_path(self.state, "environment.portal_frame", False) is not True and missing not in {0, False}:
            return False, "missing_portal_frame", {}
        set_path(self.state, "environment.portal_lit", True)
        return True, None, {"environment.portal_lit": True}

    def _enchant(self, item: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if item != "pickaxe":
            return False, "unknown_enchant_item", {}
        if not get_path(self.state, "inventory.enchanting_table", False):
            return False, "missing_enchanting_table", {}
        if get_path(self.state, "inventory.lapis", 0) < 1:
            return False, "missing_lapis", {}
        if get_path(self.state, "inventory.xp_level", 0) < 1:
            return False, "missing_experience", {}
        if get_path(self.state, "environment.high_level_enchant_required", False) is True and not get_path(self.state, "environment.bookshelf_placed", False):
            return False, "missing_bookshelf", {}
        self._add("inventory.lapis", -1)
        set_path(self.state, "inventory.enchanted_pickaxe", True)
        return True, None, {"inventory.lapis": -1, "inventory.enchanted_pickaxe": True}

    def _fight(self, target: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if target == "blaze":
            if get_path(self.state, "environment.fortress_has_blaze", True) is not True:
                return False, "blaze_unavailable", {}
            self._add("inventory.blaze_rod", 1)
            return True, None, {"inventory.blaze_rod": 1}
        if target == "monsters":
            if get_path(self.state, "environment.safe_combat", True) is not True:
                return False, "combat_unsafe", {}
            self._add("inventory.xp_level", 10)
            return True, None, {"inventory.xp_level": 10}
        if target == "enderman":
            if get_path(self.state, "environment.enderman_route_available", True) is not True and get_path(self.state, "environment.nighttime", False) is not True:
                return False, "enderman_unavailable", {}
            self._add("inventory.ender_pearl", 1)
            return True, None, {"inventory.ender_pearl": 1}
        if target == "magma_cube":
            if get_path(self.state, "environment.nearby_magma_cube", False) is not True:
                return False, "magma_cube_unavailable", {}
            self._add("inventory.slimeball", 1)
            return True, None, {"inventory.slimeball": 1}
        return False, "unknown_fight_target", {}

    def _use_anvil(self, item: str | None, material: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        if item != "pickaxe" or material != "enchanted_book":
            return False, "unknown_anvil_recipe", {}
        if not get_path(self.state or {}, "inventory.anvil", False) or get_path(self.state or {}, "inventory.enchanted_book", 0) < 1:
            return False, "missing_anvil_or_book", {}
        self._add("inventory.enchanted_book", -1)
        set_path(self.state, "inventory.enchanted_pickaxe", True)
        return True, None, {"inventory.enchanted_book": -1, "inventory.enchanted_pickaxe": True}

    def _milk(self, target: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        if target != "cow":
            return False, "unknown_milk_target", {}
        if get_path(self.state or {}, "environment.nearby_cow", False) is not True:
            return False, "cow_unavailable", {}
        buckets = get_path(self.state or {}, "inventory.bucket", 0)
        if buckets < 1:
            return False, "missing_bucket", {}
        amount = min(3, buckets)
        self._add("inventory.milk_bucket", amount)
        return True, None, {"inventory.milk_bucket": amount}

    def _plant(self, item: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        if item != "wheat":
            return False, "unknown_plant_item", {}
        if get_path(self.state or {}, "environment.farmland_available", False) is not True or get_path(self.state or {}, "inventory.seeds", 0) < 1:
            return False, "cannot_plant_wheat", {}
        self._add("inventory.seeds", -1)
        set_path(self.state, "environment.wheat_planted", True)
        return True, None, {"inventory.seeds": -1, "environment.wheat_planted": True}

    def _harvest(self, item: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        if item != "wheat" or get_path(self.state or {}, "environment.wheat_planted", False) is not True:
            return False, "cannot_harvest_wheat", {}
        self._add("inventory.wheat", 3)
        return True, None, {"inventory.wheat": 3}

    def _throw(self, item: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        if item != "splash_potion_of_weakness" or get_path(self.state or {}, "inventory.splash_potion_of_weakness", 0) < 1:
            return False, "missing_splash_potion_of_weakness", {}
        self._add("inventory.splash_potion_of_weakness", -1)
        set_path(self.state, "environment.weakness_applied", True)
        return True, None, {"inventory.splash_potion_of_weakness": -1, "environment.weakness_applied": True}

    def _use(self, item: str | None, target: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        if item != "golden_apple" or target != "zombie_villager":
            return False, "unknown_use_action", {}
        if get_path(self.state or {}, "environment.zombie_villager_present", False) is not True:
            return False, "no_zombie_villager", {}
        if get_path(self.state or {}, "environment.weakness_applied", False) is not True:
            return False, "weakness_not_applied", {}
        if get_path(self.state or {}, "inventory.golden_apple", 0) < 1:
            return False, "missing_golden_apple", {}
        self._add("inventory.golden_apple", -1)
        set_path(self.state, "environment.golden_apple_used", True)
        return True, None, {"inventory.golden_apple": -1, "environment.golden_apple_used": True}

    def _wait(self, event: str | None) -> tuple[bool, str | None, dict[str, Any]]:
        assert self.state is not None
        if event == "night":
            set_path(self.state, "environment.nighttime", True)
            return True, None, {"environment.nighttime": True}
        if event == "egg_drop":
            set_path(self.state, "environment.egg_available", True)
            return True, None, {"environment.egg_available": True}
        if event == "cure_complete":
            if get_path(self.state, "environment.weakness_applied", False) is not True or get_path(self.state, "environment.golden_apple_used", False) is not True:
                return False, "cure_requirements_missing", {}
            set_path(self.state, "environment.zombie_villager_cured", True)
            return True, None, {"environment.zombie_villager_cured": True}
        return False, "unknown_wait_event", {}

    def _add(self, dotted_key: str, amount: int) -> None:
        assert self.state is not None
        current = get_path(self.state, dotted_key, 0)
        set_path(self.state, dotted_key, current + amount)

    def _hidden(self, key: str, default: Any) -> Any:
        assert self.current_case is not None
        return self.current_case.get("hidden_facts", {}).get(key, self.state.get("hidden_facts", {}).get(key, default))

