# MyTextCraft Task Standard

Last updated: 2026-06-21

MyTextCraft is the symbolic crafting benchmark used by ExperienceGraph. It should be treated as a benchmark artifact, not only as a helper environment. New tasks should follow one shared schema so that agents, oracle checks, analysis scripts, and paper claims all talk about the same task definition.

## Design Goal

MyTextCraft tasks should test reusable experience, not one-off puzzle memorization. A good task family has several viable routes, hidden or ambiguous facts that can be resolved by actions, and subgoals that appear again in other task families.

Future main evaluation should avoid impossible cases by default. As the world gains more rules and actions, most cases should have at least one viable route. Impossible or no-route cases can stay as diagnostic assets for parser, reporting, and safety checks, but they should not drive the main success-rate claim unless the metric is explicitly named `case_resolved`.

## Current Implementation Status

- `MyTextCraftAdapter.available_actions()` uses `tasks.<task_id>.action_space` from `world_cases/textcraft_rules.yaml` when present.
- `craft(...)` first checks declarative action rules in `world_cases/textcraft_rules.yaml`, then falls back to legacy Python branches for compatibility.
- Declarative craft coverage currently includes diamond armor pieces, crafting table, golden apple, golden helmet, cake, sugar, potion ingredients, enchanting-table items, and eye of ender.
- `report_impossible` remains available for diagnostic cases, but future main experiments should prefer solvable schedules.
- `run_experiment.py --case-ids` can run a fixed subset in a requested order, which is useful for repeated-round learning experiments without clearing the graph.
- `available_actions()` filters declared actions by the current visible state so agents are not offered routes that are already unavailable.
- The panel summary API reports repeated-case round summaries and graph growth so learning progress can be inspected during long runs.

## Required Task Family Fields

Each file in `world_cases/task_families/` should define one task family. Shared success conditions and action spaces should live in `world_cases/textcraft_rules.yaml` under `tasks.<task_id>`:

```yaml
version: 0.2-draft
world: mytextcraft_mc
status: draft_not_active
family: golden_apple
goal: Obtain a golden apple.
visible_manual_refs:
  - golden_apple_crafting
  - apple_gathering
required_engine_actions:
  - craft
  - gather
  - mine
  - smelt
  - report_impossible
action_space:
  - craft(golden_apple)
  - gather(apple)
minecraft_alignment:
  - Vanilla golden apple crafting uses one apple and eight gold ingots.
cases:
  - id: GA_001
    title: Craft golden apple from existing ingredients.
    difficulty: easy
    characteristic: direct_success
    initial_state: {...}
    oracle:
      solvable: true
      reference_plan: ["craft(golden_apple)"]
      min_steps: 1
      expected_strategy: direct_crafting
    expected_experience:
      path_type: crafting
      learn: [golden_apple_requires_apple_and_8_gold_ingots]
```

Required fields:

- `family`: stable task id used by `TaskSpec.id`.
- `goal`: human-readable objective shown to humans and eventually to agents.
- `visible_manual_refs`: task-local manuals that may be exposed in prompts.
- `required_engine_actions`: action names needed by this task family.
- `tasks.<task_id>.action_space` in `textcraft_rules.yaml`: concrete action templates exposed to agents.
- `minecraft_alignment`: notes for simplified mechanics.
- `cases`: concrete initial states, oracle plans, and expected learning signals.

## Required Case Fields

Each case should include:

- `id`: stable id with a family prefix, such as `GA_010`.
- `title`: short case description.
- `difficulty`: `easy`, `medium`, `hard`, or `diagnostic` for future no-route checks.
- `characteristic`: one route or challenge label, such as `hidden_gold_mine`.
- `initial_state`: visible state plus optional `hidden_facts` and `ambiguous` fields.
- `oracle.solvable`: whether the goal can be reached under the current rule set.
- `oracle.reference_plan`: one valid reference plan, not the whole action space.
- `oracle.expected_strategy`: route family used by the reference plan.
- `expected_experience.path_type`: the kind of path the graph should learn.
- `expected_experience.learn`: concise expected learning tags.

## Metrics Contract

Use two separate notions in analysis:

- `goal_achieved`: the task success conditions are satisfied.
- `case_resolved`: the task was completed or a diagnostic no-route case was correctly reported.

Current runner logs still use `success`. Before final experiments with diagnostic cases, either remove diagnostic cases from the main suite or add explicit `goal_achieved` and `case_resolved` fields.

## Action Space Contract

`available_actions` comes from `tasks.<task_id>.action_space` in `world_cases/textcraft_rules.yaml` when that field is present. Oracle plans remain references for validation. They should not define what the agent is allowed to try.

Preferred direction:

1. Keep action names finite and typed: `craft(item)`, `mine(resource)`, `trade(villager, offer, want)`, etc.
2. Define recipes and hard preconditions in rules YAML.
3. Generate `ActionSpec` entries from `tasks.<task_id>.action_space`.
4. Validate oracle plans against the generated action space.

## Compositional Task Pattern

Complex tasks should combine reusable subgoals across families. For example:

1. Obtain an apple through orchard gathering, trade, or loot.
2. Obtain gold ingots through mining, smelting, trade, or loot.
3. Craft a golden apple.
4. Use the golden apple as a subgoal for a later objective, such as curing a zombie villager or crafting a gold-related equipment chain.

This pattern is better for ExperienceGraph than a single long custom route because it lets the graph reuse paths across episodes and tasks.

## Promotion Checklist

Before promoting a new task family to the active suite:

- The family file loads through `MyTextCraftAdapter`.
- Every reference plan passes `CaseOracle` validation.
- Hidden facts are absent from observations and prompts.
- The task-local manual does not expose unrelated recipes.
- Scripted smoke works for the family.
- Fake-provider LLM smoke still writes `llm_input_messages`, `llm_output`, `llm_raw_response`, `llm_parse_error`, `llm_finish_reason`, and `llm_attempts`.
- Main evaluation excludes diagnostic no-route cases unless metrics distinguish `goal_achieved` from `case_resolved`.



