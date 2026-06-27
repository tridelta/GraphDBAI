# MyTextCraft World v1 Benchmark

Last updated: 2026-06-27

## Purpose

`mytextcraft_world_v1` is the world-level benchmark suite for MyTextCraft. It loads all current task families into one shared symbolic world schema, while each episode still resets to the selected case's own initial setup.

The suite is intended for open-world-style experiments where an agent faces many task types over repeated rounds and must reuse experience across tasks.

## Files

- `world_cases/mytextcraft_world_v1.yaml`: benchmark suite manifest.
- `world_cases/mytextcraft_world_v1_short10.yaml`: short high-difficulty suite for quick iteration.
- `world_cases/mytextcraft_default_state.yaml`: shared default state schema.
- `world_cases/task_families/`: all current task family cases.
- `world_cases/textcraft_rules.yaml`: task action spaces and world rules.

## Loaded Task Families

The suite currently includes all 9 task families:

- `cake`
- `cure_zombie_villager`
- `diamond_set`
- `enchant_pickaxe`
- `eye_of_ender`
- `fire_resistance_potion`
- `golden_apple`
- `golden_equipment_chain`
- `nether_portal`

The current loader reads 102 cases from these families.

## State Schema

Every episode starts from:

```text
default_state + case.initial_state
```

This means all tasks share the same top-level state shape:

- `location`
- `inventory`
- `environment`
- `tool`
- `ambiguous`
- `hidden_facts`

The concrete values differ by case. For example, one case may set `nearby_mine: true`, another may set `nearby_mine: unknown`, and another may leave it as the default `false`.

`ambiguous` remains sparse by design. It only lists facts currently hidden from the agent. This avoids filling prompts with unrelated unknown fields.

## Reset Semantics

State is not carried across episodes. A new episode resets the environment to the selected case setup.

Experience is carried by the agent or graph store, not by the environment state. This keeps the benchmark clean:

- environment state resets every episode
- ExperienceGraph persists across episodes inside the same run
- repeated rounds test whether experience improves future decisions

## Main Evaluation Filter

Main benchmark runs should use solvable cases only:

```powershell
--solvable-only
```

Older no-route cases are retained for diagnostics but should not be mixed into the primary success-rate metric.

## 5 x n Round Protocol

For quick iteration, use the short high-difficulty suite:

Recommended wrapper:

```powershell
uv run python -B tools/run_mytextcraft_world_v1_experiment.py `
  --preset short10 `
  --provider deepseek `
  --model deepseek-v4-flash `
  --rounds 5 `
  --max-steps 30 `
  --max-budget-rmb 50 `
  --ack-external-api
```

The wrapper starts the experiment in the background, writes logs under `output/logs/`, writes a progress hook under `output/experiment_jobs/`, and starts the local panel at `http://127.0.0.1:8765/`.

The short suite contains 10 solvable medium/hard cases, so 5 rounds expands to 50 episodes.

Direct command:

```powershell
uv run eg-run-experiment `
  --agent graph `
  --variant full `
  --cases world_cases/mytextcraft_world_v1_short10.yaml `
  --rules world_cases/textcraft_rules.yaml `
  --task-id all `
  --solvable-only `
  --rounds 5 `
  --case-schedule ordered `
  --max-steps 30 `
  --run-id mytextcraft_world_v1_graph_full_flash_s1
```

If `n` solvable cases are selected, `--rounds 5` expands to `5 * n` episodes.

Use `--preset all` only when intentionally running the full world suite.

Use the same command with different agents or variants for comparisons:

- `--agent react`
- `--agent graph --variant no_graph_context`
- `--agent graph --variant full`

Recommended first comparison:

```text
react vs graph/no_graph_context vs graph/full
```

This isolates whether improvement comes from cross-episode experience and graph retrieval.

## Reporting

A round-level report should show:

- success rate per round
- average successful steps per round
- failure reasons
- repeated action rate
- graph node/edge/path growth
- candidate path count entering prompts
- token and cost per successful episode

For the paper, this suite should be described as an early world-level benchmark. Claims should remain limited until multi-seed baseline runs are complete.
