# MyTextCraft World Cases

Last updated: 2026-06-27

This folder contains structured benchmark assets for MyTextCraft and ExperienceGraph.

Files:

- `textcraft_rules.yaml`: legacy filename for MyTextCraft world rules used by the simulator and oracle planner.
- `textcraft_cases.yaml`: legacy filename for the original diamond-set active suite.
- `mytextcraft_default_state.yaml`: shared default state schema used by world-level suites.
- `mytextcraft_world_v1.yaml`: manifest that loads all current task families into one world-level benchmark suite.
- `task_families/`: draft multi-family MyTextCraft task suite with 9 families and 102 cases.
- `sample_experiences.jsonl`: example `ExperienceRecord` rows derived from representative cases.

The goal is to keep these files readable for research discussion while making them strict enough for automated validation.

## Case Schema

Each case contains:

- `id`: stable case id.
- `difficulty`: `easy`, `medium`, `hard`, or `diagnostic` for future no-route checks.
- `task`: target task id and success conditions when defined per case.
- `initial_state`: symbolic MyTextCraft state.
- `oracle`: expected solvability and a reference plan.
- `expected_experience`: what a correct ExperienceGraph update should learn from the case.
- `notes`: short explanation for humans.

See `docs/mytextcraft_task_standard.md` for the current task-family standard. Shared task action spaces are declared in `textcraft_rules.yaml`.

## Impossible Cases

Older suites contain `impossible` cases because the early world had fewer actions and rules. Future main evaluation should prefer solvable cases and keep no-route cases as optional diagnostics. If diagnostic cases are included in an experiment, analysis should distinguish goal completion from correct case resolution.

