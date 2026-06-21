# MyTextCraft World Cases

This folder contains structured benchmark assets for MyTextCraft and ExperienceGraph.

Files:

- `textcraft_rules.yaml`: legacy filename for MyTextCraft world rules used by the simulator and oracle planner.
- `textcraft_cases.yaml`: legacy filename for the original diamond-set active suite.
- `task_families/`: draft multi-family MyTextCraft task suite.
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

See `docs/mytextcraft_task_standard.md` for the current task-family standard.

## Impossible Cases

Older suites contain `impossible` cases because the early world had fewer actions and rules. Future main evaluation should prefer solvable cases and keep no-route cases as optional diagnostics. If diagnostic cases are included in an experiment, analysis should distinguish goal completion from correct case resolution.
