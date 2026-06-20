# TextCraft-MC World Cases

This folder contains the first structured test assets for TextCraft-MC and ExperienceGraph.

Files:

- `textcraft_rules.yaml`: world rules used by the simulator and oracle planner.
- `textcraft_cases.yaml`: hand-written task cases with initial states, expected outcomes, oracle paths, and expected graph updates.
- `sample_experiences.jsonl`: example `ExperienceRecord` rows derived from representative cases.

The goal is to keep these files readable for research discussion while making them strict enough for later automated tests.

## Case Schema

Each case in `textcraft_cases.yaml` contains:

- `id`: stable case id.
- `difficulty`: `easy`, `medium`, `hard`, or `impossible`.
- `task`: target task id and success conditions.
- `initial_state`: symbolic TextCraft-MC state.
- `oracle`: expected solvability and a reference plan.
- `expected_experience`: what a correct ExperienceGraph update should learn from the case.
- `notes`: short explanation for humans.

## Experience Semantics

An experience is not just a text reflection. It records:

- what state the agent saw,
- what action it took,
- what changed,
- what failed,
- what hidden facts were discovered,
- what graph nodes, edges, or preconditions should be updated.

These cases intentionally include success, failure, hidden information, mixed paths, and one impossible task.
