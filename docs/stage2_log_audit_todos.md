# Stage 2 Log Audit TODOs

These items came from the Stage 2 real S2C run audit on seed 2501.

## Code Follow-ups

- [ ] Add a smoke test that runs each real agent with `EXPERIENCE_GRAPH_LLM_PROVIDER=fake` and checks every prompt has `prompt_hidden_facts=false`.
- [ ] Add an analyzer warning for repeated identical actions, using `repeated_action_count >= 3` as the first alert threshold.
- [ ] Extend resume cleanup to detect the rare case where graph store files were saved but `metrics.jsonl` was not updated.
- [ ] Add a run preflight check that refuses non-`--resume` runs when any episode-scoped JSONL already exists in the target run directory.

## Experiment Design Follow-ups

- [ ] Separate "runner built graph metrics" from "agent consumed graph context" in analysis tables.
- [ ] Report `prompt_candidate_paths`, `prompt_retrieved_skills`, and `prompt_retrieved_trajectories` in Stage 2 summaries.
- [ ] Decide whether SkillLibrary should have a warm-start phase; in short 3-episode runs it may have no retrieved skills before the final episode.
- [ ] Decide whether VectorTrajectory should receive structured positive/negative trajectory fields instead of one free-text summary.
- [ ] Add an ablation note that ReAct, Reflexion, SkillLibrary, and VectorTrajectory do not consume graph candidate paths unless explicitly wired to do so.

## Reporting Follow-ups

- [ ] Mark the audited seed-2501 Stage 2 real S2C runs as affected by prompt leakage and stale ambiguous-state logging.
- [ ] Re-run the five medium seed-2501 agents after these fixes before using their numbers in `paper/paper_draft.md`.
- [ ] Keep `[PROJECTED]` results in the paper as projected until the fixed runs produce verified metrics.
