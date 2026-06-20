# Stage 2 Real Pilot Run Package

This package runs the fixed Stage 2 pilot locally with real DeepSeek calls and then builds the analysis report and `matplotlib.pyplot` figures.

## What It Runs

- Conditions:
  - `react/full`
  - `reflexion/full`
  - `vector_trajectory/full`
  - `skill_library/full`
  - `graph/full`
  - `graph/no_graph_context` as ExperienceGraph without ExperienceGraph context
- Difficulty: `medium`
- Seed: `2501`
- Episodes per condition: `5`
- Max steps per episode: `12`
- Case schedule: `shuffled_cycle`
- Provider/model: DeepSeek / `deepseek-v4-flash`
- Base output cap: `--llm-max-tokens 4096`
- Retry output cap: `--llm-retry-max-tokens 8192`
- Retry policy: `--llm-retries 1`
- Route-failure policy: continue after recoverable environment/precondition failures while step budget remains
- Impossible-report action: `report_impossible(reason=no_viable_plan)` after visible alternatives are exhausted
- Budget guard: `--max-budget-rmb 50` per run command

The script runs episodes sequentially inside each run, preserving cross-episode graph/memory learning. It can run independent conditions in parallel with `-Parallel`; this does not change the learning semantics because each condition writes to its own run directory. If a run directory already contains `config.yaml`, it passes `--resume`.

The key ablation is `graph/full` vs `graph/no_graph_context`: both use `ExperienceGraphAgent`, but `no_graph_context` sets graph retrieval `top_k=0`, so the agent receives no Experience Graph candidate paths.

## Before Running

From the repository root, set your DeepSeek key in the same PowerShell terminal:

```powershell
$env:DEEPSEEK_API_KEY = "YOUR_KEY_HERE"
```

Optional price settings, if you want a custom RMB estimate:

```powershell
$env:EG_INPUT_PRICE_PER_M_RMB = "1.0"
$env:EG_OUTPUT_PRICE_PER_M_RMB = "3.0"
```

## Run Everything

```powershell
powershell -ExecutionPolicy Bypass -File tools\stage2_real_pilot_package\run_stage2_real_pilot.ps1
```

PowerShell 7 also works:

```powershell
pwsh -ExecutionPolicy Bypass -File tools\stage2_real_pilot_package\run_stage2_real_pilot.ps1
```

To run independent conditions concurrently:

```powershell
powershell -ExecutionPolicy Bypass -File tools\stage2_real_pilot_package\run_stage2_real_pilot.ps1 -Parallel -MaxWorkers 3
```

Do not use episode-level parallelism for the main learning runs. `graph`, `reflexion`, `vector_trajectory`, and `skill_library` intentionally update state after each episode, so parallel episodes inside one run would change the experiment definition.

## Useful Overrides

```powershell
powershell -ExecutionPolicy Bypass -File tools\stage2_real_pilot_package\run_stage2_real_pilot.ps1 -MaxBudgetRmb 100 -LlmMaxTokens 4096 -LlmRetryMaxTokens 8192 -LlmRetries 1
```

## Analyze Only

If all six real runs already finished and you only want to rebuild CSVs, PNGs, and the report:

```powershell
powershell -ExecutionPolicy Bypass -File tools\stage2_real_pilot_package\run_stage2_real_pilot.ps1 -AnalyzeOnly
```

## Outputs

Run directories:

- `runs/stage2_pilot_real_s2e_react_medium_seed2501`
- `runs/stage2_pilot_real_s2e_reflexion_medium_seed2501`
- `runs/stage2_pilot_real_s2e_vector_trajectory_medium_seed2501`
- `runs/stage2_pilot_real_s2e_skill_library_medium_seed2501`
- `runs/stage2_pilot_real_s2e_graph_full_medium_seed2501`
- `runs/stage2_pilot_real_s2e_graph_no_graph_context_medium_seed2501`

Analysis directory:

- `runs/stage2_pilot_real_s2e_analysis`

Expected analysis artifacts:

- `main_comparison.csv`
- `token_cost.csv`
- `graph_growth.csv`
- `learning_curve_medium.csv`
- `ablation.csv`
- `summary_tables.md`
- `experiment_writeup.md`
- `stage2_pilot_report.md`
- `figures/learning_curve_medium.png`
- `figures/graph_growth.png`
- `figures/token_cost.png`

## Built-in Validation

The runner checks that:

- each condition has exactly 5 `metrics.jsonl` rows,
- required JSONL/config files exist,
- `prompt_hidden_facts` count is 0,
- all three figure files are valid PNG files.

If validation passes, the last line should include:

```text
VALIDATION OK
```

## After It Finishes

Tell Codex that the real run is complete. I can then inspect `runs/stage2_pilot_real_s2e_analysis/stage2_pilot_report.md`, verify the logs again, and deliver the final Stage 2 report summary.
