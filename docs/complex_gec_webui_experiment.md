# Complex GEC WebUI Experiment

Last updated: 2026-06-21

The ExperienceGraph panel includes a local WebUI for running the complex `golden_equipment_chain` experiment.

Start the panel:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run eg-panel --run-dir runs --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765/
```

Default experiment:

- Agent: `graph`
- Task: `golden_equipment_chain`
- Cases: `GEC_004,GEC_005,GEC_006`
- Episodes: `9`
- Schedule: `ordered`, so the three cases repeat for three rounds
- Max steps: `14`
- Provider: `deepseek`
- Model: `deepseek-v4-pro`
- Budget: `20 RMB`

The WebUI starts `experience_graph.scripts.run_experiment` in a background process and writes stdout/stderr to `output/logs/<run_id>.out.log` and `output/logs/<run_id>.err.log`.

For `deepseek` or `openai`, the WebUI requires the external API acknowledgement checkbox before starting. This is intentional because prompts include visible task state, available actions, and retrieved experience summaries.

The result panel shows:

- Overall success rate and successful-step average
- Round summaries inferred from the repeated case cycle
- Graph node, edge, and path growth
- Latest episode status and failure reasons

Use the round table to judge whether ExperienceGraph is improving: later rounds should either increase success rate or keep success high while reducing successful steps. If the trend is flat, inspect `steps.jsonl` for repeated invalid output, unavailable actions, or candidate path usage.
