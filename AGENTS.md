# Project Instructions

- 通常使用中文回复；代码标识、API 名、论文术语可保留英文。
- 动手修改前先说明假设、改动范围和验证方式；需求已经明确时直接执行。
- 不要把 `paper/paper_draft.md` 中的 `[PROJECTED]` 结果改写为已验证结论。
- 修改实验相关代码时，对照 `paper/paper_draft.md` 和 `docs/experience_graph_system_design.md` 检查 claim 与可产出日志是否一致。
- 保持改动聚焦：只修改和当前任务直接相关的文件，不做无关重构。

## Experiment Rules

- 正式调用付费 LLM 前，必须先用 `EXPERIENCE_GRAPH_LLM_PROVIDER=fake` 或 `--agent scripted` 做 smoke run。
- 每次实验必须写入可复现实验配置，包括 `run_id`、`agent`、`variant`、`seed`、`episodes`、`case_schedule`、`difficulty`、`max_steps`、`top_k`、`token_budget`。
- 每个 episode 必须记录：`episode_id`、`case_id`、`difficulty`、`seed`、`agent`、`variant`、`success`、`steps`、`failure_reason`、图节点数、图边数、路径数、新增/更新图元素数量。
- 每个 step 必须记录：动作、动作结果、失败原因、reward/cost、experience view id、candidate path 数量、view token estimate、LLM 调用和 token 增量。
- LLM token 统计优先使用 API usage；没有 usage 时可以用估算值，但日志字段必须标记来源。
- Baseline 和 ablation 没有真实实现前，只能标为 planned 或 unsupported，不能产出论文表格数据。
- 不同 agent/variant 对比时，必须使用相同 seed 下生成的 case schedule。

## Verification

- 首选测试命令：

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run pytest
```

- 如果本机 pytest 临时目录权限异常，可指定仓库内或其它可写目录作为 `--basetemp`。
- 正式实验前至少检查一次输出目录，确认存在 `config.yaml`、`metrics.jsonl`、`steps.jsonl`、`episodes.jsonl`、`experience_views.jsonl`、`graph_nodes.jsonl`、`graph_edges.jsonl`、`path_records.jsonl`。
## Stage 2 Pilot Tools

- 过滤分析本次实验 run：

```powershell
$env:PYTHONPATH='src'
python -B tools\analyze_stage2.py --run-dir runs --prefix stage2_pilot_real_s2c_ --output-dir runs\stage2_pilot_real_s2c_analysis --window 2
```

- 生成 pilot 报告：

```powershell
$env:PYTHONPATH='src'
python -B tools\build_stage2_report.py --analysis-dir runs\stage2_pilot_real_s2c_analysis --output runs\stage2_pilot_real_s2c_analysis\stage2_pilot_report.md --viewer tools\episode_log_viewer.html
```

- 查看单个 episode 决策过程：直接打开 `tools/episode_log_viewer.html`，拖入某个 run 目录下的 `steps.jsonl`。该页面会显示左侧 step 时间线和右侧 LLM prompt / raw response / parsed output / action result。

## Stage 2 Pilot Notes

- DeepSeek JSON mode 需要同时设置 `response_format={"type":"json_object"}`，并在 prompt 中明确要求 JSON 和给出 JSON 示例；`DeepSeekLLMClient` 里默认 `DEEPSEEK_MAX_TOKENS=2048`，避免较长 graph prompt 被截断。
- `LLMClient` 会记录 `last_trace`；`EpisodeRunner` 会把 `llm_input_messages`、`llm_output`、`llm_raw_response` 写入 `steps.jsonl`。后续改 agent prompt 时，要检查这些字段仍存在。
- 如果模型返回根级 action JSON，例如 `{ "name": "inspect", "args": {...} }`，当前 agent parser 会当作有效 action；如果返回 `{}` 或 malformed JSON，应记录为 `invalid_llm_output`，不要让实验进程中断。
- Windows 上如果 pytest 或 py_compile 因 `__pycache__`、`.pytest_cache`、`.venv` 权限失败，可用 `python -B` 做 import/smoke check；完整测试仍优先尝试 `$env:UV_CACHE_DIR='.uv-cache'; uv run pytest`。
- 对比不同 agent/variant 时，继续使用相同 seed 和 `case_schedule=shuffled_cycle`，并通过 `tools/analyze_stage2.py --prefix <run_id_prefix>` 避免历史 run 混入分析。

