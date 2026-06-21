# Project Instructions

## 0. 与 Leo 协作

- 通常使用中文回复 Leo；代码标识、API 名、论文术语可保留英文。
- 动手修改前说明三件事：假设、改动范围、验证方式。Leo 说“开始干”时，说明会用到的技术和文件后直接修改。
- 若需求有多种解释，列出差异并问清楚；需求已经明确时，不要反复确认。
- 如果 Leo 很有把握但判断可能有误，要直接提醒，并给出可验证的依据。
- 保持表达自然、简洁。汇报功能层面的变化，少堆实现细节。
- 避免使用夸张比喻、过度保证、讨好式表达和不自然的技术黑话。

## 1. 项目定位

ExperienceGraph 是一个 Python 3.11 / uv 项目，用于研究 LLM agent 的跨 episode 结构化经验管理。

核心任务：

- MyTextCraft 符号环境：可复现的 crafting/trading/exploration 任务。
- ExperienceGraph：把 episode trajectory 组织为 condition node、action edge、path record 和统计信息。
- 多 agent 对比：`scripted`、`react`、`reflexion`、`vector_trajectory`、`skill_library`、`graph`。
- JSONL 实验日志：为论文实验、可观测性检查、Stage 2 pilot 分析服务。
- 轻量 panel：查看 run summary、episode、graph。

重要边界：

- 不要把 `paper/paper_draft.md` 中的 `[PROJECTED]` 结果改写为已验证结论。
- Baseline、ablation 没有真实实现和日志证据前，只能标为 `planned` 或 `unsupported`。
- 修改实验相关代码时，对照 `paper/paper_draft.md` 和 `docs/experience_graph_system_design.md`，检查论文 claim、实现能力、日志字段是否一致。

## 2. 目录速览

- `src/experience_graph/core/`：数据模型、condition、experience 构建、序列化。
- `src/experience_graph/envs/`：环境 adapter、MyTextCraft 实现、oracle。
- `src/experience_graph/agents/`：各 agent、prompt、memory 工具。
- `src/experience_graph/graph/`：JSON graph store、organizer、retriever、merge/relevance。
- `src/experience_graph/runners/`：`EpisodeRunner`，连接 env、agent、retriever、organizer、logger。
- `src/experience_graph/evaluation/`：JSONL logger。
- `src/experience_graph/scripts/`：实验执行与结果分析 CLI。
- `src/experience_graph/panel/`：FastAPI panel。
- `world_cases/`：MyTextCraft 规则、active cases、task family cases、样例经验。
- `tests/`：环境、runner、LLM client、graph、panel、world cases 测试。
- `tools/`：Stage 2 分析、报告生成、HTML viewer、real pilot package。
- `docs/`：系统设计、风险记录、Stage 2 审计事项。
- `paper/`：论文草稿。
- `runs/`：实验输出；不要把历史 run 混入新分析。

注意：仓库里可能存在 `.pytest_cache`、`.uv-cache`、`.venv`、`.pytest-tmp-*`、`retry_trace_pytest_tmp`、`tmp_pytest_*` 等缓存或临时目录，其中部分在 Windows 上可能无读取权限。搜索代码时优先限定 `src tests docs tools world_cases paper`。

## 3. 开发原则

- 保持改动聚焦：只修改和当前任务直接相关的文件，不做无关重构。
- 匹配现有代码风格。不要为了个人偏好重排相邻代码。
- 工作区可能已有 Leo 的未提交改动。修改前看 `git status --short`，不要覆盖无关变化。
- 新增逻辑要有可验证结果：测试、fake-provider smoke、日志字段检查或分析脚本输出。
- 修改 prompt、parser、LLM client、runner 时，要检查 `steps.jsonl` 中的 `llm_input_messages`、`llm_output`、`llm_raw_response`、`llm_parse_error`、`llm_finish_reason`、`llm_attempts` 是否仍写入。
- 修改 MyTextCraft 任务、规则或 world cases 时，要同时考虑 oracle plan、action space、success condition、prompt manual 和 tests。
- 不要用真实 LLM 调用验证普通代码改动。付费调用只在 Leo 明确需要时执行，并遵守 smoke 规则。

## 4. 常用命令

首选完整测试：

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run pytest
```

若 Windows 临时目录权限异常，使用仓库内新的 basetemp：

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run pytest --basetemp .tmp\pytest-basetemp
```

只执行某个测试文件：

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run pytest tests\test_runner.py
```

scripted smoke run：

```powershell
uv run eg-run-experiment --agent scripted --episodes 1 --max-steps 12 --run-id smoke_scripted
```

fake-provider LLM smoke run：

```powershell
$env:EXPERIENCE_GRAPH_LLM_PROVIDER='fake'
uv run eg-run-experiment --agent react --episodes 1 --max-steps 3 --run-id smoke_fake_react
```

启动 panel：

```powershell
uv run eg-panel --run-dir runs --host 127.0.0.1 --port 8765
```

常规结果分析：

```powershell
uv run eg-analyze-results --run-dir runs --output-dir runs\analysis --window 50 --run-id-prefix <prefix>
```

Stage 2 过滤分析：

```powershell
$env:PYTHONPATH='src'
python -B tools\analyze_stage2.py --run-dir runs --prefix <prefix> --output-dir runs\<analysis_dir> --window 2
```

Stage 2 报告生成：

```powershell
$env:PYTHONPATH='src'
python -B tools\build_stage2_report.py --analysis-dir runs\<analysis_dir> --output runs\<analysis_dir>\stage2_pilot_report.md --viewer tools\episode_log_viewer.html
```

Stage 2 real pilot 的 Python runner 优先于 PowerShell runner：

```powershell
python -B tools\stage2_real_pilot_package\run_stage2_real_pilot.py --full-cycle --prefix stage2_pilot_real_s2g_ --parallel --max-workers 6
```

## 5. 实验安全规则

- 正式调用付费 LLM 前，必须执行 `EXPERIENCE_GRAPH_LLM_PROVIDER=fake` 或 `--agent scripted` smoke run。
- DeepSeek 相关 real pilot 默认需要 `DEEPSEEK_API_KEY`，并建议设置预算限制，例如 `--max-budget-rmb 50`。
- DeepSeek JSON mode 需要 `response_format={"type":"json_object"}`，prompt 中也要明确要求 JSON 并给出 JSON 示例。
- `DeepSeekLLMClient` 默认 `DEEPSEEK_MAX_TOKENS=4096`、retry cap `8192`；修改前确认不会导致 graph prompt 被截断。
- 主学习实验不要做 episode 级并行。`graph`、`reflexion`、`vector_trajectory`、`skill_library` 会在 episode 间更新状态；并行 episode 会改变实验定义。
- 不同 agent/variant 对比时，必须使用相同 seed 下生成的 `case_schedule`，默认推荐 `shuffled_cycle`。
- 分析历史 run 时必须使用 prefix 过滤，避免旧 run 混入统计。
- Warm-start 只复用 graph/memory 文件；`react` 和 `graph/no_graph_context` 属于无经验 baseline，通常不参与 warm-start。

## 6. Run 配置契约

每次实验必须写入可复现实验配置，至少包括：

- `run_id`
- `agent`
- `variant`
- `seed`
- `episodes`
- `case_schedule`
- `difficulty`
- `task_id`
- `max_steps`
- `top_k`
- `token_budget`
- `llm_provider`
- `llm_model`
- `llm_max_tokens`
- `llm_retry_max_tokens`
- `llm_retries`
- `continue_after_env_failure`
- `cross_task_mode`
- `warm_start_run`

`run_experiment.py` 的主要参数：

- `--agent`: `scripted`、`react`、`reflexion`、`vector_trajectory`、`skill_library`、`graph`
- `--variant`: `full`、`no_exploration`、`no_statistics`、`random_retrieval`、`no_node_merging`、`no_failure_preconditions`、`no_graph_context`
- `--cases`: 默认 `world_cases/textcraft_cases.yaml`，task family 可传目录 `world_cases/task_families`
- `--rules`: 默认 `world_cases/textcraft_rules.yaml`
- `--difficulty`: `all`、`easy`、`medium`、`hard`、`impossible`
- `--task-id`: 默认 `diamond_set`，多任务可用 `all`
- `--case-schedule`: `ordered`、`shuffled_cycle`、`random`
- `--resume`: 从已有 `metrics.jsonl` 数量继续，并清理未完成 episode 的 step/view/budget rows
- `--allow-config-mismatch`: 只在明确接受配置变化时使用
- `--warm-start-run`: 从已有 run 复制 graph/memory 文件

## 7. 日志字段契约

正式实验前至少检查输出目录存在：

- `config.yaml`
- `metrics.jsonl`
- `episodes.jsonl`
- `steps.jsonl`
- `experience_views.jsonl`
- `graph_nodes.jsonl`
- `graph_edges.jsonl`
- `path_records.jsonl`

每个 episode 必须记录：

- `episode_id`
- `case_id`
- `task`
- `difficulty`
- `seed`
- `agent`
- `variant`
- `success`
- `steps`
- `failure_reason`
- `oracle_failure_reason`
- `graph_nodes`
- `graph_edges`
- `graph_paths`
- `dormant_edges`
- `added_nodes`
- `added_edges`
- `updated_edges`
- `added_paths`
- `llm_usage_delta`
- `llm_usage_cumulative`

每个 step 必须记录：

- `action`
- `ok`
- `done`
- `episode_done`
- `episode_success`
- `episode_terminal_reason`
- `failure_reason`
- `reward`
- `cost`
- `experience_view_id`
- `candidate_paths`
- `experience_view_tokens`
- `prompt_diagnostics`
- `llm_usage_delta`
- `llm_model`
- `llm_input_messages`
- `llm_output`
- `llm_raw_response`
- `llm_parse_error`
- `llm_finish_reason`
- `llm_retry_count`
- `llm_max_tokens`
- `llm_attempts`

LLM token 统计优先使用 API usage；没有 usage 时可以用估算值，但日志字段必须保留 `token_source`。

如果模型返回根级 action JSON，例如 `{ "name": "inspect", "args": {...} }`，agent parser 应视为有效 action。若返回 `{}` 或 malformed JSON，应记录 `invalid_llm_output`，实验进程不能因为 parser 错误中断。

## 8. MyTextCraft 与 world cases

- `world_cases/textcraft_rules.yaml` 是规则来源。
- `world_cases/textcraft_cases.yaml` 是当前 diamond-set 主套件。
- `world_cases/task_families/` 包含 8 个 task family，测试期望可加载 96 个 cases。
- `docs/mytextcraft_task_standard.md` 是新增 task family 的统一标准。
- 每个 case 需要包含稳定 `id`、`difficulty`、`task`、`initial_state`、`oracle`、`expected_experience`。
- 后续主评测默认应避免 `impossible` / no-route cases；这类 case 保留为 diagnostic，并在分析中区分 `goal_achieved` 与 `case_resolved`。
- 任务相关变更要确认 `CaseOracle` reference plan 与 expected solvability 一致。
- 新 task family 不能只加 YAML；还要检查 `MyTextCraftAdapter.available_actions()`、`is_success()`、`step()`、`MYTEXTCRAFT_ACTION_GUIDE`、`report_impossible()` 是否 task-aware。
- prompt 不应暴露 hidden facts。已有测试和 Stage 2 validation 会检查 `prompt_hidden_facts=false`。

## 9. Agent 与 Graph 约束

- `ReActAgent`：无跨 episode 内部记忆。
- `ReflexionAgent`：使用自然语言 reflection memory，不使用 graph context。
- `VectorTrajectoryAgent`：基于相似 trajectory 的扁平记忆，不把结果说成 graph。
- `SkillLibraryAgent`：复用 action-sequence skill，不建条件路径图。
- `ExperienceGraphAgent`：可使用 graph retrieval，并可生成 novel candidate path。
- `graph/no_graph_context` 通过 `top_k=0` 实现，是 ExperienceGraphAgent 无 graph candidate path 的 ablation。
- `GraphRetriever` 输出固定预算的 `ExperienceView`，不要把完整图谱塞进 prompt。
- `GraphOrganizer` 的 node merging、failure precondition learning、dormant edge 状态都属于实验变量，修改时要保留 variant 可区分性。

## 10. 论文与报告写作

- 报告真实实验结果时，以 `runs/<analysis_dir>/` 下 CSV、PNG、report、JSONL 审计为依据。
- 不要把 Stage 2 pilot 的 5 episode 小样本写成最终主实验结论。
- 如果修改论文中方法描述，需要确认工程实现已经支持对应机制；未实现内容要标为 planned、future work 或 limitation。
- 引用图增长、token、success rate、ablation 前，检查分析脚本是否按 prefix 过滤了本次 run。
- `tools/episode_log_viewer.html` 可用于查看单 episode 的 step timeline、prompt、raw response、parsed output、action result。

## 11. 推荐工作流程

1. 看 `git status --short`，确认哪些文件已有变化。
2. 阅读相关模块和测试，不扩大到无关目录。
3. 向 Leo 说明假设、文件范围、验证方式。
4. 修改直接相关文件。
5. 执行合适验证：单测、完整 pytest、scripted smoke、fake LLM smoke、Stage 2 validation 中的一种或多种。
6. 汇报功能变化、验证结果、未验证风险。

对小型文档修改，可用阅读检查代替完整测试；对 runner、env、agent、graph、LLM client、analysis 脚本修改，应至少执行相关测试文件。


