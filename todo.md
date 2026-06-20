# ExperienceGraph 实验计划与进度记录

本文档用于把 `paper/paper_draft.md` 中所有 `[PROJECTED]` 实验声明，转化为可执行、可断点续跑、可复现、可控制预算的实验计划。它同时记录当前实现进度，避免在日志、baseline 或分析脚本未准备好时直接调用付费 LLM。

## 0. 当前状态

### 0.1 已完成

- 已有 TextCraft 环境雏形，覆盖 crafting、mining、trading、inspection、exploration、多难度 case 和 hidden facts。
- 已有核心 ExperienceGraph 数据结构：nodes、edges、paths、hard preconditions、success/failure stats、dormant edge。
- 已有基础 agent：`scripted`、`react`、`graph`。
- 已有实验日志文件：`episodes.jsonl`、`metrics.jsonl`、`experience_views.jsonl`、`graph_nodes.jsonl`、`graph_edges.jsonl`、`path_records.jsonl`、`merge_decisions.jsonl`。
- 已新增更详细的 per-step 日志规划：正式实验必须产出 `steps.jsonl`。
- 已建立项目级 `AGENTS.md`，明确中文沟通、实验前 smoke test、不得把 `[PROJECTED]` 写成 verified。
- 已完成初步实验入口扩展方向：seed、difficulty、variant、token budget、case schedule、LLM usage 都应进入配置和日志。

### 0.2 当前缺口

- `Reflexion`、`VectorTrajectory`、`SkillLibrary` baseline 尚未完整实现。
- `ExperienceGraphAgent` 目前还不是完整的 explore-then-compare：缺少独立生成 novel candidate path 再与 retrieved paths 比较的流程。
- `EG - no exploration` 和 `EG - no decay` 消融尚未完全可测。
- TextCraft 目前主要依赖固定 case schedule；seed 还没有真正生成 procedural initial-state distribution。
- 节点合并目前主要是 deterministic rule-based merge；论文中的 LLM semantic merge 还没实现。
- 还缺统一分析脚本：主表、学习曲线、消融表、图增长、token cost、case study、failure mode 统计。
- 当前 pytest 在本机存在临时目录权限问题，不能只依赖完整 pytest 作为验证入口；需要 smoke run + syntax check 作为补充。

## 1. 总体原则

### 1.1 不直接大规模烧钱

正式调用 DeepSeek 前，必须先完成：

1. baseline 和 ablation 开关可运行；
2. 日志字段齐全；
3. fake/scripted smoke run 成功；
4. 分析脚本能从 smoke 日志生成结果文件；
5. 预算保护和断点续跑生效。

### 1.2 当前预算

- 真实 LLM provider：DeepSeek。
- 模型：`deepseek-v4-flash`。
- 当前预算上限：10 RMB 以内。
- 需要在 `config.yaml`、`metrics.jsonl` 和最终报告中记录实际使用模型。
- 每次真实实验必须记录 token 使用；如果 API usage 不可用，则记录估算 token，并标明 `token_source=estimated`。

### 1.3 结果写作规则

- 所有 `[PROJECTED]` 数字在真实实验完成前不能改成结论。
- 如果 pilot 规模不足，报告中必须写成 pilot observation，不能写成 final claim。
- 如果真实结果不支持 draft 中的预测，要改论文叙述，不强行贴合预测值。

## 2. 最终需要替换的 projected 内容

### 2.1 主要性能声明

需要替换的预测声明：

- ExperienceGraph 在 episode 300 达到约 85% success rate。
- ReAct 和 Reflexion 分别约为 45% 和 65%。
- ExperienceGraph 需要约少 40% episodes 达到稳定性能。
- ExperienceGraph 比最强 baseline 高 20+ percentage points，并约快 40% 收敛。

需要数据：

- Easy、Medium、Hard 的 `SR@300`。
- Medium 的 `CS`，即滑动窗口成功率首次达到 80% 的 episode。
- Medium 的 `SE`，即成功 episode 平均步数。
- 每个方法 5 seeds 的 mean/std。

### 2.2 主对比表

最终表格目标：

| 方法 | SR@300 Easy | SR@300 Medium | SR@300 Hard | CS Medium | SE Medium |
|---|---:|---:|---:|---:|---:|
| ReAct | actual | actual | actual | actual | actual |
| Reflexion | actual | actual | actual | actual | actual |
| VectorTrajectory | actual | actual | actual | actual | actual |
| SkillLibrary | actual | actual | actual | actual | actual |
| ExperienceGraph | actual | actual | actual | actual | actual |

### 2.3 学习曲线

需要生成：

- Medium difficulty 上 0-300 episodes 的 sliding-window success rate。
- window size：50 episodes。
- 5 seeds 的 mean 曲线和 standard error band。
- checkpoint：50、100、150、200、250、300。
- paired significance test：ExperienceGraph vs each baseline。

### 2.4 消融研究

最终需要的 variants：

| 变体 | 当前状态 | 备注 |
|---|---|---|
| EG full | 部分可运行 | 需补完整 explore-then-compare |
| EG - no exploration | 待实现 | 依赖 novel candidate path 机制 |
| EG - no statistics | 已规划/部分实现 | 检索展示时隐藏 success stats |
| EG - no node merging | 已规划/部分实现 | 必须确认不会复用相同 node id |
| EG - no failure edges | 已规划/部分实现 | 关闭 failure precondition 学习 |
| EG - no decay | 待实现 | 需要先实现 stats decay |
| EG - random retrieval | 已规划/部分实现 | 随机排序候选路径 |
| EG - no graph context | 已规划/部分实现 | 作为额外 sanity baseline |

### 2.5 图增长和 prompt budget

需要在 Medium ExperienceGraph runs 中记录：

- episode 50、100、150、200、300 的 nodes、edges、paths、dormant_edges。
- 每个 episode 新增节点数、边数、路径数。
- experience view token estimate。
- 实际 LLM prompt tokens。
- 图是否在 150-200 episodes 后接近稳定。

### 2.6 Token cost

需要统计：

- tokens per episode。
- tokens per successful episode。
- total tokens for 300 episodes。
- estimated RMB cost。
- 每个方法的 raw cost 和 success-normalized cost。

### 2.7 定性分析

需要从真实日志中抽样生成：

- 路径发现案例研究。
- failure mode 分析。
- node merge quality 分析。

## 3. Stage 1：工程补齐，不调用真实 LLM

目标：让所有实验条件能 fake/scripted 跑通，并能生成分析产物。

### 3.1 Baseline 实现

#### ReAct

状态：已有基础实现。

需要检查：

- 不使用跨 episode memory。
- 不读取 graph candidate paths，或在 no_graph_context 下公平运行。
- 记录 LLM call 和 token。

#### Reflexion

目标：实现 Shinn et al. 风格 flat natural-language memory baseline。

设计：

- 每个 episode 结束后写一条 reflection。
- 后续 prompt 中提供最近 K 条或按简单相关性筛选的 reflections。
- memory 类型是 append-only text list，不包含图结构和 path-level statistics。
- 持久化文件：`agent_memory_reflexion.jsonl`。
- 支持 resume：重新启动时读回 memory 文件。

#### VectorTrajectory

目标：实现 embedding-free 的 lightweight trajectory retrieval baseline，必要时后续再接 embedding。

设计：

- 将历史 trajectory 序列化为文本。
- 用 current state condition overlap / bag-of-conditions 相似度检索 top-K。
- prompt 中展示相似历史轨迹，但不提供 graph node/edge structure。
- 持久化文件：`agent_memory_vector_trajectory.jsonl`。
- 日志记录 retrieved trajectory ids 和 similarity scores。

#### SkillLibrary

目标：实现 Voyager-style 的简化 action-sequence library baseline。

设计：

- 成功 episode 后保存 action sequence 作为 skill。
- skill 按 task、difficulty、initial condition signature 建索引。
- 决策时检索最相近 skill，并让 LLM 选择下一步或改写。
- 不维护条件路径图和 edge/path success stats。
- 持久化文件：`agent_memory_skill_library.jsonl`。

### 3.2 ExperienceGraph 补齐

需要完成：

- 独立生成 novel candidate path。
- 检索 graph candidate paths。
- compare novel path 与 graph paths。
- 输出 selected strategy、next_action、reason、expected_next_condition。
- 支持 no-exploration 消融：不生成 novel candidate path。
- 支持 no-statistics 消融：不向 LLM 展示成功率和 attempts。

### 3.3 可断点续跑

必须支持：

- 如果 run 目录已存在，读取 `config.yaml`。
- 从 `metrics.jsonl` 判断已完成 episode 数。
- 从图文件恢复 `JsonGraphStore`。
- 从 baseline memory 文件恢复 agent memory。
- 继续写后续 episode，不覆盖旧日志。
- 如果配置不一致，拒绝 resume，除非显式传入 override。

### 3.4 预算保护

必须支持：

- `--max-budget-rmb 10`。
- `--input-price-per-million` 和 `--output-price-per-million` 可配置。
- 每个 episode 后估算累计成本。
- 接近预算时写入 `budget_stop.json` 并停止。
- 停止原因进入最终 summary。

### 3.5 分析脚本

新增脚本建议：`src/experience_graph/scripts/analyze_results.py`。

需要输出：

- `analysis/summary_tables.md`
- `analysis/main_comparison.csv`
- `analysis/ablation.csv`
- `analysis/graph_growth.csv`
- `analysis/token_cost.csv`
- `analysis/learning_curve_medium.csv`
- `analysis/figures/learning_curve_medium.png`
- `analysis/figures/graph_growth.png`
- `analysis/figures/token_cost.png`
- `analysis/experiment_writeup.md`

## 4. Stage 2：Smoke Test，不调用真实 LLM

目标：确认所有 agent、variant、日志、分析脚本能跑通。

### 4.1 Fake smoke matrix

运行：

- agents：`react`、`reflexion`、`vector_trajectory`、`skill_library`、`graph`
- variants：graph 至少跑 `full`、`no_statistics`、`random_retrieval`、`no_node_merging`、`no_failure_preconditions`、`no_graph_context`
- episodes：每个条件 2-3 episodes
- provider：`fake`
- difficulty：`easy` 和 `medium`

成功标准：

- 所有 run 目录有完整 `config.yaml`、`metrics.jsonl`、`steps.jsonl`。
- 分析脚本能读入所有 run 并生成表格。
- 不出现缺字段、JSON parse error、resume 覆盖旧日志。

### 4.2 Scripted smoke

目的：确认环境和图更新逻辑不依赖 LLM。

运行：

- `scripted/full`
- `scripted/no_node_merging`
- `scripted/no_failure_preconditions`

成功标准：

- positive oracle case 成功。
- negative case 能记录 failure reason。
- graph files 能恢复并继续写。

## 5. Stage 3：DeepSeek-v4-flash 小规模 pilot，预算 10 RMB 内

目标：用真实 LLM 验证 prompt、动作选择、token usage 和成本估算。

### 5.1 Pilot 配置

建议配置：

- provider：DeepSeek。
- model：`deepseek-v4-flash`。
- budget：10 RMB hard stop。
- agents：先跑 `react`、`reflexion`、`graph`。
- difficulty：`medium`。
- seeds：1-2。
- episodes：每个 method 每 seed 10-20。
- max_steps：30。
- case_schedule：`shuffled_cycle`。

如果预算消耗低，再加入：

- `vector_trajectory`
- `skill_library`
- graph ablations：`no_statistics`、`random_retrieval`、`no_graph_context`

### 5.2 Pilot 成功标准

- 所有方法能完成 episode，不大量失败于 JSON 格式错误。
- token usage 能正常记录。
- cost estimate 合理。
- graph agent 能看到 candidate paths，且后续 episode 的 candidate path 数量增长。
- 分析脚本能生成 pilot 报告。

### 5.3 Pilot 输出

- `analysis/pilot_summary.md`
- pilot learning curve。
- 每个 agent 的 failure reason 分布。
- 每个 agent 的 token cost。
- 是否建议进入 full run 的结论。

## 6. Stage 4：可断点 full run

目标：在工程和 pilot 都确认后，执行接近论文规模的实验。

### 6.1 主实验

配置：

- methods：ReAct、Reflexion、VectorTrajectory、SkillLibrary、ExperienceGraph。
- difficulties：Easy、Medium、Hard。
- episodes：300。
- seeds：5。
- provider/model：记录实际使用模型。
- case schedule：同一 seed 下所有 method 使用同一 schedule。

产出：

- Main comparison table。
- Learning curves。
- Convergence speed。
- Step efficiency。
- Token cost。

### 6.2 消融实验

配置：

- difficulty：Medium。
- variants：full、no exploration、no statistics、no node merging、no failure edges、no decay、random retrieval。
- episodes：300。
- seeds：5。

前置条件：

- no exploration 和 no decay 必须先真实实现。

产出：

- Ablation table。
- Delta vs full。
- 对贡献声明的证据。

### 6.3 图增长实验

可复用 ExperienceGraph full logs。

产出：

- graph nodes/edges/paths by episode。
- prompt token by episode。
- dormant edge count。
- graph saturation 判断。

## 7. Stage 5：人工分析与论文写作材料

### 7.1 Failure mode annotation

分类建议：

- invalid action / precondition failure。
- exploration overhead。
- stale statistics。
- retrieval failure。
- LLM selection error。
- incorrect merge。
- environment dead end。
- JSON/schema failure。

产出：

- `analysis/failure_modes.csv`
- `analysis/failure_mode_summary.md`

### 7.2 Node merge quality annotation

采样：

- 至少 50 条 merge decisions。
- 如果日志足够，优先采样 100 条。

分类：

- correct merge。
- conservative non-merge。
- incorrect merge。
- incorrect non-merge。
- not applicable。

注意：

- 如果尚未实现 LLM semantic merge，则只能报告 deterministic merge quality，不能声称 LLM merge call ratio。

### 7.3 Case study

选择标准：

- 有明显路径演化。
- 同时出现 mining、trading 或 mixed strategy。
- 有失败路径转化为后续 precondition 的证据。

产出：

- `analysis/case_study.md`
- 图快照或路径统计表。

## 8. 最终交付文件

最终需要给论文合并使用的文件：

- `analysis/experiment_writeup.md`

建议结构：

1. Experimental Setup。
2. Implemented Baselines。
3. Main Results。
4. Learning Curves。
5. Ablation Study。
6. Graph Growth and Prompt Budget。
7. Token Cost Analysis。
8. Case Study。
9. Failure Mode Analysis。
10. Node Merging Quality。
11. Claim-Evidence Map。
12. Limitations and Deviations from Original Draft。

## 9. 当前执行顺序

当前从这里继续：

1. 补齐 `JsonGraphStore` resume 读取能力。
2. 实现 baseline agent：Reflexion、VectorTrajectory、SkillLibrary。
3. 补 ExperienceGraph novel path generation 和 no-exploration variant。
4. 补预算保护和 resume config 校验。
5. 补分析脚本和图表生成。
6. 跑 fake/scripted smoke matrix。
7. 跑 DeepSeek-v4-flash pilot。
8. 根据 pilot 成本和质量决定 full run 规模。

## 10. 暂不做或谨慎做的事项

- 暂不把 pilot 结果写成论文主结论。
- 暂不做真实 300 episode full run，直到 resume、budget stop、analysis pipeline 全部通过 smoke test。
- 暂不声称 node merge 有 LLM semantic judgment，除非代码实现并产生日志。
- 暂不声称 seed 控制 procedural distribution，除非实现 procedural case sampler。
- 暂不加入 LATS 或 test-time search baseline，除非主实验结果需要回应 reviewer 风险。
