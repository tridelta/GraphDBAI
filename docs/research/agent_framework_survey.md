# LLM 智能体框架与相关思路调研

更新时间：2026-06-20

本文面向 `ExperienceGraph: 基于经验图谱的 LLM 智能体增量学习框架`，梳理已有智能体框架、长期记忆方法、图结构方法、自动化 workflow 搜索与评测思路。

范围说明：

- 重点关注会在环境中行动、调用工具、执行多步规划、跨 episode 利用经验的 LLM agent。
- 同时覆盖研究方法和工程框架。前者影响论文定位，后者影响之后的系统实现。
- 这里的“图”包含 reasoning graph、workflow graph、knowledge graph、memory graph、agent computation graph。它们都相关，但不等同于 ExperienceGraph。

## 核心判断

ExperienceGraph 更适合被表述为：

> 面向开放任务环境的跨 episode 条件化路径记忆框架。它把历史轨迹合并为图，并基于当前状态检索、比较和复用可执行路径。

当前 proposal 里“首个将 LLM 智能体经验组织为条件路径图”的说法风险较高，因为已有很多图相关工作：

- 图式推理：Graph of Thoughts。
- 图式检索与记忆：GraphRAG、HippoRAG、A-MEM。
- 图式 agent 编排：LangGraph、Google ADK graph workflows。
- 可优化 agent 计算图：GPTSwarm / Language Agents as Optimizable Graphs。

更准确的贡献边界：

- 现有图方法多用于推理过程、文档知识、工具流程或静态 agent 拓扑。
- ExperienceGraph 关注的是环境交互产生的历史轨迹，把它们组织成带条件、带统计信号、可比较的任务路径。
- 关键点不是“用了图”，而是“把跨 episode 执行经验变成决策时可用的条件路径图”。

建议修改 proposal：

- 增加更强 baseline：Voyager-style skill library、GraphRAG/HippoRAG-style graph memory、A-MEM-style linked memory、LATS/search-only planning。
- 增加状态合并、路径统计、top-k 检索、路径新鲜度/衰减相关消融实验。
- 不建议把“随经验增加单调提升”作为假设。记忆检索错误、节点合并错误、探索带来的短期损失都会造成波动。更合适的说法是“在平均意义上提升样本效率和最终成功率”。
- 增加成本归一化指标。近期 agent evaluation 工作反复指出，只看准确率容易偏向高成本、多调用的 agent。

## 1. 单智能体行动与反思

| 工作 | 核心机制 | 与 ExperienceGraph 的关系 | 可指出的局限 |
|---|---|---|---|
| ReAct | 推理和行动交替进行。 | 适合作为无跨 episode 记忆 baseline。 | 默认不积累长期经验。 |
| Reflexion | 失败/成功后生成文字反思，并在后续任务中复用。 | 是最直接的 flat memory baseline。 | 记忆多为文本列表，缺少条件化路径结构。 |
| Voyager | Minecraft lifelong agent，包含自动课程、代码技能库、自我改进。 | MyTextCraft 与 Minecraft 相似，因此 Voyager 是重要对照。 | 存储的是 executable skills，不是带条件统计的多路径经验图。 |
| GITM | 面向 Minecraft 的文本知识、结构化动作、规划和记忆。 | 适合作为 embodied/game agent 相关工作。 | 更像 planner + memory system，不以轨迹图增量学习为重点。 |
| DEPS | Describe、Explain、Plan、Select 的开放世界多任务规划。 | 与 MyTextCraft 的开放任务规划关系较近。 | 关注计划选择，不强调跨 episode 图式经验积累。 |

对 proposal 的影响：

- ReAct 和 Reflexion 必须保留。
- 建议加入简化版 Voyager-style skill library，因为 MyTextCraft 的 Minecraft 来源很明显。
- GITM 和 DEPS 可放入 related work；如果完整复现成本过高，不一定要作为实验 baseline。

## 2. 推理时搜索与规划

| 工作 | 核心机制 | 与 ExperienceGraph 的关系 | 主要差异 |
|---|---|---|---|
| Tree of Thoughts | 同时探索多条 reasoning path，并用模型评估。 | 支持“行动前生成多个候选路径”的设计。 | 主要是单任务内搜索，不是长期经验存储。 |
| Graph of Thoughts | 把 LLM thought 表示成任意图，并允许聚合、变换、排序等操作。 | 是“图式推理”方向的重要相关工作。 | 图表示的是 thought 操作，不是环境历史轨迹。 |
| RAP | 把 LLM 当作 policy 和 world model，使用 MCTS 风格规划。 | 与探索-利用和环境模型有关。 | 重点在推理时搜索，不是持久经验图。 |
| LATS | 结合行动、规划、反思和 Monte Carlo Tree Search。 | 可作为强 search baseline。 | 提升的是任务内 deliberation，不是跨 episode 路径记忆。 |
| Tree Search for LM Agents | 在 web 环境中用 best-first tree search 提升 agent 表现。 | 说明 test-time search 对交互式任务有效。 | 关注当前任务搜索，不是历史经验图增长。 |
| Branch-and-Browse | Web agent 中使用树结构推理和跨 session action memory。 | 与“action memory”很接近，值得关注。 | 面向 web exploration，领域假设与 MyTextCraft 不同。 |

对 proposal 的影响：

- “先生成新路径，再和已有路径比较”应明确和 search/planning 工作区分。
- 推荐表述：ExperienceGraph 可以与推理时搜索互补，把有用轨迹沉淀为持久、状态条件化的路径图。
- 可加入以下对照：
  - ReAct only
  - ReAct + LATS-style search
  - ReAct + flat memory
  - ExperienceGraph without search
  - ExperienceGraph with search

## 3. 长期记忆与经验组织

| 工作 | 核心机制 | 与 ExperienceGraph 的关系 | 主要差异 |
|---|---|---|---|
| Generative Agents | 观察存储、记忆检索、反思合成、计划生成。 | 早期 memory + reflection + planning 架构。 | 面向社会模拟，不强调任务路径统计。 |
| MemoryBank | 带遗忘和强化机制的长期用户记忆。 | 可参考记忆衰减和重要性评分。 | 关注对话/用户记忆，不是动作轨迹。 |
| MemGPT / Letta | 把上下文视为分层内存，由 agent 显式管理。 | 对处理 context window 很有参考价值。 | 是记忆管理框架，不是路径图学习方法。 |
| HippoRAG | 用知识图谱和 Personalized PageRank 做长期记忆检索。 | 是强 graph-memory baseline 思路。 | 检索事实知识，不更新行动路径成功统计。 |
| GraphRAG | 构建图索引和社区摘要，用于 corpus-level 问答。 | 是必须引用的 graph retrieval prior。 | 面向文档分析，不面向环境交互。 |
| A-MEM | 动态记忆节点、语义链接和记忆演化。 | 与“自组织记忆图”很接近。 | 不是显式的条件路径规划和成功率统计。 |
| Agentic Memory / AgeMem | 学习由 agent 决定记忆写入、更新、检索等操作。 | 指向未来的 RL-trained memory policy。 | 更关注学习记忆操作，不提供可解释路径图。 |

对 proposal 的影响：

- 需要区分“有记忆”和“记忆能直接改善决策”。
- 建议增加两个 baseline：
  - `Flat Memory + vector retrieval`
  - `Graph Memory + graph retrieval`，但不使用 path-level statistics
- 核心问题可以改成：当任务存在重复且可替代路线时，条件路径图是否优于文本记忆、向量轨迹记忆和普通 linked memory？

## 4. 工程框架与 agent runtime

这些框架主要是实现工具，不是直接科学对手。需要提到它们，因为 reviewer 可能会把 ExperienceGraph 理解成“又一个 agent framework”。论文里应明确：ExperienceGraph 是方法，LangGraph/ADK/AutoGen 这类是 runtime 或 orchestration layer。

| 框架 | 当前定位 | 对本项目有用的点 | 为什么不是直接 baseline |
|---|---|---|---|
| LangGraph | 面向长期运行、状态化 agent 的低层 orchestration runtime，支持 persistence、memory、human-in-the-loop、streaming、deployment。 | 适合实现 graph workflow 和持久状态。 | 它的图是 workflow/control graph，不是 experience graph。 |
| OpenAI Agents SDK | 轻量 agent runtime，包含 agents、tools、handoffs、guardrails、sessions、tracing、sandbox agents。 | 适合做 tracing、sessions、受控工具执行。 | 不定义学习算法。 |
| AutoGen | 多智能体对话框架，包含 AgentChat、Core、Extensions、Studio。 | 后续可用于 planner、critic、graph organizer 多角色设计。 | 对话编排不等于轨迹记忆。 |
| CrewAI | Crews 和 Flows，用于协作 agent、memory、knowledge、observability、自动化。 | 可用于角色型实验。 | 更偏 workflow automation。 |
| LlamaIndex Agents | 工具调用 agent，与 RAG、workflow、query engine、graph/property index 集成。 | 如果 MyTextCraft 需要 RAG 或图存储集成，会比较方便。 | 主要是数据/RAG 框架。 |
| Google ADK | 开源 agent development kit，强调 graph workflows、sessions、memory、context management、evaluation、deployment。 | ADK 2.0 的 graph workflow 与协作 agent 值得关注。 | workflow graph 和 experience graph 不是同一个层次。 |
| Semantic Kernel | 面向企业 AI agent 的 middleware，强调 plugins、function calling、model integration。 | 适合抽象工具和函数调用。 | 不是学习型 agent 方法。 |
| AutoGPT | 用于构建、部署和运行 continuous AI agents 的平台，也有 block-based workflows 和 benchmark 历史。 | 作为自治 agent 工程历史参考。 | 不是具体的长期记忆学习算法。 |

实现建议：

- 研究原型建议用 Python + NetworkX 或轻量图存储，方法更透明。
- LangGraph 或 ADK 可以作为未来工程化方向；如果一开始使用它们，可能会增加变量，让实验解释变复杂。

## 5. 多智能体研究模式

| 工作 | 核心机制 | 与本项目关系 |
|---|---|---|
| CAMEL | 通过角色扮演式 communicative agents 进行协作。 | 如果后续加入 planner/evaluator 角色，可参考。 |
| AgentVerse | 动态多智能体协作与 emergent behavior。 | 说明多 agent 可能提升能力，也会引入协调成本。 |
| ChatDev | 把软件开发分成多个角色 agent 和阶段。 | 展示基于角色的任务分解。 |
| MetaGPT | 用 SOP 组织多 agent 协作。 | 可用于说明结构化 workflow 降低错误传播。 |

对 proposal 的影响：

- 第一版 ExperienceGraph 建议保持单智能体为主。
- Graph Organizer 可以作为模块存在，不必设计成独立 agent。这样实验更容易解释。

## 6. 自动化 agent 设计与 workflow 优化

| 工作 | 核心机制 | 与本项目关系 |
|---|---|---|
| GPTSwarm / Language Agents as Optimizable Graphs | 把 language agents 建模为可优化计算图，优化 prompts、nodes、edges。 | 重要 prior。需要区分 computation graph 与 experience graph。 |
| AFlow | 使用 MCTS 搜索代码表示的 agent workflow，并根据执行反馈改进。 | 与“从反馈中改进 workflow”相关。 |
| ADAS / Meta Agent Search | 由 meta-agent 编写并改进新的 agent 设计。 | 可作为 future work，范围比 ExperienceGraph 大很多。 |
| Agent Lightning | 将 agent 执行与 RL training 解耦，把轨迹转换为训练 transition。 | 如果以后要把 ExperienceGraph 轨迹变成训练数据，可参考。 |
| AgentGym / AgentEvol | 多环境 agent benchmark 与自演化方法。 | 可参考 benchmark 设计和轨迹数据组织。 |

对 proposal 的影响：

- ExperienceGraph 不是自动发现新 agent 架构。
- 它更像一个结构化经验层，未来可为 RL 或 workflow search 提供轨迹数据。

## 7. Benchmarks 与评测风险

| 工作 | 对 MyTextCraft 实验的提醒 |
|---|---|
| WebArena | 真实长程环境很难，强模型 agent 与人类仍有较大差距。 |
| OSWorld | 基于真实执行结果的评测很重要；GUI/computer tasks 暴露了 agent 能力缺口。 |
| SWE-bench | 真实任务需要仓库理解、工具使用和可靠验证。 |
| SWE-agent | Agent-computer interface 会显著影响表现，环境接口和 prompt 同样重要。 |
| AI Agents That Matter | 应报告成本、准确率、复现性、holdout 设计，避免过拟合 benchmark。 |

对 MyTextCraft 的建议：

- 设计 hidden test seeds/tasks，不只在 prompt 调试过的任务上评估。
- 同时报告 LLM 调用次数、token 成本、wall-clock time、成功率、成功 episode 步数。
- 环境要在 seed 下可复现，让不同 agent 面对可比任务。
- 增加失败类型统计：非法动作、错误子目标、过度探索、错误记忆检索、错误图合并、prompt 超长。

## 对 ExperienceGraph 的直接建议

### 仍然有价值的研究空白

- ReAct 没有长期记忆。
- Reflexion 存的是非结构化文字经验。
- Voyager 存的是技能，不是条件化多路径经验。
- GraphRAG、HippoRAG、A-MEM 组织的是知识或一般记忆，不是带成功统计的行动路径。
- LangGraph、ADK、GPTSwarm 的图多为控制图或计算图，不是环境轨迹经验图。

ExperienceGraph 可以重点强调：

- 状态条件化 preconditions。
- 指向同一目标的替代路线。
- 边/路径级成功与失败统计。
- 失败轨迹与不可行条件。
- 基于当前状态相似度和图可达性的检索。

### 建议弱化的表述

原表述：

> 首个将 LLM 智能体经验组织为条件路径图的框架

建议：

> 提出一种面向开放任务环境的条件化经验路径图，将跨 episode 的执行轨迹、状态条件和成功统计统一组织为可检索、可比较的决策记忆。

原假设：

> 随游戏次数增加，决策质量单调递增

建议：

> 在相同环境分布和预算下，ExperienceGraph 相比无记忆与扁平记忆方法具有更高样本效率和更好的最终成功率；在复杂任务上收益更明显。

原因：

- 记忆系统可能因为检索噪声而退步。
- 节点合并可能引入错误泛化。
- 探索会造成短期成功率波动。

### 推荐 baseline

高优先级：

- `ReAct`：无跨 episode 记忆。
- `Reflexion-style flat memory`：文字经验，top-k 检索。
- `VectorTrajectory`：把历史轨迹文本化/向量化，用 state/goal embedding 检索。
- `SkillLibrary`：存储成功 action sequences 或 macros，接近简化版 Voyager。
- `ExperienceGraph`：完整方法。

中优先级：

- `Graph memory without path stats`：有图检索，但不展示成功/失败统计。
- `LATS/search-only`：加强推理时搜索，但没有持久经验图。
- `A-MEM-style linked notes`：动态链接记忆节点，但没有显式路径语义。

消融实验：

- No state merge。
- No path statistics。
- No negative/failure edges。
- No exploration candidate generation。
- No path decay/freshness。
- Random retrieval instead of state-conditioned retrieval。
- Top-1 vs top-k path retrieval。

### 方法细节需要提前定义

- Node identity：节点到底是 action、subgoal、condition，还是 state abstraction？
- Path identity：什么情况下两条路径算同一条路径？
- State similarity：精确符号匹配、embedding similarity、规则特征，还是混合方法？
- Graph update timing：每个动作后更新、子目标完成后更新，还是 episode 结束后更新？
- Negative memory：失败路径是否存储？
- Statistics：使用原始成功率、Wilson interval、Bayesian beta posterior，还是 recency-weighted estimate？
- Retrieval budget：多少条候选路径进入 prompt？
- Conflict handling：高成功率路径违反 hard condition 时如何处理？

推荐路径评分：

- 使用 beta-binomial score，而不是原始成功率：
  - `score = (successes + alpha) / (attempts + alpha + beta)`
  - 初始可取 `alpha=1, beta=1`
  - 对采样不足的路径，可使用 lower-confidence-bound 排序

这样可以避免 `1/1` 成功路径总是压过 `30/40` 成功路径。

### 修订后的实验矩阵

主实验：

| Agent | Memory | Planning/search | 目的 |
|---|---|---|---|
| ReAct | 无 | one-step/tool loop | 基础 baseline |
| Reflexion | 文字经验 | one-step/tool loop | 扁平记忆对照 |
| VectorTrajectory | 轨迹文本/向量库 | 检索 top-k 历史轨迹 | 测试图结构是否必要 |
| SkillLibrary | 成功 macros | 复用/改写 action sequence | 对比 Voyager-style 记忆 |
| ExperienceGraph | 条件路径图 | 检索 + 比较 + 探索 | 完整方法 |

消融实验：

| Variant | 移除内容 |
|---|---|
| EG-no-stats | 路径成功/失败统计 |
| EG-no-merge | 状态/节点合并 |
| EG-no-failure | 失败路径和 negative edges |
| EG-no-explore | 执行前的新路径生成 |
| EG-no-decay | recency/freshness weighting |

指标：

- 每 50 episode window 的成功率。
- 达到目标成功率所需 episode 数。
- 成功 episode 的平均步数。
- 非法动作率。
- 每个成功 episode 的 LLM calls 和 token 成本。
- 图规模：nodes、edges、paths。
- 检索可执行率：检索出的路径是否满足当前 hard conditions。
- Path adoption rate。
- 首次非法动作后的恢复率。

## Sources

### Proposal 中已有的核心工作

- ReAct: https://arxiv.org/abs/2210.03629
- Reflexion: https://arxiv.org/abs/2303.11366
- Voyager: https://arxiv.org/abs/2305.16291
- GITM: https://arxiv.org/abs/2305.17144
- DEPS: https://arxiv.org/abs/2302.01560

### 推理时搜索与图式推理

- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- Graph of Thoughts: https://arxiv.org/abs/2308.09687
- RAP: https://arxiv.org/abs/2305.14992
- LATS: https://arxiv.org/abs/2310.04406

### 长期记忆与图记忆

- Generative Agents: https://arxiv.org/abs/2304.03442
- MemoryBank: https://arxiv.org/abs/2305.10250
- MemGPT: https://arxiv.org/abs/2310.08560
- HippoRAG: https://arxiv.org/abs/2405.14831
- GraphRAG: https://arxiv.org/abs/2404.16130
- A-MEM: https://arxiv.org/abs/2502.12110
- Agentic Memory / AgeMem: https://arxiv.org/abs/2601.01885

### 工程框架

- LangGraph: https://docs.langchain.com/oss/python/langgraph/overview
- OpenAI Agents SDK: https://openai.github.io/openai-agents-python/
- AutoGen: https://microsoft.github.io/autogen/stable/
- CrewAI: https://docs.crewai.com/
- LlamaIndex Agents: https://developers.llamaindex.ai/python/framework/understanding/agent/
- Google ADK: https://adk.dev/
- Semantic Kernel: https://learn.microsoft.com/en-us/semantic-kernel/overview/
- AutoGPT: https://github.com/Significant-Gravitas/AutoGPT

### 自动化 agent 与 workflow 改进

- GPTSwarm / Language Agents as Optimizable Graphs: https://arxiv.org/abs/2402.16823
- AFlow: https://arxiv.org/abs/2410.10762
- ADAS: https://arxiv.org/abs/2408.08435
- AgentGym: https://arxiv.org/abs/2406.04151
- Agent Lightning: https://arxiv.org/abs/2508.03680

### 评测与 benchmark

- WebArena: https://arxiv.org/abs/2307.13854
- OSWorld: https://arxiv.org/abs/2404.07972
- SWE-bench: https://arxiv.org/abs/2310.06770
- SWE-agent: https://arxiv.org/abs/2405.15793
- AI Agents That Matter: https://arxiv.org/abs/2407.01502

## 后续建议

1. 为 proposal 写一个短版 related-work table，列出 memory type、graph type、cross-episode learning、statistics、environment。
2. 改写 contribution statement，避免把“图”本身当成唯一创新。
3. 确定第一版要实现哪些 baseline：
   - 必做：ReAct、Reflexion、VectorTrajectory、ExperienceGraph。
   - 可选：SkillLibrary、LATS/search-only。
4. 在写 MyTextCraft 代码前，定义 ExperienceGraph schema 和 update algorithm。
5. 做一个 30-50 episode 的 pilot experiment，观察图检索出的路径在随机初始状态下是否可执行。



