
Last updated: 2026-06-21
可以这样表述：

目前 `retriever.py` 中的 `retrieve()` 主要依赖已有 `PathRecord` 与当前任务、图结构之间的精确匹配。节点合并也主要基于 `condition signature` 的规则化比较：只有当两个节点的条件集合在规范化后完全一致时，才会被视为同一个节点。这种方式虽然简单、可解释，但会限制 ExperienceGraph 的泛化能力。

实际运行中，两个节点可能表达的是同一个语义状态，但它们的条件写法不同。例如，一个节点写成 “has enough diamonds”，另一个节点写成 `inventory.diamond >= 24`；或者一个节点描述为 “village trading route available”，另一个节点包含 `environment.village_has_armorer == true` 和 `inventory.emerald >= 10`。如果只依赖精确 signature，这些语义相近或等价的节点可能无法合并，也无法在检索时相互复用。

更进一步，即使两个状态并不完全相同，它们也可能对应相似的策略选择。例如，当前状态和某个历史状态的资源数量、工具条件、环境位置略有差异，但它们都适合走“交易路线”或“先采矿再合成”的路径。在这种情况下，我们希望 retriever 不只返回完全匹配的路径，而是能够找到与当前状态语义上最相近、经验上最有参考价值的已有节点和路径。

因此，一个改进方向是为每个 node 或 path 构造语义表示，并生成 embedding 向量。节点 embedding 可以由节点的 `required conditions`、`suggested` 描述、节点 label、上下文信息等组成；路径 embedding 可以由整条路径的 action sequence、起点条件、终点条件、成功/失败统计和自然语言摘要组成。检索时，将当前 observation 或当前 condition set 也编码成同一向量空间的 query embedding，然后在历史 node/path embedding 中做近邻搜索，返回语义上最接近当前状态的候选节点或路径。

这种方式可以把检索从“精确结构匹配”扩展为“语义相似性检索”。它有几个潜在好处：

- 能复用表达不同但语义等价的经验。
- 能召回状态不完全相同但策略相关的历史路径。
- 能缓解 condition signature 过于严格导致的经验碎片化。
- 能让 ExperienceGraph 更接近“经验泛化”，而不是只记住完全重复的状态。
- 可以与现有 hard precondition 过滤结合：先用 embedding 召回相似路径，再用硬条件检查排除当前状态不可执行的路径。

可以写成方法设计里的版本：

> To improve retrieval beyond exact condition matching, we propose augmenting each graph node and path with a semantic embedding. For a node, the embedding is computed from its normalized conditions, label, suggested natural-language hints, and local graph context. For a path, the embedding summarizes the start conditions, action sequence, terminal condition, and accumulated outcome statistics. At retrieval time, the current observation and condition set are encoded into the same embedding space, and the retriever performs nearest-neighbor search over stored node/path embeddings. The top semantic matches are then filtered by hard preconditions and ranked by both similarity and success statistics. This allows ExperienceGraph to retrieve experiences that are not structurally identical to the current state but are semantically relevant, enabling reuse across paraphrased conditions, partially overlapping states, and similar strategy contexts.

## MyTextCraft task suite risk

The task-family suite now loads through `MyTextCraftAdapter`, and oracle reference plans are validated by `tests/test_world_cases_load.py`. The remaining risk is no longer basic multi-task loading; it is task-standard drift.

Current risks:

- `available_actions()` now uses `tasks.<task_id>.action_space` when present, but older suites still rely on the fallback path if a task omits action declarations.
- Many non-craft action semantics still live in Python branches inside `MyTextCraftAdapter.step()` and helper methods. Core `craft(...)` recipes have started moving to declarative rules.
- Prompt manuals are not yet generated from `visible_manual_refs`, so task-local rules are recorded in YAML but not fully used by LLM agents.
- Older suites include `impossible` cases. These are useful diagnostics, but future main evaluation should prefer solvable cases and separate `goal_achieved` from `case_resolved` if diagnostics are included.
- `run_experiment.py --task-id all` supports mixed schedules, but final comparison protocols still need a fixed case schedule across agents and seeds.

Recommended handling:

- Use `docs/mytextcraft_task_standard.md` as the source of truth for new task families.
- Keep no-route cases as optional diagnostics, not as part of the default main success-rate claim.
- Move recipe and action precondition definitions toward declarative YAML before adding many more tasks.
- Before real LLM calls, run scripted or fake-provider smoke runs for each new task family and verify that prompts expose only the selected task's visible manual, never unrelated recipes or hidden facts.
- Treat simplified Minecraft rules as MyTextCraft rules and mark any deviation from real Minecraft mechanics in rule notes, especially villager trading and route shortcuts.

