从一个 agent 的视角，它进入 step loop 之后，每一步大概经历的是这个过程：

```text
观察当前状态
  -> 提取当前条件
  -> 从 ExperienceGraph 检索相关经验路径
  -> 组合成 AgentInput
  -> agent 读当前状态 + 可选动作 + 候选经验路径
  -> agent 生成/选择计划
  -> agent 输出 next_action
  -> 环境执行 action
  -> 得到新 observation / reward / failure_reason / revealed_conditions
  -> 记录这个 step
  -> 如果成功或失败则退出，否则进入下一步
```

具体在代码里，step loop 在 [episode_runner.py](C:/Bernie/DevG/git/TriDelta/GraphDBAI/src/experience_graph/runners/episode_runner.py:84)：

```python
for step_index in range(self.max_steps):
```

**1. 我先看到当前环境状态**

每一步开始时，runner 手里有当前的 `observation`。这个 observation 包含：

- 当前 inventory
- 当前 environment
- 当前 location
- 当前 step_count
- 当前 case_id

也就是 agent 在这个 step 能看到的世界状态。

**2. 环境把当前状态抽象成 conditions**

代码在这里：

```python
conditions = self.env.extract_conditions(observation)
```

这一步由 [textcraft.py](C:/Bernie/DevG/git/TriDelta/GraphDBAI/src/experience_graph/envs/textcraft.py:88) 完成。

它会把 raw state 转成更适合图检索的条件，比如：

```text
inventory.diamond >= 24
inventory.emerald >= 10
inventory.crafting_table == True
tool.pickaxe_level == 3
environment.nearby_mine == True
environment.village_has_armorer == unknown
```

这一步相当于：把“完整状态”压缩成“决策相关条件”。

**3. retriever 从经验图里找候选路径**

接着 runner 调用：

```python
view = self.retriever.retrieve(task, observation, conditions)
```

这会进入 [retriever.py](C:/Bernie/DevG/git/TriDelta/GraphDBAI/src/experience_graph/graph/retriever.py:16)。

从 agent 的视角看，这一步就是：

> “根据我现在的任务和状态，图里有没有过去类似情况下成功过的路径？”

retriever 会遍历已有 `PathRecord`，筛选同一个 task 的路径，然后检查路径里每条 edge 的 hard preconditions：

- 如果条件满足，路径可能是 `available`。
- 如果需要未知信息，路径可能是 `needs_info`。
- 如果条件不满足或 edge dormant，路径就是 `blocked`。

然后它会给候选路径排序：

1. 可用路径优先。
2. 成功率高的优先。
3. 平均步数少的优先。

最后返回一个 `ExperienceView`。这个 view 里面有：

- 当前相关 conditions
- candidate paths
- 每条路径的 success rate
- attempts
- avg steps
- missing conditions
- actions preview
- evidence，比如 `3/5 successes`

**4. runner 把所有信息打包成 AgentInput**

代码在 [episode_runner.py](C:/Bernie/DevG/git/TriDelta/GraphDBAI/src/experience_graph/runners/episode_runner.py:88)：

```python
agent_input = AgentInput(
    task=task,
    observation=observation,
    available_actions=self.env.available_actions(observation),
    experience_view=view,
    step_budget_remaining=self.max_steps - step_index,
)
```

所以 agent 在每一步真正收到的是：

- 当前任务
- 当前 observation
- 当前可选动作列表
- ExperienceGraph 检索出来的经验视图
- 还剩多少 step budget

这就是 agent 的“决策输入”。

**5. agent 根据输入选择下一步动作**

代码调用：

```python
output = self.agent.act(agent_input)
```

如果是 ExperienceGraph agent，会进入 [experience_graph_agent.py](C:/Bernie/DevG/git/TriDelta/GraphDBAI/src/experience_graph/agents/experience_graph_agent.py:15)。

这个 agent 会把下面这些内容发给 LLM：

- task
- observation
- available actions
- exploration 是否开启
- graph candidate paths
- 每条 candidate path 的适用性、成功率、尝试次数、平均步数、缺失条件、动作预览等
- warnings / summaries
- 剩余 step budget

system prompt 里要求它：

```text
If exploration is enabled, first propose one novel candidate path,
then compare it with graph candidate paths.
Return JSON with novel_candidate_path, selected_strategy,
selected_plan, next_action, reason, confidence.
```

也就是说，agent 每一步不是单纯选动作，而是：

1. 可以先提出一个新的探索路径。
2. 再和图里检索到的历史路径比较。
3. 选择一个策略或计划。
4. 输出当前 step 真正要执行的 `next_action`。

**6. 如果 agent 没有输出合法动作，这个 episode 失败**

runner 会检查：

```python
if output.action is None:
    failure_reason = ...
    break
```

也就是说，如果 LLM 输出格式不对、没有 action，当前 episode 就中断。

**7. 环境执行 agent 选择的 action**

如果 action 存在，runner 调用：

```python
result = self.env.step(output.action)
```

这一步会进入 MyTextCraft 环境，根据 action 类型执行不同逻辑：

- `craft(...)`
- `gather(...)`
- `mine(...)`
- `move_to(...)`
- `inspect(...)`
- `explore(...)`
- `trade(...)`

环境返回 `StepResult`，里面包括：

- `ok`: 动作是否成功
- `observation`: 执行动作后的新观察
- `reward`
- `done`: 任务是否完成
- `failure_reason`
- `revealed_conditions`
- `state_delta`

比如 agent 执行：

```text
trade(armorer, emerald, diamond_set)
```

但当前还不知道村庄有没有 armorer，那么环境会返回：

```text
ok = False
failure_reason = "armorer_unknown"
```

**8. runner 记录这个 step**

每一步都会被保存成 `StepRecord`：

```python
StepRecord(
    step_index=step_index,
    observation_before=before,
    experience_view_id=view.view_id,
    agent_thought=output.rationale,
    action=output.action,
    result=result,
    observation_after=observation,
)
```

同时写入 `steps.jsonl`，包括：

- step index
- experience view id
- candidate path 数量
- action
- agent rationale
- confidence
- ok / done / reward
- failure reason
- revealed conditions
- token usage

所以每一步都是可回放、可分析的。

**9. 根据结果决定是否继续循环**

执行完 action 后有三个可能：

- `result.done == True`：任务完成，episode 成功，退出 loop。
- `result.ok == False`：动作失败，episode 失败，退出 loop。
- 否则：更新 observation，进入下一 step。

代码：

```python
if result.done:
    success = True
    break

if not result.ok:
    failure_reason = result.failure_reason
    break
```

如果一直没成功也没失败，但 step 用完了，就是：

```python
failure_reason = "step_budget_exhausted"
```

**10. episode 结束后，经验才被写回图**

这点很重要：graph 不是每一步立刻更新，而是 episode 结束后统一整合。

代码在 [episode_runner.py](C:/Bernie/DevG/git/TriDelta/GraphDBAI/src/experience_graph/runners/episode_runner.py:171)：

```python
experience = self.builder.build(...)
update = self.organizer.integrate(experience)
```

也就是说：

- step loop 中，agent 使用旧图做决策。
- episode 结束后，整条 trajectory 被整理成 `ExperienceRecord`。
- `GraphOrganizer` 把这条轨迹合并进图。
- 下一 episode 才能用到这次新经验。

用 agent 视角一句话总结就是：

> 每个 step 中，我先观察当前状态，然后系统帮我从经验图里找过去相似或可用的路径；我把这些路径和自己新生成的探索计划进行比较，选择下一步动作；环境执行动作并返回结果；如果任务没结束也没失败，我带着新状态继续下一步。整个 episode 结束后，我这次走过的路径才会被整理进 ExperienceGraph，供以后使用。


