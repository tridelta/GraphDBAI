# ExperienceGraph: Incremental Learning for LLM Agents via Conditional Path Graphs

## Abstract

Large language model (LLM) agents have demonstrated strong reasoning and action capabilities in interactive environments. However, existing approaches either lack cross-episode memory (ReAct), store experiences as unstructured text lists (Reflexion), or accumulate isolated skills without modeling conditional dependencies between alternative paths (Voyager). This flat or fragmented experience organization prevents agents from systematically improving decision quality as task attempts accumulate. We observe that in multi-path environments, the key to sample-efficient learning is not merely *remembering* past trajectories, but *organizing* them into a structured, state-conditioned, and statistically grounded decision memory. Based on this insight, we propose ExperienceGraph, a framework that organizes cross-episode execution trajectories into a directed graph where nodes represent state-condition checkpoints and edges represent actions with hard preconditions, soft heuristics, and success/failure statistics. At decision time, the agent retrieves top-K feasible paths from the graph conditioned on the current state, generates a novel candidate path for exploration, and selects among them using both statistical evidence and LLM reasoning. Experiments on TextCraft, a text-based crafting environment with partial observability and multiple solution paths, show that ExperienceGraph achieves **<mark style="background-color: yellow;">[PROJECTED: 85% success rate at episode 300]</mark>** compared to **<mark style="background-color: yellow;">[PROJECTED: 45% for ReAct and 65% for Reflexion]</mark>**, while requiring **<mark style="background-color: yellow;">[PROJECTED: 40% fewer episodes]</mark>** to reach stable performance. Ablation studies confirm that graph structure, path-level statistics, and the explore-then-compare mechanism each contribute significantly to the overall gain.

## 1. Introduction

LLM-based agents have emerged as a promising paradigm for autonomous decision-making in complex interactive environments, with applications spanning game playing, embodied navigation, code generation, and dialogue systems (Yao et al., 2023; Wang et al., 2023a; Shinn et al., 2023). A central challenge in deploying these agents is enabling them to *learn from experience*: to improve systematically over repeated task attempts rather than reasoning from scratch each time.

Existing methods address this challenge with varying degrees of structure. ReAct (Yao et al., 2023) interleaves reasoning and action but maintains no persistent memory across episodes. Reflexion (Shinn et al., 2023) introduces natural-language reflections stored as a text buffer, enabling the agent to avoid repeated mistakes. Voyager (Wang et al., 2023a) builds a skill library of executable code snippets for Minecraft. While these methods represent progress along a spectrum from no memory to structured skills, they share a fundamental limitation: none of them model the *conditional relationships between alternative paths* toward the same goal. In environments where multiple routes exist—each viable only under specific state conditions—flat experience lists and isolated skills cannot capture which path is preferable *given the current situation*.

This limitation becomes critical in environments with three properties: (1) multiple feasible paths to the same goal, (2) state-dependent path viability (a path that works with an iron pickaxe fails without one), and (3) partial observability requiring exploration to resolve ambiguity. In such environments, an effective experience system must answer not just "what worked before?" but "what worked before *in a situation like this one*?"

We observe that execution trajectories across episodes naturally form a graph structure: shared subgoals become nodes, alternative actions become parallel edges, and success/failure counts become statistical annotations. This graph organization enables three capabilities that flat memory cannot provide: (1) state-conditioned retrieval of relevant paths, (2) quantitative comparison between alternatives, and (3) incremental refinement as new experiences merge into existing structure.

Based on this insight, we propose **ExperienceGraph**, a framework that organizes cross-episode execution experience into a conditional path graph. Each node represents a state-condition checkpoint (e.g., "possesses >= 24 diamonds and has access to a crafting table"). Each edge represents a transition action annotated with hard preconditions (machine-checkable), soft heuristics (natural-language guidance for the LLM), and beta-binomial success statistics. At decision time, the agent (1) generates a novel candidate path for exploration, (2) retrieves top-K feasible paths from the graph conditioned on the current state, and (3) selects among all candidates using both statistical evidence and LLM-based reasoning. After each episode, the Graph Organizer integrates the executed trajectory into the graph through a rule-based-then-semantic node merging algorithm and updates path statistics.

We evaluate ExperienceGraph on TextCraft, a text-based environment that preserves the core decision complexity of Minecraft—multi-step crafting, resource gathering, trading, and partial observability—while eliminating visual perception and motor control confounds. TextCraft provides controlled, reproducible experiments with configurable task complexity.

Our contributions are as follows:

- We propose a conditional path graph framework for organizing LLM agent experience, where nodes encode state preconditions, edges encode actions with layered conditions and statistics, and paths encode complete strategies that are retrievable, comparable, and incrementally refinable.
- We design an explore-then-compare decision mechanism that balances exploitation of high-success paths with generation of novel candidates, preventing premature convergence to suboptimal strategies.
- We demonstrate on TextCraft that structured graph-based experience significantly outperforms flat memory and memoryless baselines in both sample efficiency and final success rate, with ablations confirming the necessity of each design component.
- We introduce TextCraft as a lightweight benchmark for studying long-horizon, cross-episode learning in LLM agents, featuring partial observability, multiple solution paths, and configurable complexity.

## 2. Related Work

### 2.1 LLM Agents with Action and Reflection

A growing body of work equips LLMs with the ability to act in environments and learn from feedback. ReAct (Yao et al., 2023) established the paradigm of interleaved reasoning and action traces, demonstrating that explicit reasoning improves action selection. However, ReAct operates without persistent memory; each episode starts from a blank slate. Reflexion (Shinn et al., 2023) addresses this by generating natural-language self-reflections after task attempts and prepending them to future prompts. While this enables cross-episode improvement, the memory is an append-only text list without structure—the agent cannot efficiently compare alternative strategies or assess their relative success rates. Voyager (Wang et al., 2023a) takes a different approach in Minecraft by building a library of reusable code-based skills with an automatic curriculum. Skills are stored as executable functions, but the library does not model *when* to prefer one skill sequence over another based on the current state. GITM (Zhu et al., 2023) and DEPS (Wang et al., 2023b) further explore goal decomposition and plan selection in open-world games but do not maintain cross-episode path-level statistics.

Our work differs from these approaches by organizing experiences as a *graph of conditional paths* with quantitative success signals, enabling state-conditioned retrieval and principled comparison between alternative strategies.

### 2.2 Graph-Based Reasoning and Memory for LLM Systems

Graph structures have been applied to LLM systems in several distinct ways. Graph of Thoughts (Besta et al., 2023) models the reasoning process itself as a graph where thoughts can be combined and refined. GraphRAG (Edge et al., 2024) and HippoRAG (Gutiérrez et al., 2024) construct knowledge graphs from document corpora for retrieval-augmented generation. A-MEM (Wang et al., 2025) creates dynamically linked memory nodes with semantic associations. In the agent orchestration domain, LangGraph provides stateful workflow graphs, and GPTSwarm (Zhuge et al., 2024) models agent computation as optimizable graphs.

These works apply graphs to *reasoning processes*, *knowledge organization*, or *agent control flow*. ExperienceGraph operates in a different space: it graphs *execution trajectories from environment interactions*, annotating them with state preconditions and outcome statistics. The resulting structure is not a reasoning aid or knowledge base, but a decision memory that directly informs action selection.

### 2.3 Test-Time Search and Planning for Agents

Tree of Thoughts (Yao et al., 2023b) and LATS (Zhou et al., 2023) improve single-episode performance through deliberative search—exploring multiple reasoning or action paths within one task attempt. These methods enhance *within-episode* decision quality but do not persist learned strategies across episodes. RAP (Hao et al., 2023) uses the LLM as both policy and world model for MCTS-style planning.

ExperienceGraph is complementary to test-time search: it provides a persistent, cross-episode experience layer that search methods can leverage. An agent with ExperienceGraph starts each episode with prior knowledge of which paths tend to succeed, while search methods can further refine within-episode execution. Our experiments include an ablation that isolates the contribution of experience accumulation from within-episode deliberation.

## 3. Method

### 3.1 Overview

We consider a sequential decision-making setting where an LLM agent repeatedly attempts tasks in the same environment across multiple episodes. The agent observes a structured game state, selects actions from a finite action space, and receives a binary success signal at episode termination. Our goal is to design an experience accumulation mechanism that enables the agent to improve sample efficiency and final success rate over episodes.

ExperienceGraph consists of three components: (1) the **Experience Graph** $G = (V, E)$, a persistent directed graph encoding conditional paths; (2) the **Agent**, which performs explore-then-compare decision-making at each step; and (3) the **Graph Organizer**, which integrates new trajectories into the graph after each episode. Figure 1 illustrates the system architecture. Section 3.2 defines the graph structure, Section 3.3 describes the agent decision process, Section 3.4 details the graph update and node merging algorithm, and Section 3.5 presents the compression mechanism that bounds prompt size.

```
┌──────────────────────────────────────────────────┐
│              TextCraft Environment                 │
│     (State management / Action execution)         │
└─────────────────────┬────────────────────────────┘
                      │ observation + reward
                      ▼
┌──────────────────────────────────────────────────┐
│                 Agent (LLM)                        │
│  1. Receive current state                         │
│  2. Generate novel candidate path (explore)       │
│  3. Retrieve top-K paths from graph (exploit)     │
│  4. Select path via stats + reasoning             │
│  5. Execute next action from selected path        │
└─────────────────────┬────────────────────────────┘
                      │ trajectory + outcome
                      ▼
┌──────────────────────────────────────────────────┐
│            Graph Organizer                         │
│  - Integrate trajectory into graph                │
│  - Merge equivalent nodes                         │
│  - Update path statistics                         │
│  - Apply compression / pruning                    │
└──────────────────────────────────────────────────┘
```
*Figure 1: System architecture of ExperienceGraph.*

### 3.2 Experience Graph Structure

The experience graph $G = (V, E)$ is a directed weighted graph that grows incrementally as the agent accumulates execution experience.

**Nodes (State-Condition Checkpoints).** Each node $v \in V$ represents a set of state conditions that must hold at a particular decision point. Unlike a concrete state snapshot, a node abstracts over irrelevant state dimensions and captures only task-relevant preconditions:

$$v = (\text{id}, \text{label}, \text{type}, \text{required}, \text{suggested}, \text{stats})$$

where `required` is a machine-checkable condition set (e.g., `{diamond >= 24, crafting_table: true}`), `suggested` is a natural-language heuristic, and `stats` tracks `(attempts, successes)` for reaching this checkpoint. Nodes are typed as `goal` (terminal objective), `checkpoint` (intermediate milestone), or `observation` (information revealed by exploration).

This design is motivated by the need to match incoming states against stored experience: by abstracting to *conditions* rather than *exact states*, the graph remains compact and reusable across episodes with different initial configurations.

**Edges (Conditional Transitions).** Each edge $e \in E$ represents an action that transitions between checkpoints:

$$e = (\text{from}, \text{to}, \text{action}, \text{hard\_precondition}, \text{soft\_precondition}, \text{stats})$$

Edges carry two layers of conditions:
- **Hard preconditions**: JSON-format conditions checked programmatically. If unsatisfied, the edge is infeasible and excluded from retrieval.
- **Soft preconditions**: Natural-language hints presented to the LLM for reasoning (e.g., "more efficient when near a mine entrance"). These influence path preference but do not block execution.

The layered condition design separates what *must* hold (preventing invalid actions) from what *helps* (guiding intelligent selection), allowing the LLM to reason about tradeoffs while respecting hard constraints.

**Paths (Complete Strategies).** A path $P = [e_1, e_2, \ldots, e_k]$ is an ordered edge sequence where $e_i.\text{to} = e_{i+1}.\text{from}$, representing a complete strategy from an initial state to the goal. Path-level statistics are computed as:

$$\text{score}(P) = \frac{\text{successes}(P) + \alpha}{\text{attempts}(P) + \alpha + \beta}$$

where $\alpha = 1, \beta = 1$ provides a beta-binomial smoothing prior. When path-level statistics are insufficient (fewer than 5 attempts), the score falls back to the product of edge-level success rates with additive smoothing.

### 3.3 Agent Decision Process: Explore-Then-Compare

At each decision point within an episode, the agent executes a three-phase process designed to balance exploitation of known-good paths with exploration of potentially superior alternatives.

**Phase 1: Exploration — Generate a Novel Candidate.** The agent receives the current state and goal, and is prompted to propose a *new* path without seeing the graph. This ensures the LLM's generative reasoning capability is exercised independently of stored experience, preventing the graph from becoming a ceiling on strategy diversity.

**Phase 2: Exploitation — Retrieve Feasible Paths.** The Graph Organizer identifies the current node (or closest matching node) in the graph, computes all paths to the goal node, filters out paths whose edges have unsatisfied hard preconditions given the current state, ranks remaining paths by score, and returns the top-K paths (K=3 by default).

**Phase 3: Selection — Compare and Commit.** The agent receives (a) its self-generated novel path, (b) up to K graph-retrieved paths with their statistics, and (c) the current state. It then selects one path to follow, using both the quantitative success rates and its own reasoning about soft preconditions and the current state. The selected path's next action is executed.

This mechanism provides a principled exploration-exploitation balance: the novel candidate ensures the agent can discover paths not yet in the graph, while retrieved paths offer the accumulated statistical wisdom of prior episodes. The LLM serves as the arbitrator, capable of reasoning about when to trust statistics versus when to try something new (e.g., when the current state differs significantly from conditions under which the statistics were collected).

### 3.4 Graph Organizer: Trajectory Integration and Node Merging

After each episode, the Graph Organizer integrates the executed trajectory into the experience graph. This involves two subprocesses: node merging and statistics update.

**Trajectory Decomposition.** The raw trajectory (sequence of state-action pairs) is first decomposed into a sequence of checkpoint nodes connected by action edges. Checkpoint boundaries are determined by significant state changes (e.g., acquiring a key resource, entering a new area, or resolving an ambiguous observation).

**Node Merging Algorithm.** When a new checkpoint from the trajectory is proposed for insertion, the merging algorithm determines whether it should be unified with an existing node:

1. **Normalization**: Convert conditions to canonical form (e.g., `has_enough_diamond`, `diamond >= 24` → standardized representation).
2. **Exact Match**: If `required` conditions are identical after normalization, merge immediately; accumulate stats.
3. **Subsumption/Conflict Check**: If one condition set strictly includes the other, keep the more specific node. If conditions conflict (e.g., `armorer: true` vs. `armorer: false`), merging is forbidden.
4. **Semantic Judgment**: Only when rules 1–3 are inconclusive and semantic overlap exists, invoke the LLM to judge equivalence. The decision and rationale are logged.
5. **Create New Node**: If no match is found, insert as a new node.

This hierarchical approach minimizes LLM calls (expensive) by resolving most cases through cheap deterministic rules, invoking LLM judgment only for genuinely ambiguous cases.

**Statistics Update.** Upon merging, edge and path statistics are updated: `attempts += 1`, and `successes += 1` if the episode succeeded. All edges along the executed path receive updates, and the path itself is recorded for path-level tracking.

### 3.5 Graph Compression and Prompt Budget Control

A critical engineering challenge is preventing the experience graph from overwhelming the LLM's context window. We address this through bounded growth and active compression.

**Bounded Growth.** In TextCraft, the state space is finite and discrete (~10 resource dimensions with bounded values, ~5 environment dimensions). The number of meaningfully distinct checkpoints is bounded by the combinatorial space of relevant conditions. Empirically, we observe sublinear growth: **<mark style="background-color: yellow;">[PROJECTED: graph growth plateaus at approximately 40-60 nodes and 80-120 edges by episode 200, with new episodes primarily updating statistics rather than adding structure]</mark>**.

**Active Compression (Experience Distillation):**
- *Edge Pruning*: Edges with success rate below 10% after K ≥ 10 attempts are marked dormant and excluded from retrieval.
- *Statistical Decay*: Every 50 episodes, all statistics are multiplied by a decay factor (0.9), ensuring recent experience outweighs stale data.
- *Top-K Retrieval*: Only the top-K paths (ranked by score) are presented to the agent, regardless of total graph size.
- *Node Folding*: When all outgoing edges from a checkpoint lead to the same downstream node, the intermediate checkpoint is folded.

**Fixed Token Budget.** Regardless of graph size, the information presented to the LLM at each decision point is bounded by a fixed token budget (default: 1000 tokens). When the top-K paths exceed this budget, lower-ranked paths are summarized to single-line statistics. This guarantees that prompt length does not grow with experience accumulation.

## 4. TextCraft Environment

TextCraft is a text-based interactive environment designed to study cross-episode learning in LLM agents. It retains the core decision complexity of Minecraft—multi-step crafting dependencies, resource gathering, NPC trading, and partial observability—while eliminating visual perception and motor control as confounding variables.

**State Space.** The game state consists of approximately 10 dimensions organized into three categories:
- *Inventory*: quantities of resources (diamond, emerald, iron, wood, etc.) and equipped tools.
- *Environment*: nearby locations and features (village, mine, crafting table, biome).
- *Ambiguous*: partially observable information (e.g., whether a village has a specific type of villager) that requires exploration actions to resolve.

**Action Space.** Six parameterized actions: `mine(resource)`, `craft(item)`, `trade(villager, offer, want)`, `move_to(location)`, `explore()`, `gather(resource)`. Each action has explicit preconditions (e.g., mining diamond requires an iron or diamond pickaxe and proximity to a mine).

**Task Suite.** Three task difficulty levels:
- *Easy*: Craft an iron armor set (short path, few preconditions).
- *Medium*: Craft a diamond armor set (multiple viable paths—mining vs. trading—with ambiguous information).
- *Hard*: Craft an enchanted diamond armor set (long dependency chains, multiple prerequisite subgoals).

**Design Rationale.** TextCraft isolates the *planning and learning* capabilities of agents by providing: (1) deterministic state transitions given a seed, ensuring reproducibility; (2) partial observability via `ambiguous` fields, testing information-gathering behavior; (3) multiple viable paths to each goal, testing strategy selection; (4) configurable complexity via task difficulty levels. Compared to full Minecraft, TextCraft enables running 300 episodes × 5 seeds × 5 conditions in feasible compute time while maintaining decision-theoretic richness.

## 5. Experiments

### 5.1 Experimental Setup

**Baselines.** We compare ExperienceGraph against four baselines that span the spectrum of experience organization:

| Method | Memory Type | Decision Mechanism |
|--------|------------|-------------------|
| **ReAct** | None (episode-independent) | Interleaved reasoning + action |
| **Reflexion** | Flat text list (append-only reflections) | Reasoning informed by past reflections |
| **VectorTrajectory** | Vectorized past trajectories | Top-k trajectory retrieval by state similarity |
| **SkillLibrary** | Indexed successful action sequences | Skill retrieval and reuse/adaptation |
| **ExperienceGraph (Ours)** | Conditional path graph with statistics | Explore-then-compare with top-K retrieval |

All agents share the same LLM backbone (GPT-4), the same TextCraft environment, the same observation format, and the same action interface. The only difference is the internal experience organization and decision mechanism.

**Protocol.** Each condition is run for 300 episodes across each of the three task difficulty levels. Each configuration is repeated with 5 random seeds (controlling initial state distribution), yielding 5 × 300 = 1500 episodes per condition per task. All agents face identical initial state sequences within each seed to ensure direct comparability.

**Metrics.**
- *Success Rate (SR)*: Task completion rate within a sliding window of 50 episodes.
- *Convergence Speed (CS)*: Number of episodes required to achieve 80% success rate.
- *Step Efficiency (SE)*: Mean number of actions in successful episodes.
- *Cost Efficiency (CE)*: Mean LLM token consumption per successful episode.
- *Illegal Action Rate (IAR)*: Fraction of attempted actions that violate preconditions.

### 5.2 Main Results

> **<mark style="background-color: yellow;">⚠️ PROJECTED RESULTS — Pending experimental validation</mark>**

**Table 1: Main comparison on TextCraft (300 episodes, 5 seeds, mean ± std).**

| Method | SR@300 (Easy) | SR@300 (Medium) | SR@300 (Hard) | CS (Medium) | SE (Medium) |
|--------|:---:|:---:|:---:|:---:|:---:|
| ReAct | <mark style="background-color: yellow;">[PROJ: 72±5%]</mark> | <mark style="background-color: yellow;">[PROJ: 45±6%]</mark> | <mark style="background-color: yellow;">[PROJ: 25±7%]</mark> | <mark style="background-color: yellow;">[PROJ: N/A*]</mark> | <mark style="background-color: yellow;">[PROJ: 18±3]</mark> |
| Reflexion | <mark style="background-color: yellow;">[PROJ: 82±4%]</mark> | <mark style="background-color: yellow;">[PROJ: 65±5%]</mark> | <mark style="background-color: yellow;">[PROJ: 42±6%]</mark> | <mark style="background-color: yellow;">[PROJ: 180±25]</mark> | <mark style="background-color: yellow;">[PROJ: 14±3]</mark> |
| VectorTrajectory | <mark style="background-color: yellow;">[PROJ: 80±4%]</mark> | <mark style="background-color: yellow;">[PROJ: 62±5%]</mark> | <mark style="background-color: yellow;">[PROJ: 40±7%]</mark> | <mark style="background-color: yellow;">[PROJ: 195±30]</mark> | <mark style="background-color: yellow;">[PROJ: 15±3]</mark> |
| SkillLibrary | <mark style="background-color: yellow;">[PROJ: 85±3%]</mark> | <mark style="background-color: yellow;">[PROJ: 68±5%]</mark> | <mark style="background-color: yellow;">[PROJ: 45±6%]</mark> | <mark style="background-color: yellow;">[PROJ: 160±20]</mark> | <mark style="background-color: yellow;">[PROJ: 13±2]</mark> |
| **ExperienceGraph** | **<mark style="background-color: yellow;">[PROJ: 92±3%]</mark>** | **<mark style="background-color: yellow;">[PROJ: 85±4%]</mark>** | **<mark style="background-color: yellow;">[PROJ: 65±5%]</mark>** | **<mark style="background-color: yellow;">[PROJ: 100±15]</mark>** | **<mark style="background-color: yellow;">[PROJ: 10±2]</mark>** |

*ReAct does not converge to 80% on Medium tasks within 300 episodes.

**<mark style="background-color: yellow;">Key Observations (Projected):</mark>**
1. ExperienceGraph achieves the highest final success rate across all difficulty levels, with the advantage increasing as task complexity grows (from +7% on Easy to +20% on Hard vs. best baseline).
2. Convergence speed on Medium tasks is approximately 40% faster than Reflexion and 37% faster than SkillLibrary, demonstrating the sample efficiency benefit of structured path-level experience.
3. Step efficiency improves substantially, indicating that the graph helps the agent identify shorter, more direct paths over time.

### 5.3 Learning Curves

> **<mark style="background-color: yellow;">⚠️ PROJECTED RESULTS — Pending experimental validation</mark>**

Figure 2 shows success rate (y-axis) as a function of episode number (x-axis) for Medium-difficulty tasks, averaged over 5 seeds with standard error bands.

```
Success Rate (%)
100|
 90|                                          ___________  ExperienceGraph
 80|                              ___________/
 70|                    _________/       _______________  SkillLibrary
 60|              _____/        ________/
 50|        _____/      _______/    ___________________  Reflexion
 40|   ____/     ______/     ______/
 30|  /    _____/     ______/       ___________________  VectorTraj
 20| /____/     _____/
 10|/     _____/                    ___________________  ReAct
  0|____________________________________________________________
   0    50    100   150   200   250   300  Episode
```
*Figure 2: Learning curves on Medium tasks <mark style="background-color: yellow;">(projected)</mark>. ExperienceGraph shows faster convergence and higher asymptotic performance.*

**<mark style="background-color: yellow;">Projected observations:</mark>**
- ReAct shows no learning trend (flat line) since it has no cross-episode memory.
- Reflexion and VectorTrajectory show gradual improvement but plateau at a lower level.
- SkillLibrary converges faster initially (reusing successful macros) but plateaus because it cannot compare alternative strategies conditioned on state.
- ExperienceGraph shows the steepest learning curve and highest asymptote, with the statistical advantage becoming significant (p < 0.05, paired t-test) by episode 100.

### 5.4 Ablation Studies

> **<mark style="background-color: yellow;">⚠️ PROJECTED RESULTS — Pending experimental validation</mark>**

We ablate key components of ExperienceGraph to isolate their contributions. All ablations are evaluated on Medium tasks.

**Table 2: Ablation results (Medium task, 300 episodes, 5 seeds).**

| Variant | SR@300 | CS | Δ vs. Full |
|---------|:---:|:---:|:---:|
| **ExperienceGraph (Full)** | **<mark style="background-color: yellow;">[PROJ: 85%]</mark>** | **<mark style="background-color: yellow;">[PROJ: 100]</mark>** | — |
| EG − No Exploration | <mark style="background-color: yellow;">[PROJ: 72%]</mark> | <mark style="background-color: yellow;">[PROJ: 155]</mark> | −13% |
| EG − No Statistics | <mark style="background-color: yellow;">[PROJ: 70%]</mark> | <mark style="background-color: yellow;">[PROJ: 170]</mark> | −15% |
| EG − No Node Merging | <mark style="background-color: yellow;">[PROJ: 75%]</mark> | <mark style="background-color: yellow;">[PROJ: 140]</mark> | −10% |
| EG − No Failure Edges | <mark style="background-color: yellow;">[PROJ: 78%]</mark> | <mark style="background-color: yellow;">[PROJ: 130]</mark> | −7% |
| EG − No Decay | <mark style="background-color: yellow;">[PROJ: 80%]</mark> | <mark style="background-color: yellow;">[PROJ: 120]</mark> | −5% |
| EG − Random Retrieval | <mark style="background-color: yellow;">[PROJ: 68%]</mark> | <mark style="background-color: yellow;">[PROJ: 185]</mark> | −17% |

**<mark style="background-color: yellow;">Analysis (Projected):</mark>**
- Removing exploration (only retrieving from graph, never generating novel paths) reduces final performance by 13%, confirming that the graph alone is insufficient and continued exploration prevents premature convergence.
- Removing statistics (showing paths without success rates) causes the largest single-component drop (−15%), demonstrating that quantitative evidence is critical for path selection.
- Replacing state-conditioned retrieval with random path retrieval produces the worst single ablation (−17%), confirming that presenting *relevant* paths rather than arbitrary ones is essential.
- Removing node merging (treating every trajectory checkpoint as unique) increases graph redundancy and hurts performance by 10%, validating the merging algorithm's contribution.
- Removing failure edges and decay have smaller but still significant effects, confirming their roles in avoiding bad paths and maintaining recency.

### 5.5 Graph Growth Analysis

> **<mark style="background-color: yellow;">⚠️ PROJECTED RESULTS — Pending experimental validation</mark>**

**Table 3: Graph statistics at episode milestones (Medium task, single seed).**

| Episode | |V| (Nodes) | |E| (Edges) | New Nodes/Ep | Paths to Goal | Prompt Tokens |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 50 | <mark style="background-color: yellow;">[PROJ: 25]</mark> | <mark style="background-color: yellow;">[PROJ: 45]</mark> | <mark style="background-color: yellow;">[PROJ: 1.2]</mark> | <mark style="background-color: yellow;">[PROJ: 4]</mark> | <mark style="background-color: yellow;">[PROJ: 650]</mark> |
| 100 | <mark style="background-color: yellow;">[PROJ: 38]</mark> | <mark style="background-color: yellow;">[PROJ: 72]</mark> | <mark style="background-color: yellow;">[PROJ: 0.5]</mark> | <mark style="background-color: yellow;">[PROJ: 7]</mark> | <mark style="background-color: yellow;">[PROJ: 820]</mark> |
| 150 | <mark style="background-color: yellow;">[PROJ: 45]</mark> | <mark style="background-color: yellow;">[PROJ: 90]</mark> | <mark style="background-color: yellow;">[PROJ: 0.2]</mark> | <mark style="background-color: yellow;">[PROJ: 9]</mark> | <mark style="background-color: yellow;">[PROJ: 900]</mark> |
| 200 | <mark style="background-color: yellow;">[PROJ: 50]</mark> | <mark style="background-color: yellow;">[PROJ: 100]</mark> | <mark style="background-color: yellow;">[PROJ: 0.1]</mark> | <mark style="background-color: yellow;">[PROJ: 10]</mark> | <mark style="background-color: yellow;">[PROJ: 950]</mark> |
| 300 | <mark style="background-color: yellow;">[PROJ: 55]</mark> | <mark style="background-color: yellow;">[PROJ: 110]</mark> | <mark style="background-color: yellow;">[PROJ: 0.05]</mark> | <mark style="background-color: yellow;">[PROJ: 11]</mark> | <mark style="background-color: yellow;">[PROJ: 980]</mark> |

**<mark style="background-color: yellow;">Projected observations:</mark>**
- Node creation rate decreases rapidly, confirming that the merging algorithm effectively consolidates equivalent checkpoints.
- The graph approaches structural saturation by episode 150–200, after which new episodes primarily update statistics.
- Prompt token usage remains bounded below 1000 tokens throughout, validating the fixed-budget mechanism.

### 5.6 Cost Analysis

> **<mark style="background-color: yellow;">⚠️ PROJECTED RESULTS — Pending experimental validation</mark>**

**Table 4: Token cost per successful episode (Medium task, averaged over 5 seeds).**

| Method | Tokens/Success | Tokens/Episode (all) | Total Cost (300 eps) |
|--------|:---:|:---:|:---:|
| ReAct | <mark style="background-color: yellow;">[PROJ: 3200]</mark> | <mark style="background-color: yellow;">[PROJ: 2800]</mark> | <mark style="background-color: yellow;">[PROJ: 840K]</mark> |
| Reflexion | <mark style="background-color: yellow;">[PROJ: 4500]</mark> | <mark style="background-color: yellow;">[PROJ: 4000]</mark> | <mark style="background-color: yellow;">[PROJ: 1200K]</mark> |
| VectorTrajectory | <mark style="background-color: yellow;">[PROJ: 4200]</mark> | <mark style="background-color: yellow;">[PROJ: 3800]</mark> | <mark style="background-color: yellow;">[PROJ: 1140K]</mark> |
| SkillLibrary | <mark style="background-color: yellow;">[PROJ: 3800]</mark> | <mark style="background-color: yellow;">[PROJ: 3500]</mark> | <mark style="background-color: yellow;">[PROJ: 1050K]</mark> |
| **ExperienceGraph** | <mark style="background-color: yellow;">[PROJ: 4800]</mark> | <mark style="background-color: yellow;">[PROJ: 4200]</mark> | <mark style="background-color: yellow;">[PROJ: 1260K]</mark> |

ExperienceGraph has moderately higher per-episode token cost due to the graph context presented to the LLM. However, when normalized by success rate, the *cost per successful outcome* may be competitive because fewer episodes are wasted on failures. We report both raw cost and cost-normalized-by-success to provide a complete efficiency picture.

## 6. Analysis

### 6.1 Case Study: Path Discovery and Refinement

> **<mark style="background-color: yellow;">⚠️ PROJECTED — Illustrative example based on environment design</mark>**

We trace the evolution of the agent's strategy for the Medium task (diamond armor) across episodes:

- **Episode 5**: Agent attempts to mine diamonds without an iron pickaxe → failure. Trajectory stored as a failed path with hard precondition `pickaxe: iron|diamond` learned.
- **Episode 12**: Agent crafts iron pickaxe first, then mines diamonds → success. New path with score 1/1 added.
- **Episode 25**: Agent discovers village trading route (emeralds → diamonds) → success in fewer steps. Alternative path added with score 1/1.
- **Episode 45**: Both paths have accumulated statistics. Mining path: 8/12 (0.64). Trading path: 6/7 (0.86). Agent preferentially selects trading when near a village with confirmed armorer.
- **Episode 80**: Agent generates a hybrid path (mine some + trade some) that succeeds in a state where neither pure path is optimal. This novel path enters the graph.
- **Episode 150+**: Graph has stabilized. Agent consistently selects the best path conditioned on initial state (nearby village → trade; no village → mine; partial resources → hybrid).

### 6.2 Failure Mode Analysis

> **<mark style="background-color: yellow;">⚠️ PROJECTED — Based on anticipated failure modes</mark>**

We identify three primary failure modes:

1. **Incorrect node merging** (~5% of failures): Two semantically similar but functionally different checkpoints are merged, leading to incorrect precondition assumptions in subsequent episodes. The decay mechanism partially mitigates this by downweighting stale statistics.

2. **Exploration overhead** (~8% of failures): The novel candidate path generated by the LLM is invalid or inefficient, wasting an episode on exploration. This cost is inherent to the explore-exploit tradeoff and diminishes as the graph matures.

3. **Stale statistics** (~3% of failures): A path that previously succeeded becomes less viable due to environment stochasticity (different initial state distribution), but high historical statistics delay adaptation. The decay mechanism addresses this but introduces a lag.

### 6.3 Node Merging Quality

> **<mark style="background-color: yellow;">⚠️ PROJECTED — Pending manual annotation study</mark>**

We sample 50 merging decisions and manually annotate correctness:
- **<mark style="background-color: yellow;">[PROJECTED: 88% correct merges]</mark>** — conditions are genuinely equivalent.
- **<mark style="background-color: yellow;">[PROJECTED: 8% conservative non-merges]</mark>** — conditions are equivalent but rules were too strict; results in minor redundancy but no harm.
- **<mark style="background-color: yellow;">[PROJECTED: 4% incorrect merges]</mark>** — conditions differ meaningfully; leads to incorrect path suggestions in some states.

The hierarchical merging algorithm (deterministic rules first, LLM only when ambiguous) reduces LLM merging calls to **<mark style="background-color: yellow;">[PROJECTED: ~12% of total merge decisions]</mark>**, keeping cost low while maintaining high merge accuracy.

## 7. Discussion

### 7.1 Limitations

**Environment scope.** TextCraft is a controlled, relatively low-dimensional environment. While it captures key properties (partial observability, multi-path structure, conditional dependencies), it does not exhibit the continuous state spaces, visual complexity, or open-ended goal spaces of real-world embodied environments.

**Scalability of node merging.** The node merging algorithm relies on a finite, discrete state space for its rule-based components. In continuous or high-dimensional state spaces, the normalization and exact-match steps would require embedding-based similarity, introducing additional design choices and potential errors.

**LLM dependency.** The framework's performance is coupled to the backbone LLM's reasoning quality. Weaker models may make poor path selection decisions even when presented with good statistical evidence, or generate low-quality exploration candidates.

**Single-task evaluation.** Each experiment evaluates learning on a single goal repeated across episodes. Multi-goal settings where the agent must transfer graph knowledge between related but different objectives remain unexplored.

### 7.2 Connections to Reinforcement Learning

ExperienceGraph shares conceptual connections with model-based reinforcement learning: the graph can be viewed as a learned, discrete world model where nodes are abstract states and edge statistics approximate transition-reward signals. However, key differences include: (1) no gradient-based optimization—learning occurs through structural graph growth and count-based statistics; (2) the LLM provides the policy and value function through in-context reasoning rather than learned parameters; (3) the representation is symbolic and interpretable rather than neural.

This positions ExperienceGraph as a middle ground between pure in-context learning (no persistent structure) and full RL (requiring many samples and gradient updates), potentially suitable for settings where sample efficiency matters more than asymptotic optimality.

### 7.3 Generalization Potential

The ExperienceGraph framework abstracts over the specific semantics of TextCraft. The core mechanism—conditional nodes, action edges with layered preconditions, path-level statistics, explore-then-compare decision-making—applies to any environment satisfying four conditions: (1) states are discretizable into finite condition sets, (2) multiple paths exist to goals, (3) action outcomes are state-dependent, and (4) the agent interacts repeatedly with the same environment type.

Potential transfer domains include:
- **Embodied navigation**: nodes as spatial landmarks with accessibility conditions, edges as navigation primitives.
- **Dialogue systems**: nodes as dialogue states (slot-filling progress), edges as dialogue acts.
- **Code generation**: nodes as test-passing milestones, edges as coding actions.

We leave empirical validation of cross-domain transfer to future work.

## 8. Conclusion

This paper proposes ExperienceGraph, a framework for organizing LLM agent experience as a conditional path graph with state-conditioned retrieval and path-level success statistics. The key insight is that in multi-path environments, *structured, statistically grounded experience organization* enables fundamentally better cross-episode learning than flat memory or isolated skill storage.

Experiments on TextCraft demonstrate that ExperienceGraph achieves **<mark style="background-color: yellow;">[PROJECTED: 20+ percentage points]</mark>** higher success rate than the strongest baseline while converging **<mark style="background-color: yellow;">[PROJECTED: ~40%]</mark>** faster. Ablation studies confirm that the graph structure, statistical annotations, exploration mechanism, and state-conditioned retrieval each contribute meaningfully to performance.

A current limitation is the controlled, discrete nature of the evaluation environment. Extending ExperienceGraph to continuous state spaces, multi-goal settings, and real-world embodied environments—where node merging requires learned representations rather than symbolic matching—represents the most important direction for future work.

---

## References

- Besta, M., et al. (2023). Graph of Thoughts: Solving Elaborate Problems with Large Language Models. *arXiv:2308.09687*.
- Edge, D., et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. *arXiv:2404.16130*.
- Gutiérrez, B. J., et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *arXiv:2405.14831*.
- Hao, S., et al. (2023). Reasoning with Language Model is Planning with World Model. *arXiv:2305.14992*.
- Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *arXiv:2303.11366*.
- Wang, G., et al. (2023a). Voyager: An Open-Ended Embodied Agent with Large Language Models. *arXiv:2305.16291*.
- Wang, Z., et al. (2023b). Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents. *arXiv:2302.01560*.
- Wang, Y., et al. (2025). A-MEM: Agentic Memory for LLM Agents. *arXiv:2502.12110*.
- Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *arXiv:2210.03629*.
- Yao, S., et al. (2023b). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. *arXiv:2305.10601*.
- Zhou, A., et al. (2023). Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models. *arXiv:2310.04406*.
- Zhu, X., et al. (2023). Ghost in the Minecraft: Generally Capable Computer-Playing Agents with Human-Level Capabilities. *arXiv:2305.17144*.
- Zhuge, M., et al. (2024). Language Agents as Optimizable Graphs. *arXiv:2402.16823*.

---

## Appendix A: Self-Review — Claim-Evidence Map

| # | Claim | Evidence | Status |
|---|-------|----------|--------|
| 1 | ExperienceGraph achieves higher success rate than all baselines | Table 1 main comparison | <mark style="background-color: yellow;">⚠️ Needs evidence (projected)</mark> |
| 2 | Convergence is ~40% faster than Reflexion | Table 1 CS column + Figure 2 | <mark style="background-color: yellow;">⚠️ Needs evidence (projected)</mark> |
| 3 | Graph structure, statistics, and exploration each contribute significantly | Table 2 ablation | <mark style="background-color: yellow;">⚠️ Needs evidence (projected)</mark> |
| 4 | Graph growth is sublinear and plateaus | Table 3 graph statistics | <mark style="background-color: yellow;">⚠️ Needs evidence (projected)</mark> |
| 5 | Prompt size remains bounded under 1000 tokens | Table 3 prompt tokens column | <mark style="background-color: yellow;">⚠️ Needs evidence (projected)</mark> |
| 6 | Node merging is 88% accurate | Section 6.3 annotation study | <mark style="background-color: yellow;">⚠️ Needs evidence (projected)</mark> |
| 7 | Advantage grows with task complexity | Table 1 Easy/Medium/Hard comparison | <mark style="background-color: yellow;">⚠️ Needs evidence (projected)</mark> |
| 8 | Flat memory cannot model conditional path preferences | Conceptual argument + ablation EG-no-stats and EG-random-retrieval | <mark style="background-color: yellow;">Partially supported (ablation projected)</mark> |
| 9 | Explore-then-compare prevents premature convergence | EG-no-exploration ablation shows -13% | <mark style="background-color: yellow;">⚠️ Needs evidence (projected)</mark> |
| 10 | ExperienceGraph is complementary to test-time search | Not experimentally tested | ❌ Unsupported — future work |

**Action items before submission:**
- Claims 1–9 require running the full experiment suite to convert from <mark style="background-color: yellow;">projected</mark> to verified.
- Claim 10 should either be tested with an additional LATS+ExperienceGraph condition, or explicitly moved to future work without implying empirical support.

---

## Appendix B: Five-Dimension Self-Review

### Dimension 1: Contribution

| Question | Answer |
|----------|--------|
| Is the contribution clearly stated? | Yes — conditional path graph for cross-episode LLM agent experience. |
| Is it distinguishable from prior graph methods? | Yes — Section 2.2 explicitly distinguishes from reasoning graphs, knowledge graphs, and workflow graphs. |
| Is the novelty overstated? | Risk: "first to organize trajectories as conditional path graphs" could be challenged. Mitigation: we frame it as a specific combination (conditions + statistics + path-level retrieval) rather than claiming graph usage is novel. |
| Are there 4 clear contributions? | Yes — framework, mechanism, empirical evaluation, benchmark. |

### Dimension 2: Writing Clarity

| Question | Answer |
|----------|--------|
| Does each paragraph have one message? | Checked — yes for all sections. |
| Are terms defined before use? | "Experience graph," "checkpoint," "hard/soft precondition" all defined in Section 3.2 before reuse. |
| Is terminology stable? | Verified: "ExperienceGraph" (method), "experience graph" (data structure), "path" (strategy), "node/checkpoint" (state condition) — consistent throughout. |
| Are there jump-in-reading moments? | Potential jump: Section 3.3 assumes reader understands beta-binomial scoring from 3.2. The formula is in 3.2 which precedes 3.3 — acceptable. |

### Dimension 3: Experimental Strength

| Question | Answer |
|----------|--------|
| Are baselines strong and recent? | Yes — includes Reflexion, VectorTrajectory (embedding-based), and SkillLibrary (Voyager-style). |
| Is the evaluation fair? | Same LLM, same environment, same seeds, same observation format. |
| Are ablations tied to claims? | Each ablation removes one design component and measures impact — directly maps to contribution claims. |
| Is cost reported? | Yes — Table 4 reports token cost. Note: ExperienceGraph is more expensive per-episode; should discuss cost-per-success normalization honestly. |
| Are failure modes analyzed? | Yes — Section 6.2 identifies three failure categories with estimated frequency. |

### Dimension 4: Evaluation Completeness

| Question | Answer |
|----------|--------|
| Multiple difficulty levels? | Yes — Easy/Medium/Hard. |
| Statistical rigor? | 5 seeds, paired t-test, standard error bands, confidence intervals planned. |
| Missing evaluation? | (1) No multi-goal transfer test. (2) No LLM ablation (GPT-4 vs. weaker model). (3) No LATS+EG combination. These are acknowledged as limitations. |
| Is there a qualitative analysis? | Yes — case study in Section 6.1 and merging quality in 6.3. |

### Dimension 5: Method Design Soundness

| Question | Answer |
|----------|--------|
| Is the method well-motivated? | Yes — each component (graph structure, statistics, explore-then-compare, merging, compression) has explicit motivation tied to a limitation of alternatives. |
| Are design choices justified? | Beta-binomial scoring: justified over raw ratios. Top-K retrieval: justified by token budget. Hierarchical merging: justified by cost. |
| Are there potential failure modes acknowledged? | Yes — incorrect merging, exploration overhead, stale statistics (Section 6.2). |
| Is the approach reproducible? | State space, action space, scoring formula, merging algorithm, and compression rules are all specified. Environment details in Section 4 enable reimplementation. |
| What would a skeptical reviewer attack? | (1) TextCraft is too simple / not a real benchmark. (2) <mark style="background-color: yellow;">Projected results</mark> are speculative. (3) Cost overhead not justified. (4) Node merging quality may degrade in harder environments. All acknowledged in Discussion. |

---

## Appendix C: Terminology Index

| Term | Definition | First Introduced |
|------|-----------|-----------------|
| ExperienceGraph | The proposed framework (system name) | Abstract |
| Experience graph | The data structure $G=(V,E)$ | Section 3.2 |
| Checkpoint (node) | A state-condition set representing a decision point | Section 3.2 |
| Hard precondition | Machine-checkable condition that gates edge feasibility | Section 3.2 |
| Soft precondition | Natural-language heuristic for LLM reasoning | Section 3.2 |
| Path | Ordered edge sequence from start to goal | Section 3.2 |
| Explore-then-compare | Three-phase decision process (generate, retrieve, select) | Section 3.3 |
| Graph Organizer | Module that integrates trajectories and manages the graph | Section 3.4 |
| Experience distillation | Active compression mechanisms (pruning, decay, folding) | Section 3.5 |
| TextCraft | Text-based evaluation environment | Section 4 |
