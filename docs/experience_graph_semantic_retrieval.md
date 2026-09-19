# ExperienceGraph Semantic Retrieval

Last updated: 2026-06-27

## Purpose

This design keeps the existing ExperienceGraph as the primary memory structure and adds an optional semantic retrieval layer for harder MyTextCraft worlds.

The current graph retriever stays unchanged by default. The new mode is enabled only with:

```powershell
--retrieval-mode semantic_fallback
```

The old baseline was marked with the local git tag:

```text
pre_semantic_retriever_20260627
```

## Architecture

New code lives in:

```text
src/experience_graph/semantic_retrieval/
```

The folder has four parts:

| File | Role |
| --- | --- |
| `canonical.py` | Converts observation state and graph nodes into stable text. |
| `text_vectors.py` | Provides a deterministic local text vectorizer for first-pass testing. |
| `index.py` | Builds a node-level semantic search index over existing graph nodes. |
| `retriever.py` | Adds semantic fallback on top of the existing `GraphRetriever`. |

## Current Behavior

`SemanticFallbackGraphRetriever` first asks the normal `GraphRetriever` for candidate paths.

If the current observation can bind exactly to an existing graph node, it keeps the normal result.

If there is no exact node binding, it:

1. Converts the current task and visible observation into canonical text.
2. Searches graph nodes for semantically similar anchors.
3. Boosts candidate paths that pass through those anchor nodes.
4. Adds the semantic anchor information to `ExperienceView.summaries` and each boosted path's `retrieval_reason`.

This means semantic matching can influence ranking, but it does not create actions or claim that two states are equivalent.

## Why The First Version Uses Local Vectors

The first version uses a deterministic hashing vectorizer instead of a paid embedding API.

Reasons:

- It is safe to test during ongoing real LLM runs.
- It keeps unit tests deterministic.
- It gives us a stable interface before choosing OpenAI, BGE, E5, or another embedding backend.

Later, `text_vectors.py` can add real embedding providers without changing the runner interface.

## Experiment Plan

The next comparison should use the same MyTextCraft short suite:

```text
short10 x 5 rounds
```

Suggested runs:

| Run | Purpose |
| --- | --- |
| `graph_full` | Existing baseline. |
| `graph_full + semantic_fallback` | Tests whether semantic anchors improve retrieval in harder mixed-task settings. |
| `graph_no_graph_context` | Confirms the graph context is doing useful work. |

The current pro run should remain the baseline for the old retriever.

## Boundaries

- Semantic similarity is not treated as action reachability.
- The graph structure still owns paths, edges, preconditions, and statistics.
- The LLM still receives candidate paths through the existing `ExperienceView` format.
- Default CLI behavior remains unchanged.
