from __future__ import annotations

from experience_graph.core.models import CandidatePathView, Condition, Observation, TaskSpec
from experience_graph.graph.retriever import GraphRetriever
from experience_graph.graph.store import JsonGraphStore
from experience_graph.semantic_retrieval.canonical import observation_text
from experience_graph.semantic_retrieval.index import SemanticNodeIndex, SemanticNeighbor
from experience_graph.semantic_retrieval.text_vectors import HashingTextVectorizer


class SemanticFallbackGraphRetriever(GraphRetriever):
    def __init__(
        self,
        store: JsonGraphStore,
        token_budget: int = 1000,
        top_k: int = 5,
        include_statistics: bool = True,
        ranking_mode: str = "score",
        random_seed: int = 0,
        cross_task_mode: str = "same_task",
        semantic_top_k: int = 5,
        semantic_min_score: float = 0.05,
        semantic_bonus: float = 0.25,
        vectorizer: HashingTextVectorizer | None = None,
    ):
        super().__init__(
            store=store,
            token_budget=token_budget,
            top_k=top_k,
            include_statistics=include_statistics,
            ranking_mode=ranking_mode,
            random_seed=random_seed,
            cross_task_mode=cross_task_mode,
        )
        self.semantic_top_k = semantic_top_k
        self.semantic_min_score = semantic_min_score
        self.semantic_bonus = semantic_bonus
        self.semantic_index = SemanticNodeIndex(store, vectorizer=vectorizer)

    def retrieve(self, task: TaskSpec, observation: Observation, current_conditions: list[Condition]):
        original_top_k = self.top_k
        self.top_k = max(original_top_k, self.semantic_top_k * 2)
        try:
            view = super().retrieve(task, observation, current_conditions)
        finally:
            self.top_k = original_top_k

        if original_top_k <= 0 or self._has_exact_binding(observation):
            view.candidate_paths = view.candidate_paths[:original_top_k]
            view.token_estimate = min(view.token_estimate, self.token_budget)
            return view

        query = observation_text(task, observation, current_conditions)
        neighbors = self.semantic_index.search(query, limit=self.semantic_top_k, min_score=self.semantic_min_score)
        if not neighbors:
            view.candidate_paths = view.candidate_paths[:original_top_k]
            view.summaries.append("Semantic fallback found no close node anchors.")
            view.token_estimate = min(self._estimate_tokens(view.candidate_paths), self.token_budget)
            return view

        neighbor_by_id = {neighbor.node_id: neighbor for neighbor in neighbors}
        for candidate in view.candidate_paths:
            match = self._best_path_anchor(candidate, neighbor_by_id)
            if match is None:
                continue
            candidate.retrieval_score = round(candidate.retrieval_score + self.semantic_bonus * match.score, 4)
            candidate.state_similarity = max(candidate.state_similarity, match.score)
            candidate.retrieval_reason = (
                candidate.retrieval_reason
                + f" Semantic fallback anchor: {match.node_id} similarity={match.score:.2f}."
            )

        view.candidate_paths.sort(key=lambda c: (c.applicability == "blocked", -c.retrieval_score, -(c.success_rate or 0), c.avg_steps or 999))
        view.candidate_paths = view.candidate_paths[:original_top_k]
        anchors = ", ".join(f"{neighbor.node_id}:{neighbor.score:.2f}" for neighbor in neighbors[:3])
        view.summaries.append(f"Semantic fallback anchors: {anchors}.")
        view.token_estimate = min(self._estimate_tokens(view.candidate_paths), self.token_budget)
        return view

    def _has_exact_binding(self, observation: Observation) -> bool:
        state = observation.state
        for node in self.store.nodes.values():
            if node.required and all(condition.matches_state(state) for condition in node.required):
                return True
        return False

    def _best_path_anchor(self, candidate: CandidatePathView, neighbor_by_id: dict[str, SemanticNeighbor]) -> SemanticNeighbor | None:
        path = self.store.paths.get(candidate.path_id)
        if path is None:
            return None
        best: SemanticNeighbor | None = None
        for edge_id in path.edge_ids:
            edge = self.store.edges.get(edge_id)
            if edge is None:
                continue
            for node_id in (edge.from_node, edge.to_node):
                neighbor = neighbor_by_id.get(node_id)
                if neighbor is not None and (best is None or neighbor.score > best.score):
                    best = neighbor
        return best
