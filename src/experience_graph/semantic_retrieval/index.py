from __future__ import annotations

from dataclasses import dataclass

from experience_graph.graph.store import JsonGraphStore
from experience_graph.semantic_retrieval.canonical import node_text
from experience_graph.semantic_retrieval.text_vectors import HashingTextVectorizer, cosine


@dataclass(frozen=True)
class SemanticNeighbor:
    node_id: str
    score: float
    text: str


class SemanticNodeIndex:
    def __init__(self, store: JsonGraphStore, vectorizer: HashingTextVectorizer | None = None):
        self.store = store
        self.vectorizer = vectorizer or HashingTextVectorizer()

    def search(self, query_text: str, limit: int = 5, min_score: float = 0.05) -> list[SemanticNeighbor]:
        query_vector = self.vectorizer.embed(query_text)
        neighbors: list[SemanticNeighbor] = []
        for node in self.store.nodes.values():
            text = node_text(node)
            score = cosine(query_vector, self.vectorizer.embed(text))
            if score >= min_score:
                neighbors.append(SemanticNeighbor(node_id=node.id, score=round(score, 4), text=text))
        neighbors.sort(key=lambda item: item.score, reverse=True)
        return neighbors[:limit]
