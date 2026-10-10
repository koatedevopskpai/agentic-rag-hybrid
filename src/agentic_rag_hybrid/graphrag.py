"""GraphRAG: entity-seeded expansion over the knowledge graph.

Every returned document is reachable from a seed entity extracted from the
query — grounded by construction. Deterministic: same graph + query always
yields the same ranking.
"""

from __future__ import annotations

from .graph import KnowledgeGraph

__all__ = ["GraphRAG"]


class GraphRAG:
    def __init__(self, graph: KnowledgeGraph, max_hops: int = 2, top_k: int | None = None):
        self.graph = graph
        self.max_hops = max_hops
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[str, str, float]]:
        seeds = self.graph.extract_entities(query)
        if not seeds:
            return []
        limit = top_k if top_k is not None else self.top_k
        ranked = self.graph.traverse(seeds, max_hops=self.max_hops)
        if limit is not None:
            ranked = ranked[:limit]
        return [
            (entry["node"], self.graph.fact_text(entry["node"]), entry["score"])
            for entry in ranked
        ]