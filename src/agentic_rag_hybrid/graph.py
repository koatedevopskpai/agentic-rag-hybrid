"""Deterministic knowledge graph + traversal for GraphRAG.

No database required: entities and relations live in memory, so retrieval
behaviour is fully reproducible and unit-testable.

Production swap: point ``Neo4jGraphStore`` (``neo4j_store.py``) at a real
Neo4j instance — same interface, Cypher underneath. Each method documents its
Cypher equivalent.
"""

from __future__ import annotations

import re
from collections import deque

__all__ = ["KnowledgeGraph", "normalise"]


def normalise(text: str) -> str:
    """Lowercase alphanumeric fold, so 'RAG engine' matches id 'rag-engine'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


class KnowledgeGraph:
    def __init__(self) -> None:
        self.entities: dict[str, dict] = {}
        # node -> [(relation, neighbour, outgoing)]
        self._adj: dict[str, list[tuple[str, str, bool]]] = {}

    def add_entity(
        self, entity_id: str, type: str = "node", aliases: tuple[str, ...] = (), **props
    ) -> None:
        self.entities[entity_id] = {
            "type": type,
            "aliases": list(aliases),
            "props": dict(props),
        }
        self._adj.setdefault(entity_id, [])

    def add_relation(self, src: str, rel: str, dst: str) -> None:
        for node in (src, dst):
            if node not in self.entities:
                raise ValueError(f"unknown entity {node!r}")
        self._adj[src].append((rel, dst, True))
        self._adj[dst].append((rel, src, False))

    def neighbours(self, node: str) -> list[tuple[str, str, bool]]:
        return list(self._adj.get(node, []))

    def extract_entities(self, query: str) -> list[str]:
        """Seed set: entities whose id/alias appears in the query.

        Cypher equivalent: full-text or exact-match lookup on indexed
        ``id`` / ``aliases`` properties.
        """
        nq = normalise(query)
        hits = []
        for eid, ent in self.entities.items():
            for name in [eid, *ent["aliases"]]:
                folded = normalise(name)
                if folded and folded in nq:
                    hits.append(eid)
                    break
        return sorted(hits)

    def traverse(self, seeds: list[str], max_hops: int = 2) -> list[dict]:
        """Undirected BFS from seeds.

        Score = 1 / (1 + hops); ties break by node id (deterministic).
        Cycle-safe by construction (visited set).

        Cypher equivalent: ``MATCH p = (s)-[*1..$hops]-(m)`` ordered by
        ``length(p), m.id``.
        """
        best: dict[str, tuple[int, list[str]]] = {}
        queue: deque[str] = deque()
        for seed in seeds:
            if seed in self.entities and seed not in best:
                best[seed] = (0, [seed])
                queue.append(seed)
        while queue:
            node = queue.popleft()
            hops, path = best[node]
            if hops >= max_hops:
                continue
            for rel, neighbour, outgoing in sorted(self._adj.get(node, [])):
                if neighbour in best:
                    continue
                arrow = f"{rel}>{neighbour}" if outgoing else f"{rel}<{neighbour}"
                best[neighbour] = (hops + 1, [*path, arrow])
                queue.append(neighbour)

        results = [
            {"node": node, "path": path, "hops": hops, "score": round(1 / (1 + hops), 4)}
            for node, (hops, path) in best.items()
        ]
        results.sort(key=lambda r: (-r["score"], r["node"]))
        return results

    def fact_text(self, node: str) -> str:
        """Render a node as evidence text."""
        ent = self.entities[node]
        lines = [f"{node} [{ent['type']}]"]
        for rel, neighbour, outgoing in sorted(self._adj.get(node, [])):
            arrow = "-->" if outgoing else "<--"
            lines.append(f"  {node} -[{rel}]{arrow} {neighbour}")
        return "\n".join(lines)
