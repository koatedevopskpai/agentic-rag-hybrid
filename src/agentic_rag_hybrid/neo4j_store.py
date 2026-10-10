"""Optional live backend: the same retrieval against a real Neo4j instance.

Requires ``pip install neo4j`` and ``NEO4J_URI`` / ``NEO4J_USER`` /
``NEO4J_PASSWORD``. Import is lazy so the package works without the driver.
"""

from __future__ import annotations

import os

__all__ = ["Neo4jGraphStore", "cypher_expand"]


def cypher_expand(max_hops: int = 2) -> str:
    """The traversal query: seeds in (hop 0), ranked nodes out."""
    return (
        "MATCH (s:Node) WHERE s.id IN $seeds "
        "RETURN s.id AS node, 0 AS hops "
        "UNION "
        "MATCH p = (s:Node)-[*1..$hops]-(m:Node) WHERE s.id IN $seeds "
        "RETURN m.id AS node, length(p) AS hops "
        "ORDER BY hops, node"
    ).replace("$hops", str(max_hops))


class Neo4jGraphStore:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError(
                "neo4j driver not installed — run: pip install neo4j"
            ) from exc
        self._driver = GraphDatabase.driver(
            uri or os.environ["NEO4J_URI"],
            auth=(user or os.getenv("NEO4J_USER", "neo4j"), password or os.environ["NEO4J_PASSWORD"]),
        )

    def traverse(self, seeds: list[str], max_hops: int = 2) -> list[dict]:
        query = cypher_expand(max_hops)
        with self._driver.session() as session:
            records = session.run(query, seeds=seeds)
            rows = [{"node": r["node"], "hops": r["hops"]} for r in records]
        rows.sort(key=lambda r: (r["hops"], r["node"]))
        return [
            {"node": r["node"], "hops": r["hops"], "score": round(1 / (1 + r["hops"]), 4)}
            for r in rows
        ]

    def close(self) -> None:
        self._driver.close()
