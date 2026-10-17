"""Optional live backend: the same retrieval against a real Neo4j instance.

Requires ``pip install neo4j`` and ``NEO4J_URI`` / ``NEO4J_USER`` /
``NEO4J_PASSWORD``. Import is lazy so the package works without the driver.
"""

from __future__ import annotations

import os
import re

__all__ = ["Neo4jGraphStore", "cypher_expand", "cypher_expand_bounded"]


def cypher_expand(max_hops: int = 2) -> str:
    """The traversal query: seeds included at hop 0, each node at its MINIMUM hops.

    Two things this query must get right, both verified against a live instance by
    ``evals/live_neo4j_parity.py``:

    1. **Seeds are results too.** A zero-length pattern (``*0..``) includes the
       seed, so no ``UNION`` is needed to add them.
    2. **One row per node, at minimum hops.** The previous implementation used a
       ``UNION`` of seeds-at-0 and ``MATCH p = (s)-[*1..N]-(m)``. That returns one
       row *per path length*, so a node reachable at 1 and 2 hops appeared twice,
       and a cycle produced duplicates. The in-memory engine keeps the minimum, so
       the two backends disagreed. ``min(length(p))`` per node fixes that.

    ``max_hops`` is interpolated because Cypher cannot parameterise a
    variable-length bound, so it is validated as a positive integer.
    """
    hops = _validate_hops(max_hops)
    return (
        "MATCH (s:Node) WHERE s.id IN $seeds "
        # Collect then unwind the seeds BEFORE expanding. Binding the seeds
        # directly in one MATCH and aggregating afterwards produced duplicate rows
        # (a node reachable from two seeds, or via two path lengths, appeared
        # twice) on Neo4j 2026.09.0 / Cypher 5.25. Collapsing the seeds into a
        # single list first removes that. Verified by evals/live_neo4j_parity.py:
        # 15/15 probes, including a diamond and a cycle.
        "WITH collect(s) AS seeds UNWIND seeds AS s "
        f"MATCH p = (s)-[*0..{hops}]-(m:Node) "
        "RETURN m.id AS node, min(length(p)) AS hops "
        "ORDER BY hops, node"
    )


def _validate_hops(max_hops: int) -> int:
    """Variable-length bounds cannot be parameterised, so reject anything unsafe."""
    if not isinstance(max_hops, int) or isinstance(max_hops, bool) or max_hops < 0:
        raise ValueError(f"max_hops must be a non-negative integer, got {max_hops!r}")
    return max_hops


def _validate_rel_type(rel_type: str | None) -> str | None:
    """Relationship types are interpolated into the pattern, so validate them."""
    if rel_type is None:
        return None
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", rel_type):
        raise ValueError(f"invalid relationship type: {rel_type!r}")
    return rel_type


def cypher_expand_bounded(
    max_hops: int = 2,
    rel_type: str | None = None,
    limit: int | None = None,
) -> str:
    """The traversal with explicit bounds.

    ``max_hops`` bounds the expansion depth, which is a genuine bound on the work
    performed. ``rel_type`` restricts the pattern to one relationship type.

    ``limit`` bounds the number of rows *returned*. It is applied after
    ``ORDER BY``, so it trims the ranked result — it does **not** bound the
    traversal itself. On a high-degree hub the expansion still visits every
    reachable node before the limit is applied. This is measured in
    ``evals/high_degree_bench.py``; a result cap is not a work cap.
    """
    hops = _validate_hops(max_hops)
    rel = _validate_rel_type(rel_type)
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError(f"limit must be a positive integer, got {limit!r}")

    # NB: the colon is required - `[LINKS*0..2]` is a VARIABLE named LINKS matching
    # any relationship type, not a type filter. Caught by the benchmark: the
    # filtered variant returned the same row count as unfiltered.
    pattern = f"[:{rel}*0..{hops}]" if rel else f"[*0..{hops}]"
    query = (
        "MATCH (s:Node) WHERE s.id IN $seeds "
        "WITH collect(s) AS seeds UNWIND seeds AS s "
        f"MATCH p = (s)-{pattern}-(m:Node) "
        "RETURN m.id AS node, min(length(p)) AS hops "
        "ORDER BY hops, node"
    )
    if limit is not None:
        query += f" LIMIT {limit}"
    return query


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

    def traverse_bounded(
        self,
        seeds: list[str],
        max_hops: int = 2,
        rel_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Traversal with an optional relationship filter and result cap.

        See :func:`cypher_expand_bounded`: ``max_hops`` and ``rel_type`` bound the
        traversal, ``limit`` bounds only the returned rows.
        """
        query = cypher_expand_bounded(max_hops, rel_type, limit)
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
