"""Unit tests for the Neo4j query builders.

No database required: these assert the shape of the generated Cypher and the
input validation around it. Live behaviour is verified separately by
``evals/live_neo4j_parity.py`` (H03) and ``evals/high_degree_bench.py`` (H04),
which are opt-in because CI has no Neo4j.
"""

from __future__ import annotations

import pytest

from agentic_rag_hybrid.neo4j_store import cypher_expand, cypher_expand_bounded


class TestCypherExpand:
    def test_includes_seeds_and_minimum_hops(self):
        q = cypher_expand(2)
        # Seeds must be included without a UNION, and each node must appear once
        # at its minimum hop distance. Both were defects: the original used a
        # UNION and returned a node once per path length.
        assert "UNION" not in q
        assert "collect(s) AS seeds UNWIND seeds AS s" in q
        assert "min(length(p))" in q
        assert "*0..2" in q

    def test_hop_bound_is_interpolated(self):
        assert "*0..3" in cypher_expand(3)

    @pytest.mark.parametrize("bad", [-1, 1.5, "2", True, None])
    def test_rejects_unsafe_hop_bounds(self, bad):
        with pytest.raises(ValueError):
            cypher_expand(bad)


class TestCypherExpandBounded:
    def test_no_bounds_by_default(self):
        q = cypher_expand_bounded(2)
        assert "LIMIT" not in q
        assert "[*0..2]" in q
        assert ":" not in q.split("MATCH p")[1].split("-(m:Node)")[0]

    def test_relationship_filter_includes_the_colon(self):
        """`[LINKS*0..2]` is a variable named LINKS; `[:LINKS*0..2]` is a filter.

        The missing colon was a real defect: the filtered query matched every
        relationship type and returned the same row count as unfiltered.
        """
        q = cypher_expand_bounded(2, rel_type="LINKS")
        assert "[:LINKS*0..2]" in q

    @pytest.mark.parametrize(
        "bad",
        ["links", "LINK S", "LINKS]->() DETACH DELETE n //", "", "1LINKS", "LINKS;"],
    )
    def test_rejects_unsafe_relationship_types(self, bad):
        with pytest.raises(ValueError):
            cypher_expand_bounded(2, rel_type=bad)

    def test_limit_is_appended(self):
        assert cypher_expand_bounded(2, limit=10).rstrip().endswith("LIMIT 10")

    @pytest.mark.parametrize("bad", [0, -5, "10", 2.5])
    def test_rejects_unsafe_limits(self, bad):
        with pytest.raises(ValueError):
            cypher_expand_bounded(2, limit=bad)

    def test_bounds_combine(self):
        q = cypher_expand_bounded(1, rel_type="USES", limit=5)
        assert "[:USES*0..1]" in q
        assert "LIMIT 5" in q
        assert "min(length(p))" in q
