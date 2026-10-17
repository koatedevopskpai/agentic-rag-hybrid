import pytest

from agentic_rag_hybrid import GraphRAG, KnowledgeGraph, cypher_expand
from agentic_rag_hybrid.neo4j_store import Neo4jGraphStore


def _org_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_entity("alice", type="person")
    g.add_entity("bob", type="person")
    g.add_entity("rag-engine", type="project", aliases=("RAG engine",))
    g.add_entity("eval-harness", type="project", aliases=("eval harness",))
    g.add_entity("python", type="skill")
    g.add_entity("neo4j", type="skill")
    g.add_entity("postgres", type="skill")
    g.add_relation("alice", "WORKS_ON", "rag-engine")
    g.add_relation("bob", "WORKS_ON", "eval-harness")
    g.add_relation("rag-engine", "USES", "python")
    g.add_relation("rag-engine", "USES", "neo4j")
    g.add_relation("eval-harness", "USES", "python")
    g.add_relation("eval-harness", "USES", "postgres")
    return g


def test_traverse_multihop_scores_and_order():
    rows = _org_graph().traverse(["neo4j"], max_hops=2)
    by_node = {r["node"]: r for r in rows}
    assert by_node["rag-engine"]["hops"] == 1
    assert by_node["rag-engine"]["score"] == 0.5
    assert rows[0]["node"] == "neo4j"


def test_traverse_terminates_with_cycles():
    g = _org_graph()
    g.add_relation("alice", "KNOWS", "bob")  # extra edges, more cycles
    rows = g.traverse(["alice"], max_hops=5)
    assert rows  # terminates
    assert len({r["node"] for r in rows}) == len(rows)  # no duplicates


def test_unknown_relation_endpoint_raises():
    g = _org_graph()
    with pytest.raises(ValueError, match="unknown entity"):
        g.add_relation("alice", "KNOWS", "mallory")


def test_extract_entities_matches_alias_and_ignores_unknown():
    g = _org_graph()
    assert "eval-harness" in g.extract_entities("Who works on the eval harness?")
    assert g.extract_entities("gobbledygook xyz") == []


def test_graphrag_returns_expected_evidence():
    rows = GraphRAG(_org_graph()).retrieve("Which projects use Neo4j?", top_k=3)
    top = [node for node, _text, _score in rows]
    assert "rag-engine" in top
    assert all(text for _n, text, _s in rows)  # evidence text attached


def test_graphrag_empty_seeds_returns_nothing():
    assert GraphRAG(_org_graph()).retrieve("gobbledygook xyz") == []


def test_cypher_builder_documents_the_swap():
    """The live query must mirror the in-memory engine: seeds at hop 0, and each
    node returned once at its MINIMUM hops.

    The previous form UNIONed seeds-at-0 with paths of length 1..N, which returned
    a node once per path length. That did NOT match the in-memory engine, despite
    the old comment here claiming it did — duplicates showed up on a diamond and on
    a cycle. Recorded in evals/live_neo4j_parity.py.
    """
    query = cypher_expand(2)
    assert "MATCH p" in query
    assert "$seeds" in query
    assert "*0..2" in query          # zero-length match includes the seeds
    assert "UNION" not in query      # superseded: it duplicated nodes
    assert "min(length(p))" in query  # one row per node, at minimum hops


def test_neo4j_store_missing_driver_message():
    try:
        import neo4j  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="pip install neo4j"):
            Neo4jGraphStore()
    else:
        pytest.skip("neo4j driver installed; live test skipped")
