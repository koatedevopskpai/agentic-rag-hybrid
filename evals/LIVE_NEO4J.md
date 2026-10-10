# Live Neo4j verification

**Date:** 2026-10-10 · **Cost: $0** (local Docker, no cloud spend)

The `Neo4jGraphStore` swap (documented as the production path) was verified
against a real Neo4j instance, not just the deterministic engine.

## Method

```bash
docker run -d --name neo4j-test -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/testpassword123 neo4j:latest
pip install neo4j
# seed golden_graph.json (:Node {id} + relationships), then:
#   local  = GraphRAG(KnowledgeGraph).retrieve(query)
#   live   = Neo4jGraphStore().traverse(seeds)   # UNION query, seeds at hop 0
```

## Result — 3/3 match exactly

| Query | In-memory top | Live Neo4j top |
|---|---|---|
| Which projects use Neo4j? | neo4j, rag-engine, alice | neo4j, rag-engine, alice |
| Who works on the eval harness? | eval-harness, bob, postgres | eval-harness, bob, postgres |
| What skills does the RAG engine need? | rag-engine, alice, neo4j, python | rag-engine, alice, neo4j, python |

One real difference found and fixed during this run: the first Cypher version
omitted seed nodes (paths start at hop 1) while the engine includes them at
hop 0. Fixed with a `UNION` — both now agree exactly.

Container removed after the run (`docker stop/rm neo4j-test`); nothing lingers.
