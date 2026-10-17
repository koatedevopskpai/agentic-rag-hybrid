# Evidence index

A reviewer's route through this repository. Every claim points at something in the
repo or at a dated run.

**Five-minute path:** this file → [what "dense" means here](#what-dense-means-here)
→ [the strongest evidence](#the-strongest-evidence) →
[reproduce](#how-to-reproduce) → [what this does not establish](#what-this-does-not-establish).

---

## What it is

Hybrid retrieval over a knowledge graph: BM25 (sparse) fused with a deterministic
signature representation (dense) by reciprocal rank fusion, a groundedness filter,
and entity-seeded graph expansion with an optional Neo4j backend.

| | |
|---|---|
| Scope | **Public reference build.** Deterministic, offline-testable retrieval |
| Author | Koate Kpai — https://github.com/koatedevopskpai |
| Entry points | `src/agentic_rag_hybrid/` (`hybrid.py`, `graph.py`, `neo4j_store.py`) |

## What "dense" means here

`embedding.py` is a **deterministic signature** stand-in, not a trained semantic
model. The README says so and this file repeats it because it is the single most
likely thing for a reviewer to misread: this repository demonstrates the hybrid
pipeline, rank fusion and graph expansion — it does **not** demonstrate learned
embeddings, and no result here should be quoted as semantic-search quality.

## My contribution

- The hybrid pipeline: BM25, signature vectors, reciprocal rank fusion and the
  groundedness filter.
- Entity-seeded graph expansion and the `Neo4jGraphStore` adapter.
- The **live parity harness** and the **supernode benchmark**, and the two defects
  they found (below).

---

## The strongest evidence

### 1. Live Neo4j parity: 15/15, after fixing a real defect

A parity harness seeds a graph with a diamond, a cycle, an alias and an isolated
node into a real Neo4j instance and compares it against the in-memory engine.

It found that the shipped query did **not** match the engine. `cypher_expand`
UNIONed seeds-at-hop-0 with `MATCH p = (s)-[*1..N]-(m)`, which returns one row **per
path length**:

```
BEFORE:  11 rows for 9 distinct nodes on a diamond
         10 rows for 8 distinct nodes over a cycle
         ordered sequence parity FAILED
AFTER:   15/15 probes pass
```

Duplicate nodes would inflate any result shaped by `top_k`. Fixed with a
zero-length pattern and `min(length(p))` per node. On the way, a second subtlety
was found and recorded rather than explained away: binding the seeds directly and
aggregating afterwards still duplicated rows on Neo4j 2026.09.0, so the seeds are
collected and unwound first. The mechanism is **not** established; the working query
is.

Method and limits: [`docs/H03_H04.md`](docs/H03_H04.md),
[`evals/LIVE_NEO4J.md`](evals/LIVE_NEO4J.md).

### 2. A supernode benchmark that measures *work*, not wall-clock

`evals/high_degree_bench.py` builds a hub with N leaves plus a decoy branch and
measures each bound using `dbHits` from `PROFILE` — a deterministic proxy for work,
rather than a timing that varies with row transfer.

At N = 2000:

| Variant | rows | **dbHits** | median |
|---|---|---|---|
| unbounded 2-hop | 2401 | **9210** | 48.5 ms |
| capped `limit=10` | 10 | **9210** | 5.9 ms |
| hop-bounded (`max_hops=1`) | 2 | **11** | 2.8 ms |
| relationship-filtered | 2001 | **8010** | 42.7 ms |

**A result `LIMIT` bounds the rows returned, not the work performed** — `dbHits` is
identical to unbounded. A hop bound does bound work. A relationship filter reduces
work in proportion to the subgraph it skips. On wall-clock alone the capped variant
would have looked like a win; the deterministic metric says otherwise.

Report: [`evals/HIGH_DEGREE_REPORT.md`](evals/HIGH_DEGREE_REPORT.md).

### 3. The assertion-based evaluation gate

`evals/evaluate.py` runs 8 golden cases — 5 text-retrieval and 3 graph-expansion —
and fails on any mismatch. `ALL golden cases pass (8 cases)`. Offline suite green
with coverage above the enforced 80% floor (92% on the run recorded in
`docs/H03_H04.md`).

---

## How to reproduce

**Offline, free, no database:**

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python evals/evaluate.py          # 8 golden cases
```

**Live, local Docker, no cloud account:**

```bash
docker run -d --name neo4j-h04 -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/testpassword123 neo4j:latest
pip install neo4j
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=testpassword123

python evals/live_neo4j_parity.py    # 15/15
python evals/high_degree_bench.py    # writes evals/HIGH_DEGREE_REPORT.md

docker stop neo4j-h04 && docker rm neo4j-h04
```

Both live scripts **skip cleanly with exit 0 when `NEO4J_URI` is unset**, which is
why CI needs no database. Residual test counts in any report are per-run; the
durable statement is the committed report.

---

## One success trace, one failure

**Success.** `GraphRAG(graph).retrieve("Who works on the RAG engine?")` returns nodes
that are reachable from a query seed, so groundedness holds by construction rather
than by a filter that might miss.

**Failure, and the honest finding.** A result `LIMIT` does not protect the database
from a supernode — the expansion still visits every reachable node. The honest
mitigation is a hop bound and a relationship filter, not a cap applied afterwards.
That is measured above, not asserted.

---

## Decisions

| Decision | Where |
|---|---|
| Hybrid + RRF + groundedness, and why deterministic-first | [`README.md`](README.md) |
| Why the traversal uses `min(length(p))` and a collect-then-unwind step | `neo4j_store.py` docstrings, [`docs/H03_H04.md`](docs/H03_H04.md) |
| Which bounds actually reduce work | [`evals/HIGH_DEGREE_REPORT.md`](evals/HIGH_DEGREE_REPORT.md) |
| Live Neo4j verification, then and now | [`evals/LIVE_NEO4J.md`](evals/LIVE_NEO4J.md) |

---

## What this does not establish

- **Learned semantic retrieval.** "Dense" is a deterministic signature
  representation. Swapping in real embeddings is a documented next step, not
  something demonstrated.
- **Production scale.** Benchmark graphs are synthetic and ≤ 2401 nodes; the parity
  graph is a handful of nodes.
- **That the timings transfer.** They are one machine on one day and are not
  portable; `dbHits` is the durable figure and it is Neo4j-version-specific.
- **Degree-aware expansion.** No APOC or GDS in the stock image (`apoc.version()`
  and `gds.version()` both return *no such function*), so no bounded expansion
  beyond hop count and relationship type is claimed.
- **Anything commercial.** No customers, no uptime.

## Known weaknesses

- Seed extraction matches single-character entity ids in almost any query — an
  entity with id `e` is extracted from "what is the Alias Target". Recorded rather
  than silently changed, because a fix alters retrieval behaviour.
- The evaluation gate is 8 cases: a regression check, not a retrieval benchmark.
- The query-shape caveat (collect-then-unwind) is observed behaviour on one Neo4j
  version; the reason is unestablished.
