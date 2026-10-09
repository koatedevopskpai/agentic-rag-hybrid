# agentic-rag-hybrid

**Planner–Executor RAG with hybrid retrieval** — demo/enterprise retrieval that
avoids the classic failure modes: sparse-only misses synonyms, dense-only misses
exact terms, and ungrounded output gets trusted. This repo fuses **BM25
(sparse)** with a deterministic **dense signature** via **Reciprocal Rank
Fusion**, then filters by **groundedness** — all deterministic, no LLM in the
retrieval path.

```
query ─▶ Planner (deterministic decompose)
            │  sub-query 1   sub-query 2
            ▼
        Executor
            ├─ BM25 index ─────────────┐
            └─ signature embeddings ───┴─▶ RRF (hybrid rank)
                                              ▼
                                     merge sub-queries
                                              ▼
                                     groundedness filter (token overlap)
                                              ▼
                                     ranked, grounded documents
```

## Run it yourself (60s)

```bash
pip install -e ".[dev]"
python -m pytest            # unit tests
python evals/evaluate.py    # golden-case eval gate
```

```python
from agentic_rag_hybrid import Corpus, retrieve

corpus = Corpus({
    "d1": "Red pandas eat bamboo and live in the Himalayas.",
    "d4": "RAG answers questions by retrieving evidence before generating.",
    "d6": "Retrieval augmented generation combines a corpus with an LLM.",
})

for doc_id, score in retrieve(corpus, "retrieval augmented generation", top_k=3):
    print(doc_id, round(score, 4), corpus[doc_id])
```

## Why deterministic-first?

- **Reproducible**: the same corpus + query always returns the same top-k.
- **Testable**: golden sets assert relevancy and groundedness in CI (the eval gate).
- **Honest**: no fake-AI in retrieval; the LLM (optional, roadmap) is a *reranker*
  on top, not the retriever.

## Swapping in real embeddings

`SignatureEmbedder` is a deterministic stand-in so the demo runs offline and
deterministic. In production, replace it with **pgvector or Vertex AI vector
search** — the hybrid + RRF + groundedness pipeline is unchanged.

## Layout

```
src/agentic_rag_hybrid/
  indexing.py    # tokenizer + Corpus
  bm25.py        # Okapi BM25 sparse ranker
  embedding.py   # deterministic signature embeddings + cosine
  hybrid.py      # reciprocal rank fusion + hybrid_rank()
  executor.py    # Planner + Executor + retrieve()
evals/           # golden queries + assertion-based eval gate
tests/           # unit tests
```

## Roadmap

- pgvector / Vertex AI embedder behind a `VectorStore` protocol
- Optional LLM rerank with the same resilience discipline as
  `multi-agent-optimizer`
- FastAPI serving + Cloud Run deploy (see `cloud-run-ai-golden-path`)

## Licence

MIT — see [LICENSE](LICENSE). © 2026 Koate Kpai.