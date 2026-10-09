"""Hybrid retrieval: Reciprocal Rank Fusion over sparse + dense rankings."""

from __future__ import annotations

from .bm25 import BM25Index
from .embedding import SignatureEmbedder


def rrf(*rankings: list[tuple[str, float]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of multiple doc-rank lists."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
    return ordered


def hybrid_rank(
    bm25: BM25Index,
    embedder: SignatureEmbedder,
    query_tokens: list[str],
    per_rank: int = 10,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Sparse (BM25) + dense (signature) rankings fused with RRF."""
    sparse = bm25.rank(query_tokens, top_k=per_rank)
    dense = embedder.rank(query_tokens, top_k=per_rank)
    fused = rrf(sparse, dense)
    if top_k is not None:
        fused = fused[:top_k]
    return fused