"""Deterministic 'signature' embedding + cosine similarity.

No external model — term-frequency vectors over the token space. In production,
swap :class:`SignatureEmbedder` for real dense embeddings (pgvector / Vertex AI
embeddings); the hybrid + RRF layer is identical.
"""

from __future__ import annotations

import math

from .indexing import Corpus, tokenize


def _vector(tokens: list[str]) -> dict[str, int]:
    vec: dict[str, int] = {}
    for tok in tokens:
        vec[tok] = vec.get(tok, 0) + 1
    return vec


class SignatureEmbedder:
    def __init__(self, corpus: Corpus):
        self._vectors = {doc_id: _vector(tokenize(text)) for doc_id, text in corpus.docs.items()}

    def rank(self, query_tokens: list[str], top_k: int | None = None) -> list[tuple[str, float]]:
        qv = _vector(query_tokens)
        scored = [
            (doc_id, self._cosine(qv, self._vectors[doc_id]))
            for doc_id in self._vectors
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        return scored

    @staticmethod
    def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(cnt * b.get(tok, 0) for tok, cnt in a.items())
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)