"""Planner-Executor: decompose a query into retrieval sub-tasks, then merge."""

from __future__ import annotations

import re

from .bm25 import BM25Index
from .embedding import SignatureEmbedder
from .hybrid import hybrid_rank
from .indexing import Corpus, tokenize

_SPLIT = re.compile(r"\b(and|&|,|;|plus|\.)\b|\s+\+\s+", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class Planner:
    """Deterministic decomposition. No LLM: split on connectors/sentences."""

    def decompose(self, query: str) -> list[str]:
        parts = _SPLIT.split(query)
        parts = [p.strip() for p in parts if p and p.strip()]
        if len(parts) <= 1:
            parts = [s for s in _SENTENCE.split(query.strip()) if s]
        parts = [p for p in parts if tokenize(p)]
        return parts or [query]


class Executor:
    def __init__(self, corpus: Corpus, per_subquery: int = 8, top_k: int | None = None):
        self._corpus = corpus
        self._bm25 = BM25Index(corpus)
        self._embedder = SignatureEmbedder(corpus)
        self._planner = Planner()
        self.per_subquery = per_subquery
        self.top_k = top_k

    def _groundedness(self, query_tokens: set[str], doc_id: str) -> bool:
        """Deterministic groundedness: the doc must share at least one query token."""
        doc_tokens = set(tokenize(self._corpus[doc_id]))
        return bool(query_tokens & doc_tokens)

    def retrieve(self, query: str) -> list[tuple[str, float]]:
        subqueries = self._planner.decompose(query)
        merged: dict[str, float] = {}
        all_tokens: set[str] = set()
        for sub in subqueries:
            tokens = tokenize(sub)
            all_tokens.update(tokens)
            for doc_id, score in hybrid_rank(
                self._bm25, self._embedder, tokens, per_rank=self.per_subquery
            ):
                merged[doc_id] = merged.get(doc_id, 0.0) + score

        # Groundedness filter: drop docs with zero token overlap with the query.
        grounded = [
            (doc_id, score) for doc_id, score in merged.items() if self._groundedness(all_tokens, doc_id)
        ]
        grounded.sort(key=lambda pair: pair[1], reverse=True)
        if self.top_k is not None:
            grounded = grounded[: self.top_k]
        return grounded


def retrieve(corpus: Corpus, query: str, top_k: int | None = 3) -> list[tuple[str, float]]:
    """One-call facade: planner -> hybrid search per sub-query -> merged + grounded."""
    return Executor(corpus, top_k=top_k).retrieve(query)