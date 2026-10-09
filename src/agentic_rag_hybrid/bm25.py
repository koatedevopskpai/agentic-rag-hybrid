"""Okapi BM25 sparse ranking. Deterministic, stdlib-only."""

from __future__ import annotations

import math

from .indexing import Corpus, tokenize

K1 = 1.5
B = 0.75


class BM25Index:
    def __init__(self, corpus: Corpus, k1: float = K1, b: float = B):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len: dict[str, int] = {}
        self.term_freq: dict[str, dict[str, int]] = {}
        self.avgdl = 0.0
        self._build()

    def _build(self) -> None:
        doc_freq: dict[str, int] = {}
        total = 0
        for doc_id, text in self.corpus.docs.items():
            tokens = tokenize(text)
            self.doc_len[doc_id] = len(tokens)
            total += len(tokens)
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self.term_freq[doc_id] = tf
            for tok in tf:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
        self.avgdl = total / len(self.corpus.docs)
        self.doc_freq = doc_freq

    def _idf(self, term: str) -> float:
        n = len(self.corpus.docs)
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5)) + 1

    def score(self, query_tokens: list[str], doc_id: str) -> float:
        dl = self.doc_len[doc_id]
        tf = self.term_freq[doc_id]
        denom = 1 - self.b + self.b * (dl / self.avgdl)
        total = 0.0
        for tok in dict.fromkeys(query_tokens):
            f = tf.get(tok, 0)
            if f:
                total += self._idf(tok) * (f * (self.k1 + 1)) / (f + self.k1 * denom)
        return total

    def rank(self, query_tokens: list[str], top_k: int | None = None) -> list[tuple[str, float]]:
        scored = [(doc_id, self.score(query_tokens, doc_id)) for doc_id in self.corpus.docs]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        return scored