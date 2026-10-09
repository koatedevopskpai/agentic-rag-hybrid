"""Tokenization + in-memory corpus. Deterministic, dependency-free."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _WORD.findall(text.lower())


class Corpus:
    """Simple id -> text document store."""

    def __init__(self, docs: dict[str, str]):
        if not docs:
            raise ValueError("corpus must contain at least one document")
        self.docs = dict(docs)

    def __getitem__(self, doc_id: str) -> str:
        return self.docs[doc_id]

    def token_stream(self, doc_id: str) -> list[str]:
        return tokenize(self.docs[doc_id])