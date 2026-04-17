"""BM25 retriever implementation."""

from __future__ import annotations

import math

from hybridfind.schemas import Document


class BM25Searcher:
    """Classic Okapi BM25 scoring implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[Document] = []
        self.avgdl: float = 0.0
        self.doc_freqs: dict[str, int] = {}
        self.doc_len: list[int] = []
        self.n_docs: int = 0
        self.term_freqs: list[dict[str, int]] = []

    def index(self, docs: list[Document]) -> None:
        """Build the BM25 index from a list of documents."""
        self.docs = docs
        self.n_docs = len(docs)
        self.doc_freqs = {}
        self.term_freqs = []
        self.doc_len = []

        total_len = 0
        for doc in docs:
            tf: dict[str, int] = {}
            for token in doc.tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)
            dl = len(doc.tokens)
            self.doc_len.append(dl)
            total_len += dl
            for term in set(doc.tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avgdl = total_len / self.n_docs if self.n_docs else 0.0

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        scores: list[float] = [0.0] * self.n_docs
        for token in query_tokens:
            idf = self._idf(token)
            for idx in range(self.n_docs):
                tf = self.term_freqs[idx].get(token, 0)
                dl = self.doc_len[idx]
                if self.avgdl:
                    denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                else:
                    denom = tf + self.k1
                numerator = tf * (self.k1 + 1)
                scores[idx] += idf * (numerator / denom) if denom else 0.0

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, sc) for idx, sc in ranked[:top_k] if sc > 0]
