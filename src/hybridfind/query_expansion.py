"""Pseudo Relevance Feedback (PRF) query expansion for BM25."""

from __future__ import annotations

from hybridfind.schemas import Document
from hybridfind.utils import tokenize


class PseudoRelevanceFeedback:
    """Expands a query using terms from the top BM25 results.

    After an initial BM25 search, this takes the top-ranked documents,
    counts term frequencies across them (excluding the original query
    terms), and appends the most frequent new terms to the query.
    This helps BM25 find documents that share vocabulary with relevant
    documents but not with the original short query.
    """

    def __init__(self, top_docs: int = 3, top_terms: int = 5) -> None:
        self.top_docs = top_docs
        self.top_terms = top_terms

    def expand(self, query: str, bm25_searcher, documents: list[Document]) -> str:
        """Return an expanded query string, or the original query if expansion fails."""
        query_tokens = set(tokenize(query))
        top_results = bm25_searcher.search(tokenize(query), top_k=self.top_docs)
        if not top_results:
            return query

        term_freq: dict[str, int] = {}
        for doc_idx, _ in top_results:
            for token in documents[doc_idx].tokens:
                if token not in query_tokens:
                    term_freq[token] = term_freq.get(token, 0) + 1

        expansion_terms = sorted(term_freq, key=lambda t: term_freq[t], reverse=True)[: self.top_terms]
        if not expansion_terms:
            return query
        return query + " " + " ".join(expansion_terms)
