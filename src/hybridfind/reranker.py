"""Cross-encoder reranker for second-stage result refinement."""

from __future__ import annotations

from hybridfind.schemas import SearchResult


class CrossEncoderReranker:
    """Reranks search results using a cross-encoder model.

    The cross-encoder jointly encodes (query, document) pairs and produces
    a relevance score that is more accurate than bi-encoder cosine similarity,
    at the cost of being slower. Use it on a small candidate set (e.g. top-100)
    after initial retrieval.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None  # lazy-loaded on first rerank() call

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for reranking. "
                    "It should already be installed as a project dependency."
                ) from exc
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Score each (query, document) pair and return the top_k highest-scoring results."""
        if not results:
            return results
        model = self._load()
        pairs = [(query, r.text) for r in results]
        scores = model.predict(pairs)
        reranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [r for r, _ in reranked[:top_k]]
