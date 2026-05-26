"""Hybrid search engine orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hybridfind.config import SearchConfig
from hybridfind.embedding import TextEncoder
from hybridfind.fusion import reciprocal_rank_fusion
from hybridfind.query_expansion import PseudoRelevanceFeedback
from hybridfind.reranker import CrossEncoderReranker
from hybridfind.retrievers import BM25Searcher, DenseSearcher
from hybridfind.schemas import Document, SearchResult
from hybridfind.utils import tokenize


class HybridSearch:
    """Combines BM25 keyword search with dense retrieval via RRF."""

    def __init__(self, config: SearchConfig | None = None, dense_encoder: TextEncoder | None = None) -> None:
        self.config = config or SearchConfig()
        self.config.apply_hf_env()
        self.bm25 = BM25Searcher(k1=self.config.bm25_k1, b=self.config.bm25_b)
        self.dense = DenseSearcher(
            model_name=self.config.dense_model_name,
            normalize_embeddings=self.config.dense_normalize_embeddings,
            encoder=dense_encoder,
        )
        self.documents: list[Document] = []
        self._bm25_ready = False
        self._dense_ready = False
        self._reranker: CrossEncoderReranker | None = (
            CrossEncoderReranker(self.config.reranker_model_name)
            if self.config.reranker_model_name
            else None
        )

    def add_documents(
        self,
        texts: list[str],
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        self.documents = []
        for i, text in enumerate(texts):
            doc_id = ids[i] if ids else str(i)
            meta = metadatas[i] if metadatas else {}
            tokens = tokenize(text)
            self.documents.append(Document(doc_id=doc_id, text=text, metadata=meta, tokens=tokens))
        self._bm25_ready = False
        self._dense_ready = False
        self.bm25.index(self.documents)
        self._bm25_ready = True

    def build_indices(self) -> None:
        if not self._bm25_ready:
            self.bm25.index(self.documents)
            self._bm25_ready = True
        if self.config.dense_weight > 0 and not self._dense_ready:
            self.dense.index(self.documents)
            self._dense_ready = True

    def save_dense_cache(self, path: str | Path | None = None) -> None:
        cache_path = Path(path or self.config.resolved_dense_cache_path())
        if not self._dense_ready:
            self.dense.index(self.documents)
            self._dense_ready = True
        self.dense.save_cache(cache_path)

    def load_dense_cache(self, path: str | Path | None = None) -> bool:
        cache_path = Path(path or self.config.resolved_dense_cache_path())
        loaded = self.dense.load_cache(cache_path, self.documents)
        self._dense_ready = loaded
        return loaded

    def ensure_dense_ready(self) -> None:
        if self._dense_ready or self.config.dense_weight <= 0:
            return
        loaded = self.load_dense_cache()
        if loaded:
            return
        if not self.config.auto_build_dense_cache:
            raise RuntimeError('Dense cache is not available. Run index or enable auto_build_dense_cache.')
        self.dense.index(self.documents)
        self._dense_ready = True
        self.save_dense_cache()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        k = top_k or self.config.top_k
        if not query.strip():
            return []

        # PRF: expand the query with terms from top BM25 results before full retrieval
        effective_query = query
        if self.config.enable_prf and self._bm25_ready:
            prf = PseudoRelevanceFeedback(self.config.prf_top_docs, self.config.prf_top_terms)
            effective_query = prf.expand(query, self.bm25, self.documents)

        query_tokens = tokenize(effective_query)

        candidate_indices: set[int] | None = None
        if metadata_filter:
            candidate_indices = {
                idx
                for idx, doc in enumerate(self.documents)
                if all(doc.metadata.get(key) == val for key, val in metadata_filter.items())
            }

        bm25_results: list[tuple[int, float]] = []
        dense_results: list[tuple[int, float]] = []

        if self.config.bm25_weight > 0 and query_tokens:
            if not self._bm25_ready:
                self.bm25.index(self.documents)
                self._bm25_ready = True
            bm25_results = self.bm25.search(query_tokens, top_k=len(self.documents))

        if self.config.dense_weight > 0:
            self.ensure_dense_ready()
            dense_results = self.dense.search(effective_query, top_k=len(self.documents))

        if candidate_indices is not None:
            bm25_results = [(idx, sc) for idx, sc in bm25_results if idx in candidate_indices]
            dense_results = [(idx, sc) for idx, sc in dense_results if idx in candidate_indices]

        w_bm25, w_dense = self.config.effective_weights()
        fused = reciprocal_rank_fusion(
            [bm25_results, dense_results],
            weights=[w_bm25, w_dense],
            k=self.config.rrf_k,
        )

        # Build candidate list for reranker (or final result list)
        candidate_k = self.config.reranker_candidate_k if self._reranker else k
        candidates = [
            SearchResult(
                doc_id=self.documents[doc_idx].doc_id,
                score=score,
                text=self.documents[doc_idx].text,
                metadata=self.documents[doc_idx].metadata,
            )
            for doc_idx, score in fused[:candidate_k]
        ]

        if self._reranker:
            return self._reranker.rerank(query, candidates, top_k=k)

        return candidates[:k]
