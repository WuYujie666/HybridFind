"""Search configuration for HybridFind."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class SearchConfig(BaseModel):
    """Configuration for the hybrid search engine."""

    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for BM25 keyword results")
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for dense retrieval results")
    rrf_k: int = Field(default=60, gt=0, description="RRF constant k")
    bm25_k1: float = Field(default=1.2, gt=0.0, description="BM25 term-frequency saturation parameter")
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0, description="BM25 length-normalization parameter")
    dense_model_name: str = Field(default="BAAI/bge-small-en-v1.5", description="Dense embedding model name")
    dense_normalize_embeddings: bool = Field(default=True, description="Whether to L2-normalize dense embeddings")
    dense_cache_path: str | None = Field(default=None, description="Optional dense cache path")
    auto_build_dense_cache: bool = Field(default=True, description="Automatically build dense cache when missing")
    top_k: int = Field(default=10, gt=0, description="Number of results to return")
    reranker_model_name: str | None = Field(default=None, description="Cross-encoder model name; None disables reranking")
    reranker_candidate_k: int = Field(default=100, gt=0, description="Number of candidates passed to reranker before final top_k trim")
    enable_prf: bool = Field(default=False, description="Enable Pseudo Relevance Feedback query expansion for BM25")
    prf_top_docs: int = Field(default=3, gt=0, description="Number of top documents used for PRF expansion")
    prf_top_terms: int = Field(default=5, gt=0, description="Number of expansion terms added by PRF")
    hf_mirror: str | None = Field(default=None, description="HuggingFace mirror endpoint (e.g. https://hf-mirror.com). Set HF_ENDPOINT env var if not specified here.")

    def apply_hf_env(self) -> None:
        """Apply HuggingFace environment variables from config.

        If ``hf_mirror`` is set, it is mapped to the ``HF_ENDPOINT`` environment
        variable (unless the user already set one externally).  This should be
        called before loading any HuggingFace model.
        """
        import os
        if self.hf_mirror:
            os.environ.setdefault("HF_ENDPOINT", self.hf_mirror)

    def effective_weights(self) -> tuple[float, float]:
        total = self.bm25_weight + self.dense_weight
        if total == 0:
            return 0.5, 0.5
        return self.bm25_weight / total, self.dense_weight / total

    def resolved_dense_cache_path(self) -> Path:
        return Path(self.dense_cache_path or 'artifacts/cache/hybridfind_dense_cache.json')
