"""Embedding utilities for dense retrieval."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class TextEncoder(Protocol):
    """Minimal protocol for dense text encoders."""

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of documents into dense vectors."""

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of queries into dense vectors."""


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for dense retrieval. Install project dependencies first."
        ) from exc
    return SentenceTransformer(model_name)


class SentenceTransformerEncoder:
    """Thin adapter around sentence-transformers for project use."""

    def __init__(self, model_name: str, normalize_embeddings: bool = True) -> None:
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = _load_sentence_transformer(self.model_name)
        embeddings = model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)
