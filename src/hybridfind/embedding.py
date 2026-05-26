"""Embedding utilities for dense retrieval."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Protocol


def _apply_hf_env() -> None:
    """Apply HuggingFace environment variables before model loading.

    - ``HF_HUB_OFFLINE=1``: Force offline mode — only load from local cache,
      never contact the network.  Safe default when the model is already cached.
    - ``HF_ENDPOINT``: Mirror endpoint for downloading models (e.g.
      ``https://hf-mirror.com``).  Takes effect only when offline mode is *off*
      and the model is not yet cached locally.

    Both variables can be set externally before launching the process; the
    values here are fallback defaults only.
    """
    # Default to offline mode so that cached models load instantly without
    # network round-trips.  Set HF_HUB_OFFLINE=0 externally to allow downloads.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # If a mirror endpoint is configured (via env var or config), honour it.
    # The user can set HF_ENDPOINT=https://hf-mirror.com in their shell or .env.
    # os.environ.setdefault("HF_ENDPOINT", "")  — no default needed; the
    # huggingface_hub library already reads HF_ENDPOINT automatically.


class TextEncoder(Protocol):
    """Minimal protocol for dense text encoders."""

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of documents into dense vectors."""

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of queries into dense vectors."""


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    _apply_hf_env()
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
