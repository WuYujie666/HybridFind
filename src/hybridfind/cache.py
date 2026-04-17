"""Dense embedding cache utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_dense_cache(
    path: str | Path,
    *,
    doc_ids: list[str],
    doc_embeddings: list[list[float]],
    model_name: str,
    normalize_embeddings: bool,
) -> None:
    payload = {
        "doc_ids": doc_ids,
        "doc_embeddings": doc_embeddings,
        "model_name": model_name,
        "normalize_embeddings": normalize_embeddings,
        "document_count": len(doc_ids),
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_dense_cache(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dense_cache_matches(
    cache_payload: dict[str, Any],
    *,
    doc_ids: list[str],
    model_name: str,
    normalize_embeddings: bool,
) -> bool:
    return (
        cache_payload.get("doc_ids") == doc_ids
        and cache_payload.get("model_name") == model_name
        and cache_payload.get("normalize_embeddings") == normalize_embeddings
        and cache_payload.get("document_count") == len(doc_ids)
    )
