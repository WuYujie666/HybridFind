"""Dense retriever implementation with cache support."""

from __future__ import annotations

from pathlib import Path

from hybridfind.cache import dense_cache_matches, load_dense_cache, save_dense_cache
from hybridfind.embedding import SentenceTransformerEncoder, TextEncoder
from hybridfind.schemas import Document
from hybridfind.utils import dense_cosine_similarity


class DenseSearcher:
    """Dense embedding retrieval with cosine similarity and cache support."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        normalize_embeddings: bool = True,
        encoder: TextEncoder | None = None,
    ) -> None:
        self.docs: list[Document] = []
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.encoder = encoder or SentenceTransformerEncoder(
            model_name=model_name,
            normalize_embeddings=normalize_embeddings,
        )
        self.doc_embeddings: list[list[float]] = []

    @property
    def is_indexed(self) -> bool:
        return bool(self.doc_embeddings)

    def index(self, docs: list[Document]) -> None:
        self.docs = docs
        self.doc_embeddings = self.encoder.encode_documents([doc.text for doc in docs]) if docs else []

    def save_cache(self, path: str | Path) -> None:
        save_dense_cache(
            path,
            doc_ids=[doc.doc_id for doc in self.docs],
            doc_embeddings=self.doc_embeddings,
            model_name=self.model_name,
            normalize_embeddings=self.normalize_embeddings,
        )

    def load_cache(self, path: str | Path, docs: list[Document]) -> bool:
        cache_path = Path(path)
        if not cache_path.exists():
            return False
        payload = load_dense_cache(cache_path)
        doc_ids = [doc.doc_id for doc in docs]
        if not dense_cache_matches(
            payload,
            doc_ids=doc_ids,
            model_name=self.model_name,
            normalize_embeddings=self.normalize_embeddings,
        ):
            return False
        self.docs = docs
        self.doc_embeddings = payload["doc_embeddings"]
        return True

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if not query.strip() or not self.doc_embeddings:
            return []

        query_embedding = self.encoder.encode_queries([query])[0]
        scored = [
            (idx, dense_cosine_similarity(query_embedding, doc_embedding))
            for idx, doc_embedding in enumerate(self.doc_embeddings)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(idx, sc) for idx, sc in scored[:top_k] if sc > 0]
