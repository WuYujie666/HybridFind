"""Tests for HybridFind core engine."""

from __future__ import annotations

import math

import pytest

from hybridfind.config import SearchConfig
from hybridfind.core import BM25Searcher, DenseSearcher, Document, HybridSearch, reciprocal_rank_fusion
from hybridfind.utils import cosine_similarity, dense_cosine_similarity, tokenize


SAMPLE_TEXTS = [
    'Python is a great programming language for data science',
    'Machine learning models require large datasets for training',
    'Natural language processing enables computers to understand text',
    'Deep learning uses neural networks with many layers',
    'Information retrieval systems help users find relevant documents',
    'Search engines combine keyword matching with semantic understanding',
    'Vector databases store embeddings for similarity search',
    'BM25 is a classic algorithm used in full-text search',
]

SAMPLE_IDS = [f'doc-{i}' for i in range(len(SAMPLE_TEXTS))]
SAMPLE_META = [{'category': 'tech' if i % 2 == 0 else 'ml'} for i in range(len(SAMPLE_TEXTS))]


class FakeEncoder:
    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text, remove_stopwords=False):
            slot = sum(ord(ch) for ch in token) % self.dim
            vec[slot] += 1.0
        norm = math.sqrt(sum(value * value for value in vec))
        if norm:
            vec = [value / norm for value in vec]
        return vec


@pytest.fixture
def engine() -> HybridSearch:
    hs = HybridSearch(dense_encoder=FakeEncoder())
    hs.add_documents(texts=SAMPLE_TEXTS, ids=SAMPLE_IDS, metadatas=SAMPLE_META)
    hs.build_indices()
    return hs


class TestBM25Searcher:
    def test_basic_search(self) -> None:
        docs = [
            Document(doc_id='1', text='hello world', tokens=tokenize('hello world')),
            Document(doc_id='2', text='goodbye world', tokens=tokenize('goodbye world')),
        ]
        bm25 = BM25Searcher()
        bm25.index(docs)
        results = bm25.search(tokenize('hello'), top_k=2)
        assert len(results) >= 1
        assert results[0][0] == 0
        assert results[0][1] > 0

    def test_no_match_returns_empty(self) -> None:
        docs = [Document(doc_id='1', text='alpha beta', tokens=tokenize('alpha beta'))]
        bm25 = BM25Searcher()
        bm25.index(docs)
        assert bm25.search(tokenize('zzz_nonexistent_term'), top_k=5) == []


class TestDenseSearcher:
    def test_dense_relevance(self) -> None:
        docs = [
            Document(doc_id='1', text='machine learning algorithms', tokens=tokenize('machine learning algorithms')),
            Document(doc_id='2', text='cooking recipes bread', tokens=tokenize('cooking recipes bread')),
        ]
        ds = DenseSearcher(encoder=FakeEncoder())
        ds.index(docs)
        results = ds.search('learning algorithms', top_k=2)
        assert len(results) >= 1
        assert results[0][0] == 0

    def test_dense_cache_round_trip(self, tmp_path) -> None:
        docs = [
            Document(doc_id='1', text='machine learning algorithms', tokens=tokenize('machine learning algorithms')),
            Document(doc_id='2', text='cooking recipes bread', tokens=tokenize('cooking recipes bread')),
        ]
        ds = DenseSearcher(encoder=FakeEncoder())
        ds.index(docs)
        cache_path = tmp_path / 'dense_cache.json'
        ds.save_cache(cache_path)

        reloaded = DenseSearcher(encoder=FakeEncoder())
        assert reloaded.load_cache(cache_path, docs) is True
        assert reloaded.search('learning algorithms', top_k=2)[0][0] == 0


class TestReciprocalRankFusion:
    def test_rrf_merges_rankings(self) -> None:
        fused = reciprocal_rank_fusion([[(0, 5.0), (1, 3.0)], [(1, 9.0), (0, 4.0)]], k=60)
        assert {idx for idx, _ in fused} == {0, 1}

    def test_rrf_respects_weights(self) -> None:
        fused = reciprocal_rank_fusion([[(0, 10.0)], [(1, 10.0)]], weights=[1.0, 0.0], k=60)
        assert fused[0][0] == 0


class TestHybridSearch:
    def test_search_returns_results(self, engine: HybridSearch) -> None:
        results = engine.search('search algorithms text retrieval')
        assert len(results) > 0
        assert all(result.score > 0 for result in results)

    def test_metadata_filter(self, engine: HybridSearch) -> None:
        results = engine.search('programming language', metadata_filter={'category': 'tech'})
        assert all(result.metadata['category'] == 'tech' for result in results)

    def test_empty_query(self, engine: HybridSearch) -> None:
        assert engine.search('   ') == []

    def test_bm25_only_skips_dense_until_needed(self) -> None:
        cfg = SearchConfig(bm25_weight=1.0, dense_weight=0.0)
        hs = HybridSearch(config=cfg, dense_encoder=FakeEncoder())
        hs.add_documents(texts=SAMPLE_TEXTS, ids=SAMPLE_IDS)
        assert hs._dense_ready is False
        assert len(hs.search('BM25 full text search')) > 0

    def test_dense_cache_reuse(self, tmp_path) -> None:
        cfg = SearchConfig(bm25_weight=0.0, dense_weight=1.0, dense_cache_path=str(tmp_path / 'cache.json'))
        hs = HybridSearch(config=cfg, dense_encoder=FakeEncoder())
        hs.add_documents(texts=SAMPLE_TEXTS, ids=SAMPLE_IDS)
        hs.ensure_dense_ready()

        other = HybridSearch(config=cfg, dense_encoder=FakeEncoder())
        other.add_documents(texts=SAMPLE_TEXTS, ids=SAMPLE_IDS)
        assert other.load_dense_cache() is True
        assert len(other.search('semantic search embeddings')) > 0


class TestUtils:
    def test_tokenize_removes_stopwords(self) -> None:
        tokens = tokenize('the quick brown fox is very fast')
        assert 'the' not in tokens
        assert 'quick' in tokens

    def test_cosine_identical_vectors(self) -> None:
        vec = {'a': 1.0, 'b': 2.0}
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors(self) -> None:
        assert cosine_similarity({'x': 1.0}, {'y': 1.0}) == pytest.approx(0.0)

    def test_dense_cosine_identical_vectors(self) -> None:
        vec = [1.0, 2.0, 3.0]
        assert dense_cosine_similarity(vec, vec) == pytest.approx(1.0)
