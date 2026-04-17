# HybridFind

**Hybrid semantic + keyword search** - combines BM25 lexical matching with dense embedding retrieval, fused via Reciprocal Rank Fusion (RRF). Dense document embeddings are cached on disk by default so search and evaluation can reuse them across runs.

## Project Layout

- `src/` - retrieval engine, retrievers, cache, CLI
- `data/scifact/` - SciFact dataset
- `artifacts/cache/` - generated index snapshot and dense cache
- `artifacts/reports/` - generated experiment reports
- `docs/experiments/` - saved experiment notes

## How It Works

1. `hybridfind index` loads documents and builds BM25.
2. Dense document embeddings are encoded once and stored in `artifacts/cache/hybridfind_dense_cache.json`.
3. `hybridfind search` loads the saved document snapshot and reuses dense cache when needed.
4. `hybridfind evaluate` supports BEIR-style datasets such as SciFact and reuses one shared dense cache across experiments.

## Installation

```bash
pip install -e .
```

Default dense model: `BAAI/bge-small-en-v1.5`.

## Quick Start

```bash
hybridfind index docs/ --extensions ".txt,.md"
hybridfind search "hybrid search algorithms" --top-k 5
hybridfind search "query" --bm25-weight 0.1 --dense-weight 0.9
hybridfind evaluate --data-dir data/scifact/scifact --dataset scifact --eval-k 10 --output-csv artifacts/reports/scifact_run.csv --output-json artifacts/reports/scifact_run.json
```

## Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `bm25_weight` | `0.5` | Weight for BM25 keyword results |
| `dense_weight` | `0.5` | Weight for dense retrieval results |
| `dense_model_name` | `BAAI/bge-small-en-v1.5` | Sentence-transformers model used for embeddings |
| `dense_normalize_embeddings` | `True` | Normalize dense vectors before similarity search |
| `dense_cache_path` | `None` | Optional override for dense cache path |
| `auto_build_dense_cache` | `True` | Rebuild and save dense cache when missing |
| `rrf_k` | `60` | RRF smoothing constant |
| `bm25_k1` | `1.5` | BM25 term-frequency saturation |
| `bm25_b` | `0.75` | BM25 length normalization |
| `top_k` | `10` | Number of results to return |
