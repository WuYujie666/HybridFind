# HybridFind

**Hybrid semantic + keyword search** — combines BM25 lexical matching with dense embedding retrieval, fused via Reciprocal Rank Fusion (RRF). Dense document embeddings are cached on disk by default so search and evaluation can reuse them across runs.

---

## What Is This?

HybridFind is a hybrid information retrieval system for a course assignment. Given a query, it retrieves relevant documents from a collection using two complementary strategies:

- **BM25** — classic keyword matching, good at exact term hits (names, abbreviations, IDs)
- **Dense Embedding** — semantic matching via `BAAI/bge-small-en-v1.5`, good at synonymy and paraphrase

The two ranking lists are merged by **Reciprocal Rank Fusion (RRF)**, which combines rank positions rather than raw scores, making the fusion robust and easy to tune.

Previous versions used TF-IDF as the second retriever. It was replaced with dense retrieval after experiments showed clear improvement on every major metric (see [Experiment Results](#experiment-results)).

---

## System Architecture

```
Query
  │
  ├──► BM25 Retriever ──► ranking list A
  │
  ├──► Dense Retriever (FAISS / brute-force) ──► ranking list B
  │
  └──► RRF Fusion ──► merged ranking
        │
        └──► (optional) Cross-Encoder Reranker ──► final ranking
```

### Key modules

| Module | File | Purpose |
|---|---|---|
| Engine | `src/hybridfind/engine.py` | Orchestrates retrieval + fusion + optional reranking |
| BM25 | `src/hybridfind/retrievers/bm25.py` | BM25 keyword search |
| Dense | `src/hybridfind/retrievers/dense.py` | Dense embedding search with FAISS (falls back to brute-force) |
| Fusion | `src/hybridfind/fusion.py` | Reciprocal Rank Fusion |
| Embedding | `src/hybridfind/embedding.py` | Sentence-Transformer encoder wrapper |
| Cache | `src/hybridfind/cache.py` | Dense cache save/load/validate |
| Config | `src/hybridfind/config.py` | All configurable parameters |
| Evaluation | `src/hybridfind/evaluation.py` | BEIR-style offline evaluation |
| Reranker | `src/hybridfind/reranker.py` | Cross-Encoder reranking (implemented, not benchmarked) |
| PRF | `src/hybridfind/query_expansion.py` | Pseudo Relevance Feedback (implemented, not benchmarked) |

---

## Project Layout

```
src/hybridfind/        Retrieval engine, retrievers, cache, CLI
data/scifact/          SciFact benchmark dataset
docs/                  Search corpus (IR course notes)
artifacts/cache/       Generated index snapshot and dense cache
artifacts/reports/     Generated experiment reports
```

---

## Installation

```bash
pip install -e .
```

Optional — install FAISS for faster dense search (2.85x speedup, identical retrieval quality):

```bash
pip install faiss-cpu
```

If FAISS is not installed, the system automatically falls back to brute-force cosine search.

---

## Quick Start

### Index documents

```bash
hybridfind index docs/ --extensions ".txt,.md"
```

This reads all matching files, builds a BM25 index, encodes dense embeddings, and saves both to `artifacts/cache/`.

### Search

```bash
hybridfind search "hybrid search algorithms" --top-k 5
hybridfind search "query" --bm25-weight 0.1 --dense-weight 0.9
```

### Evaluate on SciFact

```bash
hybridfind evaluate --data-dir data/scifact/scifact --dataset scifact --eval-k 10
```

With output files:

```bash
hybridfind evaluate --data-dir data/scifact/scifact --dataset scifact --eval-k 10   --output-csv artifacts/reports/scifact_run.csv   --output-json artifacts/reports/scifact_run.json
```

### Custom weight experiments

```bash
hybridfind evaluate --data-dir data/scifact/scifact --dataset scifact --eval-k 10   --weight-pair "dense_bias_01:0.1,0.9"   --weight-pair "dense_bias_02:0.2,0.8"
```

### Run only specific experiments

```bash
hybridfind evaluate --data-dir data/scifact/scifact --dataset scifact   --only bm25_only --only dense_only
```

### Isolated timing (cold cache per experiment)

```bash
hybridfind evaluate --data-dir data/scifact/scifact --dataset scifact --isolated
```

### Limit queries (quick debug)

```bash
hybridfind evaluate --data-dir data/scifact/scifact --dataset scifact --max-queries 10
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `bm25_weight` | `0.5` | Weight for BM25 keyword results |
| `dense_weight` | `0.5` | Weight for dense retrieval results |
| `dense_model_name` | `BAAI/bge-small-en-v1.5` | Sentence-Transformers model for embeddings |
| `dense_normalize_embeddings` | `True` | L2-normalize dense vectors before similarity search |
| `dense_cache_path` | `None` | Override for dense cache path (default: `artifacts/cache/hybridfind_dense_cache.json`) |
| `auto_build_dense_cache` | `True` | Rebuild and save dense cache when missing |
| `rrf_k` | `60` | RRF smoothing constant |
| `bm25_k1` | `1.2` | BM25 term-frequency saturation |
| `bm25_b` | `0.75` | BM25 length normalization |
| `top_k` | `10` | Number of results to return |
| `reranker_model_name` | `None` | Cross-Encoder model name; `None` disables reranking |
| `reranker_candidate_k` | `100` | Candidates passed to reranker before final top-k trim |
| `enable_prf` | `False` | Enable Pseudo Relevance Feedback query expansion for BM25 |
| `prf_top_docs` | `3` | Top documents used for PRF expansion |
| `prf_top_terms` | `5` | Expansion terms added by PRF |

---

## Experiment Results

All results on **SciFact** dataset, eval cutoff `@10`.

### BM25 + TF-IDF (old baseline)

| Experiment | BM25 | TF-IDF | P@10 | R@10 | MAP | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25_only | 1.0 | 0.0 | 0.0857 | 0.7688 | 0.6105 | 0.6184 | 0.6479 |
| tfidf_only | 0.0 | 1.0 | 0.0800 | 0.7199 | 0.5269 | 0.5392 | 0.5710 |
| hybrid_rrf | 0.5 | 0.5 | 0.0860 | 0.7718 | 0.5740 | 0.5847 | 0.6215 |

### BM25 + Dense (current system)

| Experiment | BM25 | Dense | P@10 | R@10 | MAP | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25_only | 1.0 | 0.0 | 0.0857 | 0.7688 | 0.6105 | 0.6184 | 0.6479 |
| dense_only | 0.0 | 1.0 | 0.0953 | 0.8452 | 0.6818 | 0.6883 | 0.7200 |
| hybrid_rrf | 0.5 | 0.5 | 0.0933 | 0.8284 | 0.6715 | 0.6820 | 0.7075 |

### Weight sweep

| Experiment | BM25 | Dense | P@10 | R@10 | MAP | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25_only | 1.0 | 0.0 | 0.0857 | 0.7688 | 0.6105 | 0.6184 | 0.6479 |
| dense_only | 0.0 | 1.0 | 0.0953 | 0.8452 | 0.6818 | 0.6883 | 0.7200 |
| hybrid_rrf | 0.5 | 0.5 | 0.0933 | 0.8284 | 0.6715 | 0.6820 | 0.7075 |
| dense_bias_01 | 0.1 | 0.9 | **0.0960** | **0.8519** | **0.6872** | **0.6931** | **0.7255** |
| dense_bias_02 | 0.2 | 0.8 | 0.0953 | 0.8452 | 0.6838 | 0.6906 | 0.7203 |
| dense_bias_03 | 0.3 | 0.7 | 0.0940 | 0.8346 | 0.6738 | 0.6800 | 0.7093 |

### FAISS vs brute-force latency

Run with `--isolated` for cold-cache timing:

| Experiment | nDCG@10 | Avg ms/query | Total s |
|---|---:|---:|---:|
| bm25_only | 0.6479 | 16.3 | 4.9 |
| dense_faiss | 0.7200 | 123.5 | 37.1 |
| dense_brute | 0.7200 | 351.7 | 105.6 |
| hybrid_rrf | 0.7075 | 112.0 | 33.7 |

**FAISS speedup: 2.85x** over brute-force, with identical retrieval quality.

### Key conclusions

1. **Dense > TF-IDF**: Replacing TF-IDF with dense retrieval improved every major metric.
2. **Best weight is 0.1 / 0.9**: A small BM25 contribution helps dense retrieval and slightly beats `dense_only`. Increasing BM25 beyond that reduces performance.
3. **FAISS is worth it**: 2.85x faster than brute-force with no quality loss.

---

## 未完成功能

以下功能已实现但**未完成基准测试**，默认关闭：

| 功能 | 配置项 | 状态 |
|---|---|---|
| Cross-Encoder 重排序 | `reranker_model_name` | 已实现；CPU 上太慢（~4s/query），未跑完 SciFact 评测 |
| 伪相关反馈（PRF）查询扩展 | `enable_prf` | 已实现；未评测 |

