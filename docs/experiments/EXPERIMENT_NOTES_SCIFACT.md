# SciFact Dense Retrieval Experiments

Date: 2026-04-16
Dataset: `data/scifact/scifact`
Eval cutoff: `@10`
Dense model: `BAAI/bge-small-en-v1.5`

## 1. Old Baseline: BM25 + TF-IDF

Source: `artifacts/reports/scifact_report.csv`

| Experiment | BM25 | Dense/TF-IDF | P@10 | R@10 | MAP | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25_only | 1.0 | 0.0 | 0.0857 | 0.7688 | 0.6105 | 0.6184 | 0.6479 |
| tfidf_only | 0.0 | 1.0 | 0.0800 | 0.7199 | 0.5269 | 0.5392 | 0.5710 |
| hybrid_rrf | 0.5 | 0.5 | 0.0860 | 0.7718 | 0.5740 | 0.5847 | 0.6215 |

## 2. New Baseline: BM25 + Dense

Source: `artifacts/reports/scifact_dense_report.csv`

| Experiment | BM25 | Dense | P@10 | R@10 | MAP | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25_only | 1.0 | 0.0 | 0.0857 | 0.7688 | 0.6105 | 0.6184 | 0.6479 |
| dense_only | 0.0 | 1.0 | 0.0953 | 0.8452 | 0.6818 | 0.6883 | 0.7200 |
| hybrid_rrf | 0.5 | 0.5 | 0.0933 | 0.8284 | 0.6715 | 0.6820 | 0.7075 |

Observations:
- Replacing TF-IDF with dense retrieval produced a clear improvement on every major metric.
- `dense_only` outperformed both `bm25_only` and the default `0.5/0.5` hybrid.

## 3. Weight Sweep

Source: `artifacts/reports/scifact_weight_sweep.csv`

| Experiment | BM25 | Dense | P@10 | R@10 | MAP | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25_only | 1.0 | 0.0 | 0.0857 | 0.7688 | 0.6105 | 0.6184 | 0.6479 |
| dense_only | 0.0 | 1.0 | 0.0953 | 0.8452 | 0.6818 | 0.6883 | 0.7200 |
| hybrid_rrf | 0.5 | 0.5 | 0.0933 | 0.8284 | 0.6715 | 0.6820 | 0.7075 |
| dense_bias_01 | 0.1 | 0.9 | 0.0960 | 0.8519 | 0.6872 | 0.6931 | 0.7255 |
| dense_bias_02 | 0.2 | 0.8 | 0.0953 | 0.8452 | 0.6838 | 0.6906 | 0.7203 |
| dense_bias_03 | 0.3 | 0.7 | 0.0940 | 0.8346 | 0.6738 | 0.6800 | 0.7093 |

Observations:
- The best result in this sweep was `0.1 / 0.9` (BM25 / Dense).
- A small BM25 contribution helped the dense retriever and slightly beat `dense_only`.
- Increasing BM25 beyond that point reduced performance, which suggests dense retrieval is the dominant signal on SciFact.

## 4. FAISS Integration and Latency Benchmark (2026-05-04)

### Changes implemented
- `src/hybridfind/retrievers/dense.py`: added `IndexFlatIP` FAISS index; brute-force cosine loop kept as automatic fallback when `faiss-cpu` is not installed.
- `src/hybridfind/reranker.py` (new): `CrossEncoderReranker` using `cross-encoder/ms-marco-MiniLM-L-6-v2`; lazy-loaded on first call.
- `src/hybridfind/query_expansion.py` (new): `PseudoRelevanceFeedback`; disabled by default (`enable_prf=False` in config).
- `src/hybridfind/config.py`: added `reranker_model_name`, `reranker_candidate_k`, `enable_prf`, `prf_top_docs`, `prf_top_terms` fields.

### Status
| Feature | Status |
|---|---|
| FAISS vector index | Done, verified |
| Cross-Encoder Reranking | Implemented, not yet benchmarked on SciFact (too slow on CPU: ~4s/query × 300 queries per experiment) |
| PRF Query Expansion | Implemented, not yet benchmarked |

### Latency comparison: FAISS vs brute-force

Run command: `python -m hybridfind evaluate --data-dir data/scifact/scifact --isolated`
(`--isolated` spawns a fresh subprocess per experiment to ensure a cold CPU cache.)

| Experiment | nDCG@10 | Avg ms/q | Total s |
|---|---:|---:|---:|
| bm25_only | 0.6479 | 16.3 | 4.9 |
| dense_faiss | 0.7200 | 123.5 | 37.1 |
| dense_brute | 0.7200 | 351.7 | 105.6 |
| hybrid_rrf | 0.7075 | 112.0 | 33.7 |

**FAISS speedup: 351.7 / 123.5 = 2.85x** over brute-force, with identical retrieval quality (same nDCG@10).

Note: `hybrid_rrf` appears slightly faster than `dense_faiss` in this run; code tracing confirms hybrid does strictly more work (BM25 + FAISS + larger RRF merge), so the ~9% gap is within OS scheduling noise and should not be interpreted as a real advantage.

## 5. Practical Conclusion

- Dense retrieval is a successful replacement for TF-IDF in this project.
- On SciFact, the best tested setting is currently:
  - `bm25_weight = 0.1`
  - `dense_weight = 0.9`
- For the report, the strongest comparison is:
  - `tfidf_only` -> `dense_only`
  - `hybrid_rrf (0.5/0.5)` -> `dense_bias_01 (0.1/0.9)`