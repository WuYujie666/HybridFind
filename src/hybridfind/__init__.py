"""HybridFind - Hybrid semantic + keyword search library."""

__version__ = "0.1.0"

from hybridfind.config import SearchConfig
from hybridfind.core import (
    BM25Searcher,
    HybridSearch,
    SearchResult,
    VectorSearcher,
    reciprocal_rank_fusion,
)
from hybridfind.evaluation import (
    EvaluationQuery,
    ExperimentSpec,
    average_precision,
    default_experiments,
    evaluate_runs,
    ndcg_at_k,
    parse_cranfield_directory,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "HybridSearch",
    "BM25Searcher",
    "VectorSearcher",
    "SearchResult",
    "SearchConfig",
    "reciprocal_rank_fusion",
    "EvaluationQuery",
    "ExperimentSpec",
    "precision_at_k",
    "recall_at_k",
    "average_precision",
    "reciprocal_rank",
    "ndcg_at_k",
    "default_experiments",
    "evaluate_runs",
    "parse_cranfield_directory",
]
