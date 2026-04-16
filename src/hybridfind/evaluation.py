"""Offline evaluation utilities for benchmark datasets such as Cranfield and BEIR."""

from __future__ import annotations

import csv
import json
import math
import pathlib
from dataclasses import dataclass
from statistics import mean

from hybridfind.config import SearchConfig
from hybridfind.core import HybridSearch


@dataclass(frozen=True)
class EvaluationQuery:
    """A benchmark query with a stable identifier."""

    query_id: str
    text: str


@dataclass(frozen=True)
class ExperimentSpec:
    """Configuration for a single retrieval experiment."""

    name: str
    bm25_weight: float
    vector_weight: float


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Precision@k for a ranked list."""
    if k <= 0:
        return 0.0
    top_ids = ranked_ids[:k]
    if not top_ids:
        return 0.0
    hits = sum(1 for doc_id in top_ids if doc_id in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@k for a ranked list."""
    if not relevant_ids or k <= 0:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def average_precision(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute average precision for a single query."""
    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute reciprocal rank for the first relevant hit."""
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute binary-gain nDCG@k for a single query."""
    if not relevant_ids or k <= 0:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(k, len(relevant_ids))
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def parse_cranfield_directory(
    data_dir: pathlib.Path,
) -> tuple[list[str], list[str], list[EvaluationQuery], dict[str, set[str]]]:
    """Load Cranfield-style documents, queries, and qrels from a directory."""
    doc_path = _resolve_candidate(
        data_dir,
        "cran.all.1400",
        "cran.all",
        "documents.txt",
        "docs.txt",
    )
    query_path = _resolve_candidate(
        data_dir,
        "cran.qry",
        "queries.txt",
        "cran.qry.txt",
    )
    qrel_path = _resolve_candidate(
        data_dir,
        "cranqrel",
        "qrels.txt",
        "cranqrel.txt",
    )

    doc_records = _parse_cran_records(doc_path.read_text(encoding="utf-8", errors="ignore"))
    query_records = _parse_cran_records(query_path.read_text(encoding="utf-8", errors="ignore"))
    qrels = _parse_qrels(qrel_path.read_text(encoding="utf-8", errors="ignore"))

    texts = [record["text"] for record in doc_records]
    ids = [record["id"] for record in doc_records]
    queries = [
        EvaluationQuery(query_id=_normalize_identifier(record["id"]), text=record["text"])
        for record in query_records
    ]
    return texts, ids, queries, qrels


def parse_beir_directory(
    data_dir: pathlib.Path,
    split: str | None = None,
) -> tuple[list[str], list[str], list[EvaluationQuery], dict[str, set[str]]]:
    """Load a BEIR-style dataset directory such as SciFact."""
    corpus_path = _resolve_candidate(data_dir, "corpus.jsonl")
    query_path = _resolve_candidate(data_dir, "queries.jsonl")
    qrel_path = _resolve_beir_qrels_path(data_dir, split)

    texts: list[str] = []
    ids: list[str] = []
    for record in _read_jsonl(corpus_path):
        doc_id = _normalize_identifier(str(record.get("_id", "")))
        title = str(record.get("title", "")).strip()
        text = str(record.get("text", "")).strip()
        combined_text = " ".join(part for part in (title, text) if part).strip()
        if doc_id and combined_text:
            ids.append(doc_id)
            texts.append(combined_text)

    qrels = _parse_beir_qrels(qrel_path)
    queries = [
        EvaluationQuery(
            query_id=_normalize_identifier(str(record.get("_id", ""))),
            text=str(record.get("text", "")).strip(),
        )
        for record in _read_jsonl(query_path)
        if str(record.get("_id", "")).strip()
        and str(record.get("text", "")).strip()
        and _normalize_identifier(str(record.get("_id", ""))) in qrels
    ]
    return texts, ids, queries, qrels


def evaluate_runs(
    texts: list[str],
    ids: list[str],
    queries: list[EvaluationQuery],
    qrels: dict[str, set[str]],
    experiments: list[ExperimentSpec],
    eval_k: int = 10,
) -> list[dict[str, object]]:
    """Run multiple retrieval experiments and aggregate their metrics."""
    results: list[dict[str, object]] = []
    for experiment in experiments:
        config = SearchConfig(
            bm25_weight=experiment.bm25_weight,
            vector_weight=experiment.vector_weight,
            top_k=max(len(ids), eval_k),
        )
        engine = HybridSearch(config=config)
        engine.add_documents(texts=texts, ids=ids)

        query_rows: list[dict[str, object]] = []
        for query in queries:
            relevant_ids = qrels.get(query.query_id, set())
            ranked_ids = [result.doc_id for result in engine.search(query.text, top_k=len(ids))]
            query_rows.append(
                {
                    "query_id": query.query_id,
                    "precision_at_k": precision_at_k(ranked_ids, relevant_ids, eval_k),
                    "recall_at_k": recall_at_k(ranked_ids, relevant_ids, eval_k),
                    "average_precision": average_precision(ranked_ids, relevant_ids),
                    "reciprocal_rank": reciprocal_rank(ranked_ids, relevant_ids),
                    "ndcg_at_k": ndcg_at_k(ranked_ids, relevant_ids, eval_k),
                    "relevant_count": len(relevant_ids),
                    "retrieved_count": len(ranked_ids),
                    "top_results": ranked_ids[:eval_k],
                }
            )

        aggregate = {
            "experiment": experiment.name,
            "bm25_weight": experiment.bm25_weight,
            "vector_weight": experiment.vector_weight,
            "query_count": len(queries),
            f"precision@{eval_k}": _mean_metric(query_rows, "precision_at_k"),
            f"recall@{eval_k}": _mean_metric(query_rows, "recall_at_k"),
            "map": _mean_metric(query_rows, "average_precision"),
            "mrr": _mean_metric(query_rows, "reciprocal_rank"),
            f"ndcg@{eval_k}": _mean_metric(query_rows, "ndcg_at_k"),
            "per_query": query_rows,
        }
        results.append(aggregate)
    return results


def default_experiments() -> list[ExperimentSpec]:
    """Return the default comparison runs for HybridFind."""
    return [
        ExperimentSpec(name="bm25_only", bm25_weight=1.0, vector_weight=0.0),
        ExperimentSpec(name="tfidf_only", bm25_weight=0.0, vector_weight=1.0),
        ExperimentSpec(name="hybrid_rrf", bm25_weight=0.5, vector_weight=0.5),
    ]


def write_json_report(results: list[dict[str, object]], output_path: pathlib.Path) -> None:
    """Persist evaluation results as JSON."""
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv_report(results: list[dict[str, object]], output_path: pathlib.Path) -> None:
    """Persist summary metrics for each experiment as CSV."""
    if not results:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = [key for key in results[0] if key != "per_query"]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: value for key, value in row.items() if key != "per_query"})


def _mean_metric(rows: list[dict[str, object]], key: str) -> float:
    """Compute a safe arithmetic mean across query-level metric rows."""
    values = [float(row[key]) for row in rows]
    return mean(values) if values else 0.0


def _resolve_candidate(data_dir: pathlib.Path, *candidates: str) -> pathlib.Path:
    """Resolve the first existing benchmark file from a set of known names."""
    for candidate in candidates:
        path = data_dir / candidate
        if path.exists():
            return path
    expected = ", ".join(candidates)
    raise FileNotFoundError(f"Could not find any of the expected files in {data_dir}: {expected}")


def _resolve_beir_qrels_path(data_dir: pathlib.Path, split: str | None) -> pathlib.Path:
    """Resolve the qrels TSV for a BEIR dataset."""
    qrels_dir = data_dir / "qrels"
    if not qrels_dir.is_dir():
        raise FileNotFoundError(f"Could not find BEIR qrels directory: {qrels_dir}")

    candidates: list[pathlib.Path] = []
    if split:
        candidates.append(qrels_dir / f"{split}.tsv")
    candidates.extend(qrels_dir / name for name in ("test.tsv", "dev.tsv", "train.tsv"))

    for path in candidates:
        if path.exists():
            return path
    available = ", ".join(sorted(path.name for path in qrels_dir.glob("*.tsv"))) or "none"
    raise FileNotFoundError(f"Could not find a BEIR qrels split in {qrels_dir}. Available: {available}")


def _read_jsonl(path: pathlib.Path) -> list[dict[str, object]]:
    """Read newline-delimited JSON records from disk."""
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))
    return records


def _parse_cran_records(raw_text: str) -> list[dict[str, str]]:
    """Parse `.I`/field-marker records used by Cranfield docs and queries."""
    records: list[dict[str, str]] = []
    current_id: str | None = None
    current_lines: list[str] = []

    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(".I "):
            if current_id is not None:
                records.append({"id": current_id, "text": _normalize_record_text(current_lines)})
            current_id = stripped.split(maxsplit=1)[1]
            current_lines = []
            continue
        if stripped.startswith(".") and len(stripped) == 2 and stripped[1].isalpha():
            continue
        current_lines.append(line)

    if current_id is not None:
        records.append({"id": current_id, "text": _normalize_record_text(current_lines)})

    return records


def _normalize_record_text(lines: list[str]) -> str:
    """Normalize Cranfield record lines into a single search string."""
    return " ".join(line.strip() for line in lines if line.strip())


def _parse_qrels(raw_text: str) -> dict[str, set[str]]:
    """Parse Cranfield or TREC-style qrels into query -> relevant doc ids."""
    qrels: dict[str, set[str]] = {}
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) >= 4:
            query_id, doc_id, relevance = parts[0], parts[2], parts[3]
        elif len(parts) == 3:
            query_id, doc_id, relevance = parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            query_id, doc_id, relevance = parts[0], parts[1], "1"
        else:
            continue

        if _is_relevant(relevance):
            qrels.setdefault(_normalize_identifier(query_id), set()).add(_normalize_identifier(doc_id))
    return qrels


def _parse_beir_qrels(path: pathlib.Path) -> dict[str, set[str]]:
    """Parse BEIR qrels TSV into query -> relevant doc ids."""
    qrels: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_id = _normalize_identifier(str(row.get("query-id", "")))
            doc_id = _normalize_identifier(str(row.get("corpus-id", "")))
            score = str(row.get("score", "0"))
            if query_id and doc_id and _is_relevant(score):
                qrels.setdefault(query_id, set()).add(doc_id)
    return qrels


def _normalize_identifier(raw_value: str) -> str:
    """Normalize benchmark identifiers so qrels and parsed records align."""
    value = raw_value.strip()
    if value.isdigit():
        return str(int(value))
    return value


def _is_relevant(raw_relevance: str) -> bool:
    """Interpret qrels relevance labels conservatively."""
    try:
        return float(raw_relevance) > 0
    except ValueError:
        return True
