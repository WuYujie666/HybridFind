"""Offline evaluation utilities for BEIR-style datasets such as SciFact."""

from __future__ import annotations

import csv
import json
import math
import pathlib
import time
from dataclasses import dataclass, field
from statistics import mean

from hybridfind.config import SearchConfig
from hybridfind.engine import HybridSearch


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    text: str


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    bm25_weight: float
    dense_weight: float
    disable_faiss: bool = False  # bypass FAISS and use brute-force cosine (for benchmarking)


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_ids = ranked_ids[:k]
    if not top_ids:
        return 0.0
    hits = sum(1 for doc_id in top_ids if doc_id in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids or k <= 0:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def average_precision(ranked_ids: list[str], relevant_ids: set[str]) -> float:
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
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
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


def parse_beir_directory(data_dir: pathlib.Path, split: str | None = None) -> tuple[list[str], list[str], list[EvaluationQuery], dict[str, set[str]]]:
    corpus_path = _resolve_candidate(data_dir, 'corpus.jsonl')
    query_path = _resolve_candidate(data_dir, 'queries.jsonl')
    qrel_path = _resolve_beir_qrels_path(data_dir, split)

    texts: list[str] = []
    ids: list[str] = []
    for record in _read_jsonl(corpus_path):
        doc_id = _normalize_identifier(str(record.get('_id', '')))
        title = str(record.get('title', '')).strip()
        text = str(record.get('text', '')).strip()
        combined_text = ' '.join(part for part in (title, text) if part).strip()
        if doc_id and combined_text:
            ids.append(doc_id)
            texts.append(combined_text)

    qrels = _parse_beir_qrels(qrel_path)
    queries = [
        EvaluationQuery(query_id=_normalize_identifier(str(record.get('_id', ''))), text=str(record.get('text', '')).strip())
        for record in _read_jsonl(query_path)
        if str(record.get('_id', '')).strip()
        and str(record.get('text', '')).strip()
        and _normalize_identifier(str(record.get('_id', ''))) in qrels
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
    results: list[dict[str, object]] = []
    base_config = SearchConfig(
        top_k=max(len(ids), eval_k),
    )
    engine = HybridSearch(config=base_config)
    engine.add_documents(texts=texts, ids=ids)
    if any(experiment.dense_weight > 0 for experiment in experiments):
        print('[evaluate] Preparing dense cache...', flush=True)
        engine.ensure_dense_ready()
        print('[evaluate] Dense cache ready.', flush=True)

    for experiment in experiments:
        print(f'[evaluate] Running {experiment.name}...', flush=True)
        engine.config.bm25_weight = experiment.bm25_weight
        engine.config.dense_weight = experiment.dense_weight
        engine.config.top_k = max(len(ids), eval_k)

        # Temporarily disable FAISS index for brute-force comparison experiments
        saved_faiss = engine.dense._faiss_index
        if experiment.disable_faiss:
            engine.dense._faiss_index = None

        query_rows: list[dict[str, object]] = []
        total_queries = len(queries)
        experiment_start = time.perf_counter()
        for idx, query in enumerate(queries, start=1):
            if idx == 1 or idx % 50 == 0 or idx == total_queries:
                print(f'[evaluate] {experiment.name}: query {idx}/{total_queries}', flush=True)
            relevant_ids = qrels.get(query.query_id, set())
            t0 = time.perf_counter()
            ranked_ids = [result.doc_id for result in engine.search(query.text, top_k=len(ids))]
            query_time_ms = (time.perf_counter() - t0) * 1000
            query_rows.append({
                'query_id': query.query_id,
                'precision_at_k': precision_at_k(ranked_ids, relevant_ids, eval_k),
                'recall_at_k': recall_at_k(ranked_ids, relevant_ids, eval_k),
                'average_precision': average_precision(ranked_ids, relevant_ids),
                'reciprocal_rank': reciprocal_rank(ranked_ids, relevant_ids),
                'ndcg_at_k': ndcg_at_k(ranked_ids, relevant_ids, eval_k),
                'relevant_count': len(relevant_ids),
                'retrieved_count': len(ranked_ids),
                'top_results': ranked_ids[:eval_k],
                'query_time_ms': query_time_ms,
            })
        total_time_s = time.perf_counter() - experiment_start

        if experiment.disable_faiss:
            engine.dense._faiss_index = saved_faiss

        results.append({
            'experiment': experiment.name,
            'bm25_weight': experiment.bm25_weight,
            'dense_weight': experiment.dense_weight,
            'query_count': len(queries),
            f'precision@{eval_k}': _mean_metric(query_rows, 'precision_at_k'),
            f'recall@{eval_k}': _mean_metric(query_rows, 'recall_at_k'),
            'map': _mean_metric(query_rows, 'average_precision'),
            'mrr': _mean_metric(query_rows, 'reciprocal_rank'),
            f'ndcg@{eval_k}': _mean_metric(query_rows, 'ndcg_at_k'),
            'avg_query_ms': _mean_metric(query_rows, 'query_time_ms'),
            'total_time_s': total_time_s,
            'per_query': query_rows,
        })
    return results


def _isolated_worker(
    data_dir_str: str,
    split: str | None,
    experiment: ExperimentSpec,
    eval_k: int,
) -> dict[str, object]:
    """Runs one experiment in a freshly spawned subprocess (cold CPU cache)."""
    texts, ids, queries, qrels = parse_beir_directory(pathlib.Path(data_dir_str), split=split)
    return evaluate_runs(texts, ids, queries, qrels, [experiment], eval_k=eval_k)[0]


def evaluate_runs_isolated(
    data_dir: pathlib.Path,
    split: str | None,
    experiments: list[ExperimentSpec],
    eval_k: int = 10,
) -> list[dict[str, object]]:
    """Run each experiment in a separate subprocess so CPU cache is cold and equal for all."""
    import multiprocessing
    ctx = multiprocessing.get_context('spawn')
    results: list[dict[str, object]] = []
    for experiment in experiments:
        print(f'[evaluate] Spawning isolated process for {experiment.name}...', flush=True)
        with ctx.Pool(processes=1) as pool:
            result = pool.apply(_isolated_worker, (str(data_dir), split, experiment, eval_k))
        avg_ms = result.get('avg_query_ms', 0)
        print(f'[evaluate] {experiment.name} done: {avg_ms:.1f} ms/q', flush=True)
        results.append(result)
    return results


def default_experiments() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(name='bm25_only', bm25_weight=1.0, dense_weight=0.0),
        ExperimentSpec(name='dense_faiss', bm25_weight=0.0, dense_weight=1.0),
        ExperimentSpec(name='dense_brute', bm25_weight=0.0, dense_weight=1.0, disable_faiss=True),
        ExperimentSpec(name='hybrid_rrf', bm25_weight=0.5, dense_weight=0.5),
    ]


def write_json_report(results: list[dict[str, object]], output_path: pathlib.Path) -> None:
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv_report(results: list[dict[str, object]], output_path: pathlib.Path) -> None:
    if not results:
        output_path.write_text('', encoding='utf-8')
        return
    fieldnames = [key for key in results[0] if key != 'per_query']
    with output_path.open('w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: value for key, value in row.items() if key != 'per_query'})


def _mean_metric(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return mean(values) if values else 0.0


def _resolve_candidate(data_dir: pathlib.Path, *candidates: str) -> pathlib.Path:
    for candidate in candidates:
        path = data_dir / candidate
        if path.exists():
            return path
    expected = ', '.join(candidates)
    raise FileNotFoundError(f'Could not find any of the expected files in {data_dir}: {expected}')


def _resolve_beir_qrels_path(data_dir: pathlib.Path, split: str | None) -> pathlib.Path:
    qrels_dir = data_dir / 'qrels'
    if not qrels_dir.is_dir():
        raise FileNotFoundError(f'Could not find BEIR qrels directory: {qrels_dir}')
    candidates: list[pathlib.Path] = []
    if split:
        candidates.append(qrels_dir / f'{split}.tsv')
    candidates.extend(qrels_dir / name for name in ('test.tsv', 'dev.tsv', 'train.tsv'))
    for path in candidates:
        if path.exists():
            return path
    available = ', '.join(sorted(path.name for path in qrels_dir.glob('*.tsv'))) or 'none'
    raise FileNotFoundError(f'Could not find a BEIR qrels split in {qrels_dir}. Available: {available}')


def _read_jsonl(path: pathlib.Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))
    return records


def _parse_beir_qrels(path: pathlib.Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='	')
        for row in reader:
            query_id = _normalize_identifier(str(row.get('query-id', '')))
            doc_id = _normalize_identifier(str(row.get('corpus-id', '')))
            score = str(row.get('score', '0'))
            if query_id and doc_id and _is_relevant(score):
                qrels.setdefault(query_id, set()).add(doc_id)
    return qrels


def _normalize_identifier(raw_value: str) -> str:
    value = raw_value.strip()
    if value.isdigit():
        return str(int(value))
    return value


def _is_relevant(raw_relevance: str) -> bool:
    try:
        return float(raw_relevance) > 0
    except ValueError:
        return True
