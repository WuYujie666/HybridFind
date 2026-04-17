"""Tests for offline evaluation support."""

from __future__ import annotations

import json

import pytest

from hybridfind.evaluation import (
    average_precision,
    default_experiments,
    evaluate_runs,
    ndcg_at_k,
    parse_beir_directory,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    write_csv_report,
    write_json_report,
)

BEIR_CORPUS = """{"_id": "d1", "title": "COVID vaccines", "text": "Vaccines reduce severe outcomes."}
{"_id": "d2", "title": "Climate paper", "text": "Global warming affects sea levels."}
{"_id": "d3", "title": "Space article", "text": "Galaxies and telescopes are studied in astronomy."}
"""

BEIR_QUERIES = """{"_id": "q1", "text": "covid vaccine outcomes"}
{"_id": "q2", "text": "sea level climate"}
"""

BEIR_QRELS = """query-id	corpus-id	score
q1	d1	1
q2	d2	2
q2	d3	0
"""


def _write_beir_fixture(tmp_path) -> None:
    (tmp_path / 'corpus.jsonl').write_text(BEIR_CORPUS, encoding='utf-8')
    (tmp_path / 'queries.jsonl').write_text(BEIR_QUERIES, encoding='utf-8')
    qrels_dir = tmp_path / 'qrels'
    qrels_dir.mkdir()
    (qrels_dir / 'test.tsv').write_text(BEIR_QRELS, encoding='utf-8')


class TestEvaluationMetrics:
    def test_metric_helpers(self) -> None:
        ranked_ids = ['d1', 'd2', 'd3']
        relevant_ids = {'d2', 'd3'}
        assert precision_at_k(ranked_ids, relevant_ids, 2) == 0.5
        assert recall_at_k(ranked_ids, relevant_ids, 2) == 0.5
        assert average_precision(ranked_ids, relevant_ids) == pytest.approx(7 / 12)
        assert reciprocal_rank(ranked_ids, relevant_ids) == 0.5
        assert ndcg_at_k(ranked_ids, relevant_ids, 3) > 0.69


class TestBeirParsing:
    def test_parse_beir_directory(self, tmp_path) -> None:
        _write_beir_fixture(tmp_path)
        texts, ids, queries, qrels = parse_beir_directory(tmp_path)
        assert ids == ['d1', 'd2', 'd3']
        assert texts[0].startswith('COVID vaccines')
        assert [query.query_id for query in queries] == ['q1', 'q2']
        assert qrels == {'q1': {'d1'}, 'q2': {'d2'}}


class TestEvaluationRuns:
    def test_evaluate_runs_returns_default_experiments(self, tmp_path) -> None:
        _write_beir_fixture(tmp_path)
        texts, ids, queries, qrels = parse_beir_directory(tmp_path)
        results = evaluate_runs(texts, ids, queries, qrels, default_experiments(), eval_k=2)
        assert [row['experiment'] for row in results] == ['bm25_only', 'dense_only', 'hybrid_rrf']
        assert all(row['query_count'] == 2 for row in results)

    def test_report_writers(self, tmp_path) -> None:
        _write_beir_fixture(tmp_path)
        texts, ids, queries, qrels = parse_beir_directory(tmp_path)
        results = evaluate_runs(texts, ids, queries, qrels, default_experiments(), eval_k=2)
        json_path = tmp_path / 'report.json'
        csv_path = tmp_path / 'report.csv'
        write_json_report(results, json_path)
        write_csv_report(results, csv_path)
        data = json.loads(json_path.read_text(encoding='utf-8'))
        assert len(data) == 3
        assert data[0]['experiment'] == 'bm25_only'
        assert 'experiment,bm25_weight,dense_weight,query_count,precision@2,recall@2,map,mrr,ndcg@2' in csv_path.read_text(encoding='utf-8')
