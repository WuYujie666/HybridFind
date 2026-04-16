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
    parse_cranfield_directory,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    write_csv_report,
    write_json_report,
)

CRAN_DOCS = """.I 1
.T
Aircraft propulsion systems
.W
Jet engine efficiency and aircraft thrust performance.
.I 2
.T
Information retrieval evaluation
.W
Precision recall and ranking metrics for search systems.
.I 3
.T
Wing design studies
.W
Aerodynamic lift and drag measurements for aircraft wings.
"""

CRAN_QUERIES = """.I 1
.W
aircraft engine
.I 2
.W
precision recall metrics
.I 3
.W
astronomy telescope
"""

CRAN_QRELS = """1 1
2 2
"""

PADDED_CRAN_QUERIES = """.I 001
.W
aircraft engine
.I 002
.W
precision recall metrics
.I 003
.W
astronomy telescope
"""

BEIR_CORPUS = """{"_id": "d1", "title": "COVID vaccines", "text": "Vaccines reduce severe outcomes."}
{"_id": "d2", "title": "Climate paper", "text": "Global warming affects sea levels."}
{"_id": "d3", "title": "Space article", "text": "Galaxies and telescopes are studied in astronomy."}
"""

BEIR_QUERIES = """{"_id": "q1", "text": "covid vaccine outcomes"}
{"_id": "q2", "text": "sea level climate"}
"""

BEIR_QRELS = """query-id\tcorpus-id\tscore
q1\td1\t1
q2\td2\t2
q2\td3\t0
"""


def _write_cranfield_fixture(tmp_path, queries_text: str = CRAN_QUERIES) -> None:
    (tmp_path / "cran.all.1400").write_text(CRAN_DOCS, encoding="utf-8")
    (tmp_path / "cran.qry").write_text(queries_text, encoding="utf-8")
    (tmp_path / "cranqrel").write_text(CRAN_QRELS, encoding="utf-8")


def _write_beir_fixture(tmp_path) -> None:
    (tmp_path / "corpus.jsonl").write_text(BEIR_CORPUS, encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text(BEIR_QUERIES, encoding="utf-8")
    qrels_dir = tmp_path / "qrels"
    qrels_dir.mkdir()
    (qrels_dir / "test.tsv").write_text(BEIR_QRELS, encoding="utf-8")


class TestEvaluationMetrics:
    def test_metric_helpers(self) -> None:
        ranked_ids = ["d1", "d2", "d3"]
        relevant_ids = {"d2", "d3"}

        assert precision_at_k(ranked_ids, relevant_ids, 2) == 0.5
        assert recall_at_k(ranked_ids, relevant_ids, 2) == 0.5
        assert average_precision(ranked_ids, relevant_ids) == pytest.approx(7 / 12)
        assert reciprocal_rank(ranked_ids, relevant_ids) == 0.5
        assert ndcg_at_k(ranked_ids, relevant_ids, 3) > 0.69


class TestCranfieldParsing:
    def test_parse_cranfield_directory(self, tmp_path) -> None:
        _write_cranfield_fixture(tmp_path)

        texts, ids, queries, qrels = parse_cranfield_directory(tmp_path)

        assert ids == ["1", "2", "3"]
        assert len(texts) == 3
        assert queries[0].query_id == "1"
        assert "aircraft engine" in queries[0].text
        assert qrels == {"1": {"1"}, "2": {"2"}}

    def test_parse_cranfield_directory_normalizes_padded_query_ids(self, tmp_path) -> None:
        _write_cranfield_fixture(tmp_path, queries_text=PADDED_CRAN_QUERIES)

        _, _, queries, qrels = parse_cranfield_directory(tmp_path)

        assert [query.query_id for query in queries] == ["1", "2", "3"]
        assert qrels == {"1": {"1"}, "2": {"2"}}


class TestBeirParsing:
    def test_parse_beir_directory(self, tmp_path) -> None:
        _write_beir_fixture(tmp_path)

        texts, ids, queries, qrels = parse_beir_directory(tmp_path)

        assert ids == ["d1", "d2", "d3"]
        assert texts[0].startswith("COVID vaccines")
        assert [query.query_id for query in queries] == ["q1", "q2"]
        assert qrels == {"q1": {"d1"}, "q2": {"d2"}}

    def test_parse_beir_directory_filters_queries_to_selected_split(self, tmp_path) -> None:
        _write_beir_fixture(tmp_path)
        queries_path = tmp_path / "queries.jsonl"
        extra_query = '{"_id": "unused", "text": "this query is not in qrels"}\n'
        queries_path.write_text(BEIR_QUERIES + extra_query, encoding="utf-8")

        _, _, queries, qrels = parse_beir_directory(tmp_path)

        assert [query.query_id for query in queries] == ["q1", "q2"]
        assert "unused" not in qrels


class TestEvaluationRuns:
    def test_evaluate_runs_returns_default_experiments(self, tmp_path) -> None:
        _write_cranfield_fixture(tmp_path)
        texts, ids, queries, qrels = parse_cranfield_directory(tmp_path)

        results = evaluate_runs(texts, ids, queries, qrels, default_experiments(), eval_k=2)

        assert [row["experiment"] for row in results] == ["bm25_only", "tfidf_only", "hybrid_rrf"]
        assert all(row["query_count"] == 3 for row in results)
        assert all("precision@2" in row for row in results)
        assert all("ndcg@2" in row for row in results)
        assert all(len(row["per_query"]) == 3 for row in results)

        hybrid_row = next(row for row in results if row["experiment"] == "hybrid_rrf")
        first_query = next(row for row in hybrid_row["per_query"] if row["query_id"] == "1")
        third_query = next(row for row in hybrid_row["per_query"] if row["query_id"] == "3")

        assert first_query["top_results"][0] == "1"
        assert third_query["average_precision"] == 0.0
        assert third_query["reciprocal_rank"] == 0.0

    def test_report_writers(self, tmp_path) -> None:
        _write_cranfield_fixture(tmp_path)
        texts, ids, queries, qrels = parse_cranfield_directory(tmp_path)
        results = evaluate_runs(texts, ids, queries, qrels, default_experiments(), eval_k=2)

        json_path = tmp_path / "report.json"
        csv_path = tmp_path / "report.csv"
        write_json_report(results, json_path)
        write_csv_report(results, csv_path)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(data) == 3
        assert data[0]["experiment"] == "bm25_only"
        assert csv_path.exists()
        assert "experiment,bm25_weight,vector_weight,query_count,precision@2,recall@2,map,mrr,ndcg@2" in csv_path.read_text(
            encoding="utf-8"
        )
