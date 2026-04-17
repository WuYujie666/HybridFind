"""CLI entry-point: hybridfind index docs/ | hybridfind search 'query'."""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from hybridfind.config import SearchConfig
from hybridfind.engine import HybridSearch
from hybridfind.evaluation import (
    ExperimentSpec,
    default_experiments,
    evaluate_runs,
    parse_beir_directory,
    write_csv_report,
    write_json_report,
)

app = typer.Typer(help='HybridFind - hybrid semantic + keyword search CLI')
console = Console()

INDEX_PATH = pathlib.Path('artifacts/cache/hybridfind_index.json')


def _configure_utf8_stdio() -> None:
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')


def _load_index_snapshot() -> dict[str, object]:
    if not INDEX_PATH.exists():
        console.print('[red]No index found. Run `hybridfind index <dir>` first.[/red]')
        raise typer.Exit(1)
    return json.loads(INDEX_PATH.read_text(encoding='utf-8'))


def _load_engine() -> HybridSearch:
    data = _load_index_snapshot()
    engine = HybridSearch(SearchConfig(**data.get('config', {})))
    engine.add_documents(texts=data['texts'], ids=data['ids'], metadatas=data.get('metadatas', []))
    return engine


@app.command()
def index(
    directory: str = typer.Argument(..., help='Directory containing text files to index'),
    extensions: str = typer.Option('.txt,.md,.rst', help='Comma-separated file extensions'),
) -> None:
    dir_path = pathlib.Path(directory)
    if not dir_path.is_dir():
        console.print(f'[red]Directory not found: {directory}[/red]')
        raise typer.Exit(1)

    console.print('[cyan]Loading documents...[/cyan]')
    exts = {e.strip() for e in extensions.split(',')}
    texts: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []
    for fpath in sorted(dir_path.rglob('*')):
        if fpath.is_file() and fpath.suffix in exts:
            content = fpath.read_text(encoding='utf-8', errors='ignore')
            if content.strip():
                texts.append(content)
                ids.append(str(fpath))
                metadatas.append({'filename': fpath.name, 'path': str(fpath)})

    if not texts:
        console.print('[yellow]No matching files found.[/yellow]')
        raise typer.Exit(1)

    config = SearchConfig()
    engine = HybridSearch(config=config)
    engine.add_documents(texts=texts, ids=ids, metadatas=metadatas)

    console.print('[cyan]Encoding dense documents...[/cyan]')
    engine.build_indices()
    console.print('[cyan]Saving dense cache...[/cyan]')
    engine.save_dense_cache()

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(
        json.dumps({'texts': texts, 'ids': ids, 'metadatas': metadatas, 'config': config.model_dump()}, ensure_ascii=False),
        encoding='utf-8',
    )
    console.print(f'[green]Indexed {len(texts)} documents from {directory}[/green]')


@app.command()
def search(
    query: str = typer.Argument(..., help='Search query'),
    top_k: int = typer.Option(5, '--top-k', '-k', help='Number of results'),
    bm25_weight: float = typer.Option(0.5, '--bm25-weight', help='BM25 weight'),
    dense_weight: float = typer.Option(0.5, '--dense-weight', help='Dense retrieval weight'),
    filter_field: Optional[str] = typer.Option(None, '--filter-field', help='Metadata field to filter'),
    filter_value: Optional[str] = typer.Option(None, '--filter-value', help='Metadata value to match'),
) -> None:
    engine = _load_engine()
    engine.config.bm25_weight = bm25_weight
    engine.config.dense_weight = dense_weight

    if dense_weight > 0:
        console.print('[cyan]Loading dense cache...[/cyan]')
        if engine.load_dense_cache():
            console.print('[green]Dense cache loaded.[/green]')
        else:
            console.print('[yellow]Dense cache missing or stale. Rebuilding...[/yellow]')
            engine.ensure_dense_ready()

    meta_filter = {filter_field: filter_value} if filter_field and filter_value else None
    results = engine.search(query, top_k=top_k, metadata_filter=meta_filter)
    if not results:
        console.print('[yellow]No results found.[/yellow]')
        return

    table = Table(title=f'Search results for: {query}')
    table.add_column('Rank', style='bold cyan', width=5)
    table.add_column('Doc ID', style='green')
    table.add_column('Score', style='magenta', width=10)
    table.add_column('Preview', max_width=60)
    for rank, result in enumerate(results, 1):
        preview = result.text[:120].replace('\n', ' ')
        table.add_row(str(rank), result.doc_id, f'{result.score:.6f}', preview)
    console.print(table)


@app.command()
def evaluate(
    data_dir: str = typer.Option(..., '--data-dir', help='Directory containing benchmark files'),
    dataset: str = typer.Option('scifact', '--dataset', help='Benchmark dataset loader to use'),
    eval_k: int = typer.Option(10, '--eval-k', help='Cutoff for Precision/Recall/nDCG metrics'),
    output_json: Optional[str] = typer.Option(None, '--output-json', help='Optional JSON report path'),
    output_csv: Optional[str] = typer.Option(None, '--output-csv', help='Optional CSV report path'),
    weight_pair: Optional[list[str]] = typer.Option(None, '--weight-pair', help="Additional experiment as 'name:bm25,dense' or 'bm25,dense'"),
    beir_split: Optional[str] = typer.Option(None, '--beir-split', help='Optional BEIR qrels split to load'),
) -> None:
    benchmark_dir = pathlib.Path(data_dir)
    if not benchmark_dir.is_dir():
        console.print(f'[red]Data directory not found: {data_dir}[/red]')
        raise typer.Exit(1)

    dataset_name = dataset.lower()
    try:
        if dataset_name in {'scifact', 'beir', 'beir-scifact', 'beir_scifact'}:
            texts, ids, queries, qrels = parse_beir_directory(benchmark_dir, split=beir_split)
        else:
            console.print(f"[red]Unsupported dataset: {dataset}. This project now supports only BEIR-style datasets such as 'scifact'.[/red]")
            raise typer.Exit(1)
    except FileNotFoundError as exc:
        console.print(f'[red]{exc}[/red]')
        raise typer.Exit(1) from exc

    console.print(f'[cyan]Loaded {len(ids)} documents and {len(queries)} queries.[/cyan]')
    experiments = default_experiments() + _parse_weight_pairs(weight_pair or [])
    results = evaluate_runs(texts, ids, queries, qrels, experiments, eval_k=eval_k)

    _print_evaluation_table(results, eval_k)
    if output_json:
        json_path = pathlib.Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_report(results, json_path)
        console.print(f'[green]Wrote JSON report to {json_path}[/green]')
    if output_csv:
        csv_path = pathlib.Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv_report(results, csv_path)
        console.print(f'[green]Wrote CSV report to {csv_path}[/green]')


def _parse_weight_pairs(weight_pairs: list[str]) -> list[ExperimentSpec]:
    experiments: list[ExperimentSpec] = []
    for index, raw_value in enumerate(weight_pairs, start=1):
        payload = raw_value
        if ':' in raw_value:
            name, payload = raw_value.split(':', maxsplit=1)
        else:
            name = f'custom_{index}'
        parts = [part.strip() for part in payload.split(',')]
        if len(parts) != 2:
            raise typer.BadParameter("Weight pairs must look like 'name:0.7,0.3' or '0.7,0.3'.", param_hint='--weight-pair')
        try:
            bm25_weight = float(parts[0])
            dense_weight = float(parts[1])
        except ValueError as exc:
            raise typer.BadParameter('Weight pairs must use numeric values.', param_hint='--weight-pair') from exc
        experiments.append(ExperimentSpec(name=name.strip() or f'custom_{index}', bm25_weight=bm25_weight, dense_weight=dense_weight))
    return experiments


def _print_evaluation_table(results: list[dict[str, object]], eval_k: int) -> None:
    table = Table(title='Offline Evaluation Summary')
    table.add_column('Experiment', style='bold cyan')
    table.add_column('BM25', style='green', justify='right')
    table.add_column('Dense', style='green', justify='right')
    table.add_column(f'P@{eval_k}', style='magenta', justify='right')
    table.add_column(f'R@{eval_k}', style='magenta', justify='right')
    table.add_column('MAP', style='yellow', justify='right')
    table.add_column('MRR', style='yellow', justify='right')
    table.add_column(f'nDCG@{eval_k}', style='yellow', justify='right')
    for row in results:
        table.add_row(
            str(row['experiment']),
            f"{float(row['bm25_weight']):.2f}",
            f"{float(row['dense_weight']):.2f}",
            f"{float(row[f'precision@{eval_k}']):.4f}",
            f"{float(row[f'recall@{eval_k}']):.4f}",
            f"{float(row['map']):.4f}",
            f"{float(row['mrr']):.4f}",
            f"{float(row[f'ndcg@{eval_k}']):.4f}",
        )
    console.print(table)


def main() -> None:
    _configure_utf8_stdio()
    app()


if __name__ == '__main__':
    main()
