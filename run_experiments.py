"""Run all retrieval configurations on FiQA and print results.

Configurations:
  Stage 1 only:  BM25, Dense, Hybrid, BM25+Bo1, DenseBo1
  With reranker:  BM25+CE, Dense+CE, Hybrid+CE, BM25+Bo1+CE, DenseBo1+CE

Usage
-----
    python run_experiments.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

from config import RESULTS_DIR
from src.data_loader import load_all
from src.evaluation import (
    evaluate_end_to_end,
    evaluate_stage1,
    format_results_table,
)
from src.fusion import rrf_fuse
from src.query_expansion import DenseFeedbackExpander, QueryExpander
from src.reranker import CrossEncoderReranker
from src.retrieval_bm25 import BM25Retriever
from src.retrieval_dense import DenseRetriever


def _log(log_lines: list[str], msg: str) -> None:
    """Print to console and append to log buffer."""
    print(msg)
    log_lines.append(msg)


def _save_log(log_lines: list[str]) -> Path:
    """Write accumulated log to results/experiment_log_<timestamp>.txt."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"experiment_log_{ts}.txt"
    path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    log: list[str] = []

    _log(log, "=" * 70)
    _log(log, f"Full evaluation — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log(log, "=" * 70)

    #  Load data 
    corpus, queries, qrels = load_all()
    _log(log, f"\n{len(corpus):,} docs, {len(queries):,} queries")

    #  Initialise components 
    bm25 = BM25Retriever()
    dense = DenseRetriever()
    dense.load_index()
    qe_bo1 = QueryExpander()
    qe_dense = DenseFeedbackExpander(dense_retriever=dense)
    reranker = CrossEncoderReranker()

    results_table: dict[str, dict[str, float]] = {}

    #  Stage 1 retrieval 
    _log(log, "\n--- Stage 1 retrieval ---")

    t0 = time.perf_counter()
    bm25_results = bm25.retrieve_batch(queries)
    _log(log, f"  BM25: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    dense_results = dense.retrieve_batch(queries)
    _log(log, f"  Dense: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    hybrid_results = rrf_fuse([bm25_results, dense_results])
    _log(log, f"  Hybrid (RRF): {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    bo1_results = qe_bo1.retrieve_batch(queries)
    _log(log, f"  BM25+Bo1: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    dense_bo1_results = qe_dense.retrieve_batch(queries)
    _log(log, f"  DenseBo1: {time.perf_counter() - t0:.1f}s")

    dense_bo1_hybrid_results = rrf_fuse([dense_bo1_results, dense_results])

    # Evaluate all Stage 1
    stage1_configs = [
        ("BM25", bm25_results),
        ("Dense", dense_results),
        ("Hybrid", hybrid_results),
        ("BM25+Bo1", bo1_results),
        ("DenseBo1", dense_bo1_results),
        ("DenseBo1+Hyb", dense_bo1_hybrid_results),
    ]
    for name, df in stage1_configs:
        s1 = evaluate_stage1(df, qrels)
        e2e = evaluate_end_to_end(df, qrels)
        results_table[name] = {**s1, **e2e}

    # Log Stage 1 intermediate results
    _log(log, "\n--- Stage 1 metrics ---")
    _log(log, format_results_table(results_table))

    #  Stage 2 reranking 
    _log(log, "\n--- Stage 2 reranking ---")

    rerank_configs = [
        ("BM25+CE", bm25_results),
        ("Dense+CE", dense_results),
        ("Hybrid+CE", hybrid_results),
        ("BM25+Bo1+CE", bo1_results),
        ("DenseBo1+CE", dense_bo1_results),
        ("DenseBo1+Hyb+CE", dense_bo1_hybrid_results),
    ]
    for name, stage1_df in rerank_configs:
        t0 = time.perf_counter()
        reranked = reranker.rerank_batch(queries, stage1_df, corpus)
        _log(log, f"  {name}: {time.perf_counter() - t0:.1f}s")
        e2e = evaluate_end_to_end(reranked, qrels)
        results_table[name] = e2e

    #  Final results 
    _log(log, "\n" + "=" * 70)
    _log(log, "FINAL RESULTS")
    _log(log, "=" * 70)
    _log(log, format_results_table(results_table))
    _log(log, "=" * 70)

    # Save log
    log_path = _save_log(log)
    print(f"\nLog saved to {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
