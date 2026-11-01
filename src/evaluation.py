"""IR evaluation utilities built on ``pytrec_eval``.


* 1) ``Recall@100``. This is the metric the first stage exists to
  optimise: a relevant document missed at Stage 1 is unrecoverable later.
* 2)``nDCG@10``, ``MAP``, ``MRR@10``

The functions take PyTerrier-style result DataFrames (``[qid, docno, score,
rank]``) and the same ``qrels`` dict produced by ``data_loader.load_local_qrels``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import pandas as pd
import pytrec_eval

from config import METRICS_END2END, METRICS_STAGE1


def _results_df_to_run(results: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Convert a PyTerrier result DataFrame into the dict format pytrec_eval expects."""
    run: Dict[str, Dict[str, float]] = {}
    for row in results.itertuples(index=False):
        qid = str(row.qid)
        docno = str(row.docno)
        score = float(row.score)
        run.setdefault(qid, {})[docno] = score
    return run


def _normalise_qrels(qrels: Mapping[str, Mapping[str, int]]) -> Dict[str, Dict[str, int]]:
    """Make sure qrel keys are strings (pytrec_eval is strict about types)."""
    return {str(qid): {str(d): int(rel) for d, rel in docs.items()} for qid, docs in qrels.items()}


def evaluate(
    results: pd.DataFrame,
    qrels: Mapping[str, Mapping[str, int]],
    metrics: Iterable[str],
) -> Dict[str, float]:
    """Compute mean values of the requested metrics over all queries.
    Returns
    -------
    dict
        ``{metric_name: mean_value_across_queries}``.
    """
    metrics = set(metrics)
    norm_qrels = _normalise_qrels(qrels)
    run = _results_df_to_run(results)

    evaluator = pytrec_eval.RelevanceEvaluator(norm_qrels, metrics)
    per_query = evaluator.evaluate(run)

    if not per_query:
        return {m: 0.0 for m in metrics}

    averaged: Dict[str, float] = {}
    for metric in metrics:
        values = [q.get(metric, 0.0) for q in per_query.values()]
        averaged[metric] = sum(values) / len(values)
    return averaged


def evaluate_stage1(
    results: pd.DataFrame,
    qrels: Mapping[str, Mapping[str, int]],
) -> Dict[str, float]:
    """Stage-1 metrics: how good is the candidate set?"""
    return evaluate(results, qrels, METRICS_STAGE1)


def evaluate_end_to_end(
    results: pd.DataFrame,
    qrels: Mapping[str, Mapping[str, int]],
) -> Dict[str, float]:
    """End-to-end ranking metrics."""
    return evaluate(results, qrels, METRICS_END2END)


# Pretty-printing 
_METRIC_LABELS = {
    "recall_100": "Recall@100",
    "ndcg_cut_10": "nDCG@10",
    "map": "MAP",
    "recip_rank": "MRR@10",
}


def format_results_table(rows: Dict[str, Dict[str, float]]) -> str:
    
    if not rows:
        return "(no results)"

    seen_metrics = set()
    for r in rows.values():
        seen_metrics.update(r.keys())
    ordered = [m for m in _METRIC_LABELS if m in seen_metrics]
    ordered += [m for m in seen_metrics if m not in _METRIC_LABELS]

    headers = ["Config"] + [_METRIC_LABELS.get(m, m) for m in ordered]
    name_w = max(len("Config"), max(len(name) for name in rows))
    metric_w = 11

    sep = "-" * (name_w + 2 + metric_w * len(ordered))
    out = []
    out.append(f"{headers[0]:<{name_w}}  " + "  ".join(f"{h:>{metric_w - 2}}" for h in headers[1:]))
    out.append(sep)
    for name, metric_values in rows.items():
        cells = [f"{name:<{name_w}}"]
        for m in ordered:
            v = metric_values.get(m)
            cells.append(f"{v:>{metric_w - 2}.4f}" if v is not None else " " * (metric_w - 2))
        out.append("  ".join(cells))
    return "\n".join(out)
