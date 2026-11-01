"""Reciprocal Rank Fusion (Cormack et al., SIGIR 2009).

RRF merges ranked lists from multiple retrievers by summing
``1 / (k + rank_i)`` for each document across the input lists. The clever
property is that it operates on **ranks** rather than raw scores, so we
can fuse BM25 scores and cosine similarities without normalising them onto
a common scale.

A document missing from one list simply contributes nothing for that list,
which means RRF can only *add* documents to the fused result -- it never
loses anything that the base retrievers found.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from config import RRF_K, TOP_K_STAGE1


def rrf_fuse(
    runs: List[pd.DataFrame],
    k: int = RRF_K,
    top_k: int = TOP_K_STAGE1,
) -> pd.DataFrame:
    """Fuse several PyTerrier-style runs into one ranked list per query.

    Parameters
    ----------
    runs
        List of DataFrames, each with columns ``[qid, docno, score, rank]``.
        ``score`` is ignored; only ``rank`` is used.
    k
        RRF constant. The standard value 60 (Cormack et al., 2009) works well
        without tuning.
    top_k
        How many fused documents to keep per query.

    Returns
    -------
    pd.DataFrame
        Columns ``[qid, docno, score, rank]`` where ``score`` is the RRF score.
    """
    if not runs:
        return pd.DataFrame(columns=["qid", "docno", "score", "rank"])

    # Accumulator
    acc: Dict[str, Dict[str, float]] = {}

    for run in runs:
   
        for row in run.itertuples(index=False):
            qid = row.qid
            docno = row.docno
            rank = int(row.rank)
            acc.setdefault(qid, {}).setdefault(docno, 0.0)
            acc[qid][docno] += 1.0 / (k + rank + 1)
                          
            # rank is 0-indexed in PyTerrier; the original RRF paper uses
            # 1-indexed ranks, so we add 1 to align with the paper.

    # Compile as a single DataFrame
    rows = []
    for qid, doc_scores in acc.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        for new_rank, (docno, score) in enumerate(sorted_docs):
            rows.append({"qid": qid, "docno": docno, "score": score, "rank": new_rank})

    return pd.DataFrame(rows)


def rrf_fuse_single_query(
    ranked_lists: List[List[str]],
    k: int = RRF_K,
    top_k: int = TOP_K_STAGE1,
) -> List[tuple[str, float]]:
    """Convenience helper for the demo: fuse plain lists of docnos for one query.

    Parameters
    ----------
    ranked_lists
        Each inner list is an ordered list of docnos (best first).

    Returns
    -------
    List of ``(docno, rrf_score)`` tuples, sorted descending.
    """
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, docno in enumerate(lst):
            scores[docno] = scores.get(docno, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
