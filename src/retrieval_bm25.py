"""BM25 first-stage retrieval via PyTerrier.

Wraps Terrier's BM25 implementation behind a small class so the rest of the
pipeline can call ``.retrieve()`` / ``.retrieve_batch()`` 
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyterrier as pt

from config import BM25_B, BM25_INDEX_DIR, BM25_K1, TOP_K_STAGE1
from src.indexing import load_bm25_index


class BM25Retriever:
    """Lexical first-stage retriever based on BM25 (Robertson et al., 1994)."""

    def __init__(
        self,
        index_path: Path = BM25_INDEX_DIR,
        k1: float = BM25_K1,
        b: float = BM25_B,
        num_results: int = TOP_K_STAGE1,
    ):
        self.index = load_bm25_index(index_path)
        self.k1 = k1
        self.b = b
        self.num_results = num_results

        # Terrier exposes BM25 parameters via "controls".
        self._retriever = pt.terrier.Retriever(
            self.index,
            wmodel="BM25",
            controls={"bm25.k_1": str(k1), "bm25.b": str(b)},
            num_results=num_results,
        )

    #  Single query 
    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict]:
        """Retrieve top-k documents for a single free-text query.

        Returns a list of dicts: ``{"docno", "score", "rank"}``.
        """
        df = self._retriever.search(query)
        if top_k is not None:
            df = df.head(top_k)
        return [
            {"docno": rec["docno"], "score": float(rec["score"]), "rank": int(rec["rank"])}
            for rec in df.to_dict("records")
        ]

    # Batch retrieval 
    def retrieve_batch(self, queries: Dict[str, str]) -> pd.DataFrame:
        """Retrieve for many queries at once.

        Returns
        -------
        pd.DataFrame with columns ``[qid, docno, score, rank]`` -- the standard
        PyTerrier result format that ``pytrec_eval`` can use.
        """
        topics = pd.DataFrame(
            [{"qid": qid, "query": _sanitise(text)} for qid, text in queries.items()]
        )
        results = self._retriever.transform(topics)
        return results[["qid", "docno", "score", "rank"]].reset_index(drop=True)


def _sanitise(query: str) -> str:
    """Delete characters that confuse Terrier's query parser.

    Because Perrier treats ?,!,: etc. as operators. 
    """
    return "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in query).strip()
