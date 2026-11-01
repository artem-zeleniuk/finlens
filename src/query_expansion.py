"""Query expansion via pseudo-relevance feedback (Pipeline B, Stage 1).

Uses Bo1 (Bose-Einstein 1) to score candidate expansion terms by their
divergence from a random distribution. Three-step process:
  1. BM25 first pass -> top-N feedback documents
  2. Score terms in feedback docs via Bo1, pick top-M expansion terms
  3. BM25 second pass with expanded query, returns top-100 candidates
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import pyterrier as pt

from config import (
    BM25_B,
    BM25_INDEX_DIR,
    BM25_K1,
    BO1_FB_DOCS,
    BO1_FB_TERMS,
    TOP_K_STAGE1,
)
from src.indexing import load_bm25_index
from src.retrieval_bm25 import _sanitise


class QueryExpander:
    """PRF-based query expansion pipeline."""

    def __init__(
        self,
        index_path: Path = BM25_INDEX_DIR,
        fb_docs: int = BO1_FB_DOCS,
        fb_terms: int = BO1_FB_TERMS,
        num_results: int = TOP_K_STAGE1,
    ):
        self.index = load_bm25_index(index_path)
        self.fb_docs = fb_docs
        self.fb_terms = fb_terms

        # First pass: retrieve feedback documents
        bm25_fb = pt.terrier.Retriever(
            self.index,
            wmodel="BM25",
            controls={"bm25.k_1": str(BM25_K1), "bm25.b": str(BM25_B)},
            num_results=fb_docs,
        )

        qe = pt.rewrite.Bo1QueryExpansion(self.index, fb_terms=fb_terms, fb_docs=fb_docs)

        # Second pass: retrieve with expanded query
        bm25_full = pt.terrier.Retriever(
            self.index,
            wmodel="BM25",
            controls={"bm25.k_1": str(BM25_K1), "bm25.b": str(BM25_B)},
            num_results=num_results,
        )

        self._pipeline = bm25_fb >> qe >> bm25_full

    def retrieve_batch(self, queries: dict[str, str]) -> pd.DataFrame:
        """Run the full expansion pipeline for all queries."""
        topics = pd.DataFrame(
            [{"qid": qid, "query": _sanitise(text)} for qid, text in queries.items()]
        )
        results = self._pipeline.transform(topics)
        return results[["qid", "docno", "score", "rank"]].reset_index(drop=True)

    def expand_query(self, query: str) -> str:
        """Return the expanded query string (useful for demo/debugging)."""
        topics = pd.DataFrame([{"qid": "0", "query": _sanitise(query)}])

        bm25_fb = pt.terrier.Retriever(
            self.index,
            wmodel="BM25",
            controls={"bm25.k_1": str(BM25_K1), "bm25.b": str(BM25_B)},
            num_results=self.fb_docs,
        )
        qe = pt.rewrite.Bo1QueryExpansion(self.index, fb_terms=self.fb_terms, fb_docs=self.fb_docs)
        expanded = (bm25_fb >> qe).transform(topics)
        return expanded.iloc[0]["query"] if len(expanded) > 0 else query


class DenseFeedbackExpander:
    """Query expansion using dense retriever for feedback, Bo1 for term extraction.

    Hypothesis: standard PRF suffers when BM25 feedback docs are poor due to
    vocabulary mismatch. Using dense retriever for feedback selection should
    provide higher-quality documents for Bo1 term extraction.
    """

    def __init__(
        self,
        dense_retriever,
        index_path: Path = BM25_INDEX_DIR,
        fb_docs: int = BO1_FB_DOCS,
        fb_terms: int = BO1_FB_TERMS,
        num_results: int = TOP_K_STAGE1,
    ):
        self.dense = dense_retriever
        self.index = load_bm25_index(index_path)
        self.fb_docs = fb_docs
        self.fb_terms = fb_terms

        self._qe = pt.rewrite.Bo1QueryExpansion(
            self.index, fb_terms=fb_terms, fb_docs=fb_docs
        )
        self._bm25 = pt.terrier.Retriever(
            self.index,
            wmodel="BM25",
            controls={"bm25.k_1": str(BM25_K1), "bm25.b": str(BM25_B)},
            num_results=num_results,
        )

    def retrieve_batch(self, queries: Dict[str, str]) -> pd.DataFrame:
        """Dense feedback -> Bo1 expansion -> BM25 2nd pass."""
        # Step 1: Dense retriever finds feedback docs
        dense_fb = self.dense.retrieve_batch(queries, top_k=self.fb_docs)
        # Add the query text (Bo1 needs it)
        dense_fb["query"] = dense_fb["qid"].map(
            lambda qid: _sanitise(queries[qid])
        )

        # Step 2: Bo1 extracts terms from those docs (using BM25 index)
        expanded = self._qe.transform(dense_fb)

        # Step 3: BM25 2nd pass with expanded query
        results = self._bm25.transform(expanded)
        return results[["qid", "docno", "score", "rank"]].reset_index(drop=True)

    def expand_query(self, query: str) -> str:
        """Show the expanded query (for demo)."""
        dense_fb = self.dense.retrieve_batch({"0": query}, top_k=self.fb_docs)
        dense_fb["query"] = _sanitise(query)
        expanded = self._qe.transform(dense_fb)
        return expanded.iloc[0]["query"] if len(expanded) > 0 else query
