"""Cross-encoder reranker (Stage 2).

Takes the top-100 candidate documents from Stage 1 and rescores each
(query, document) pair through a BERT cross-encoder. The cross-encoder
reads both texts jointly, capturing fine-grained semantic interactions
that neither BM25 nor the bi-encoder can model.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (Nogueira & Cho, 2019 style,
distilled into MiniLM-L6). Pre-trained on MS MARCO, used zero-shot on FiQA.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from sentence_transformers import CrossEncoder

from config import CE_BATCH_SIZE, CE_MAX_LENGTH, CROSS_ENCODER_MODEL, TOP_K_FINAL


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = CROSS_ENCODER_MODEL,
        max_length: int = CE_MAX_LENGTH,
        batch_size: int = CE_BATCH_SIZE,
    ):
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidate_docnos: List[str],
        corpus: Dict[str, str],
        top_k: int = TOP_K_FINAL,
    ) -> List[Tuple[str, float]]:
        """Rerank candidates for a single query.

        Returns list of (docno, score) sorted descending.
        """
        pairs = []
        valid_docnos = []
        for docno in candidate_docnos:
            text = corpus.get(docno)
            if text:
                pairs.append((query, text))
                valid_docnos.append(docno)

        if not pairs:
            return []

        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        scored = sorted(
            zip(valid_docnos, scores), key=lambda x: x[1], reverse=True
        )
        return [(docno, float(s)) for docno, s in scored[:top_k]]

    def rerank_batch(
        self,
        queries: Dict[str, str],
        stage1_results: pd.DataFrame,
        corpus: Dict[str, str],
        top_k: int = TOP_K_FINAL,
    ) -> pd.DataFrame:
        """Rerank Stage 1 results for all queries.

        Parameters
        ----------
        queries : {qid: query_text}
        stage1_results : DataFrame [qid, docno, score, rank]
        corpus : {docno: text}
        top_k : how many to keep per query after reranking

        Returns
        -------
        DataFrame [qid, docno, score, rank]
        """
        grouped = stage1_results.groupby("qid")["docno"].apply(list).to_dict()

        rows = []
        total = len(grouped)
        for i, (qid, docnos) in enumerate(grouped.items(), 1):
            query_text = queries.get(qid)
            if not query_text:
                continue

            if i % 100 == 0 or i == total:
                print(f"      reranking {i}/{total} queries ...", flush=True)

            reranked = self.rerank(query_text, docnos, corpus, top_k=top_k)
            for rank, (docno, score) in enumerate(reranked):
                rows.append({"qid": qid, "docno": docno, "score": score, "rank": rank})

        return pd.DataFrame(rows)
