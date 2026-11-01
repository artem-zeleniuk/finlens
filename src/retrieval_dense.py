"""Dense first-stage retrieval using a sentence-transformers bi-encoder.

The bi-encoder (``all-MiniLM-L6-v2``) encodes queries and documents into
a 384-dimensional vector space. Document embeddings are pre-computed once
and stored in a FAISS index; at query time we encode the query and run an
exact inner-product search over the index.

Why this design
---------------
* **Bi-encoder, not cross-encoder**, because we need millisecond-level
  retrieval over 57k documents, but cross-encoder would require one BERT
  forward pass per (query, document) pair.
* **FAISS Flat (exact) inner-product index**, because 57k vectors easily
  fit in memory and exact search avoids any recall loss from approximation.
* **Normalised embeddings**, so that inner product equals cosine similarity.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    DENSE_BATCH_SIZE,
    DENSE_MODEL,
    FAISS_INDEX_DIR,
    TOP_K_STAGE1,
)


class DenseRetriever:
    """Bi-encoder + FAISS dense retriever."""

    INDEX_FILENAME = "index.faiss"
    DOCIDS_FILENAME = "docids.pkl"

    def __init__(
        self,
        model_name: str = DENSE_MODEL,
        device: str | None = None,
    ):
        self.model_name = model_name
        # device=None lets sentence-transformers auto-pick CUDA when available.
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index: faiss.Index | None = None
        self.docids: List[str] = []

    # Index lifecycle 
    def build_index(
        self,
        corpus: Dict[str, str],
        index_dir: Path = FAISS_INDEX_DIR,
        batch_size: int = DENSE_BATCH_SIZE,
        show_progress_bar: bool = True,
    ) -> None:
        """Encode the entire corpus and build a FAISS exact inner-product index."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        
        self.docids = list(corpus.keys())
        texts = [corpus[d] for d in self.docids]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        ).astype(np.float32)

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

        faiss.write_index(self.index, str(index_dir / self.INDEX_FILENAME))
        with (index_dir / self.DOCIDS_FILENAME).open("wb") as f:
            pickle.dump(self.docids, f)

    def load_index(self, index_dir: Path = FAISS_INDEX_DIR) -> None:
        index_dir = Path(index_dir)
        idx_file = index_dir / self.INDEX_FILENAME
        ids_file = index_dir / self.DOCIDS_FILENAME
        if not idx_file.exists() or not ids_file.exists():
            raise FileNotFoundError(
                f"No FAISS index at {index_dir}. Run build_indexes.py first."
            )
        self.index = faiss.read_index(str(idx_file))
        with ids_file.open("rb") as f:
            self.docids = pickle.load(f)
        if self.index.ntotal != len(self.docids):
            raise RuntimeError(
                f"FAISS index size ({self.index.ntotal}) does not match "
                f"docids count ({len(self.docids)}). The index files are out of sync."
            )

    #  Single query 
    def retrieve(self, query: str, top_k: int = TOP_K_STAGE1) -> List[Dict]:
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load_index() or build_index().")

        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, idxs = self.index.search(q_emb, top_k)
        results: List[Dict] = []
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0])):
            if idx < 0:
                continue
            results.append(
                {"docno": self.docids[idx], "score": float(score), "rank": rank}
            )
        return results

    # Batch retrieval 
    def retrieve_batch(
        self,
        queries: Dict[str, str],
        top_k: int = TOP_K_STAGE1,
        batch_size: int = DENSE_BATCH_SIZE,
    ) -> pd.DataFrame:
        """Retrieve for many queries.

        Encodes all query texts in one batch (cheap on GPU), then runs a single
        FAISS search for the whole matrix.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load_index() or build_index().")

        qids = list(queries.keys())
        qtexts = [queries[q] for q in qids]

        q_embs = self.model.encode(
            qtexts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        scores, idxs = self.index.search(q_embs, top_k)

        rows = []
        for qi, qid in enumerate(qids):
            for rank, (score, idx) in enumerate(zip(scores[qi], idxs[qi])):
                if idx < 0:
                    continue
                rows.append(
                    {
                        "qid": qid,
                        "docno": self.docids[idx],
                        "score": float(score),
                        "rank": rank,
                    }
                )
        return pd.DataFrame(rows)
