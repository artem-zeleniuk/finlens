"""BM25 index construction via PyTerrier.


This is the standard lexical preprocessing for BM25 and is what every BEIR
baseline uses. We do **not** apply it manually  Terrier does it during
indexing, automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import pyterrier as pt

from config import BM25_INDEX_DIR
from src.data_loader import iter_local_corpus


def _doc_iterator() -> Iterator[dict]:
    for record in iter_local_corpus():
        yield {"docno": record["docno"], "text": record["text"]}


def build_bm25_index(
    index_path: Path = BM25_INDEX_DIR,
    overwrite: bool = False,
) -> str:
 
    index_path = Path(index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    if (index_path / "data.properties").exists() and not overwrite:
        print(f"[indexing] BM25 index already exists at {index_path} — skipping.")
        return str(index_path)

    print(f"[indexing] Building BM25 index at {index_path} ...")

    indexer = pt.IterDictIndexer(
        str(index_path),
        meta={"docno": 32},   
        overwrite=overwrite,
    )
    index_ref = indexer.index(_doc_iterator())

    print(f"[indexing] Done. Index reference: {index_ref}")
    return str(index_path)


def load_bm25_index(index_path: Path = BM25_INDEX_DIR):
    """Load an existing BM25 index. Returns the underlying Terrier Index object."""
    index_path = Path(index_path)
    if not (index_path / "data.properties").exists():
        raise FileNotFoundError(
            f"No BM25 index found at {index_path}. Run build_indexes.py first."
        )
    return pt.IndexFactory.of(str(index_path))
