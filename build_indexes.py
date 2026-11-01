"""Build the BM25 and FAISS dense indexes once and persist them to disk.

After this script finishes, every other module in the project can load both
indexes from ``indexes/`` instead of rebuilding them. The team only needs to
run this once after ``prepare_dataset.py``.

Usage
-----
    python build_indexes.py                # build whatever is missing
    python build_indexes.py --rebuild      # force a full rebuild

The two indexes are independent and can be built separately:

    python build_indexes.py --bm25-only
    python build_indexes.py --dense-only
"""

from __future__ import annotations

import argparse
import sys
import time

from config import BM25_INDEX_DIR, FAISS_INDEX_DIR
from src.data_loader import load_local_corpus
from src.indexing import build_bm25_index
from src.retrieval_dense import DenseRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a full rebuild of every requested index.",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Build only the BM25 index.",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Build only the FAISS dense index.",
    )
    args = parser.parse_args()

    do_bm25 = not args.dense_only
    do_dense = not args.bm25_only

    print("=" * 70)
    print("Building indexes")
    print("=" * 70)

    if do_bm25:
        print("\n[BM25]")
        t0 = time.perf_counter()
        build_bm25_index(BM25_INDEX_DIR, overwrite=args.rebuild)
        print(f"[BM25] Finished in {time.perf_counter() - t0:.1f}s")

    if do_dense:
        print("\n[Dense]")
        t0 = time.perf_counter()

        # Skip if FAISS index already there.
        idx_file = FAISS_INDEX_DIR / DenseRetriever.INDEX_FILENAME
        ids_file = FAISS_INDEX_DIR / DenseRetriever.DOCIDS_FILENAME
        if idx_file.exists() and ids_file.exists() and not args.rebuild:
            print(f"[Dense] Index already exists at {FAISS_INDEX_DIR}, skipping.")
            print(f"[Dense] Use --rebuild to regenerate.")
        else:
            print("[Dense] Loading corpus from local snapshot ")
            corpus = load_local_corpus()

            retriever = DenseRetriever()
            retriever.build_index(corpus, FAISS_INDEX_DIR)

        print(f"[Dense] Finished ")

    print("\n" + "=" * 70)
    print("All indexes ready under indexes/")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
