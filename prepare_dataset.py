"""Download FiQA and save a local snapshot to data/.

Usage
-----
    python prepare_dataset.py
"""

from __future__ import annotations

import sys

from config import CORPUS_PATH, DATA_DIR, QRELS_PATH, QUERIES_PATH
from src.data_loader import (
    download_fiqa,
    write_corpus_jsonl,
    write_qrels_tsv,
    write_queries_tsv,
)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading FiQA via ir_datasets ...")
    corpus, queries, qrels = download_fiqa()

    # Drop empty documents
    corpus = {k: v for k, v in corpus.items() if v and v.strip()}

    print(f"  corpus : {len(corpus):,} documents")
    print(f"  queries: {len(queries):,}")
    print(f"  qrels  : {sum(len(d) for d in qrels.values()):,} judgements")

    print("Writing to data/ ...")
    write_corpus_jsonl(corpus, CORPUS_PATH)
    write_queries_tsv(queries, QUERIES_PATH)
    write_qrels_tsv(qrels, QRELS_PATH)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
