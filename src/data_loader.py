"""FiQA dataset loading utilities.

Two access patterns are supported:

1. ``download_fiqa()``  pulls the dataset fresh from ``ir_datasets``.
2. ``load_local_*()``  reads from the on-disk snapshot 
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Tuple

import ir_datasets

from config import (
    CORPUS_PATH,
    DATASET_ID,
    QRELS_PATH,
    QUERIES_PATH,
)

Corpus = Dict[str, str]
Queries = Dict[str, str]
Qrels = Dict[str, Dict[str, int]]


#  Direct download from ir_datasets 
def download_fiqa() -> Tuple[Corpus, Queries, Qrels]:
    """Download FiQA via ir_datasets.
    """
    dataset = ir_datasets.load(DATASET_ID)

    corpus: Corpus = {}
    for doc in dataset.docs_iter():
        corpus[doc.doc_id] = doc.text

    queries: Queries = {}
    for q in dataset.queries_iter():
        queries[q.query_id] = q.text

    qrels: Qrels = {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)

    return corpus, queries, qrels


#  Local snapshot writers 
def write_corpus_jsonl(corpus: Corpus, path: Path = CORPUS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc_id, text in corpus.items():
            f.write(json.dumps({"docno": doc_id, "text": text}, ensure_ascii=False))
            f.write("\n")


def write_queries_tsv(queries: Queries, path: Path = QUERIES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid, text in queries.items():
            # Strip newlines from query text just in case (TSV would break otherwise)
            clean = text.replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{qid}\t{clean}\n")


def write_qrels_tsv(qrels: Qrels, path: Path = QRELS_PATH) -> None:
    """Write qrels in TREC format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid, docs in qrels.items():
            for docno, rel in docs.items():
                f.write(f"{qid} 0 {docno} {rel}\n")


# Local snapshot readers 
def load_local_corpus(path: Path = CORPUS_PATH) -> Corpus:
    corpus: Corpus = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            corpus[obj["docno"]] = obj["text"]
    return corpus


def iter_local_corpus(path: Path = CORPUS_PATH) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_local_queries(path: Path = QUERIES_PATH) -> Queries:
    queries: Queries = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            queries[qid] = text
    return queries


def load_local_qrels(path: Path = QRELS_PATH) -> Qrels:
    qrels: Qrels = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            qid, _, docno, rel = line.split()
            qrels.setdefault(qid, {})[docno] = int(rel)
    return qrels


def load_all() -> Tuple[Corpus, Queries, Qrels]:
    """Convenience: load corpus, queries and qrels from the local snapshot."""
    return load_local_corpus(), load_local_queries(), load_local_qrels()
