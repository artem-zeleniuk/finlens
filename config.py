"""Central configuration for the IR search engine.

All hyperparameters live here
"""

from pathlib import Path

#  Paths 
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "indexes"
RESULTS_DIR = PROJECT_ROOT / "results"

CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUERIES_PATH = DATA_DIR / "queries.tsv"
QRELS_PATH = DATA_DIR / "qrels.tsv"
STATS_PATH = DATA_DIR / "stats.json"

BM25_INDEX_DIR = INDEX_DIR / "bm25_index"
FAISS_INDEX_DIR = INDEX_DIR / "faiss_index"

#  Dataset 
DATASET_ID = "beir/fiqa/test"   # ir_datasets identifier

#  Stage 1: BM25 
BM25_K1 = 1.2
BM25_B = 0.75

#  Stage 1: Dense bi-encoder 
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_DIM = 384                 # output dimension of MiniLM-L6
DENSE_BATCH_SIZE = 64           # encoding batch size

#  Stage 1: RRF fusion 
RRF_K = 60                      # standard RRF constant from Cormack et al. 2009

#  Stage 1: Bo1 query expansion 
BO1_FB_DOCS = 3                 # feedback documents
BO1_FB_TERMS = 10               # expansion terms appended to the query

#  Stage 2: Cross-encoder 
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_BATCH_SIZE = 64
CE_MAX_LENGTH = 512

#  Retrieval depths 
TOP_K_STAGE1 = 100              # candidates from Stage 1
TOP_K_FINAL = 10                # final results returned to the user

#  Evaluation 
METRICS_STAGE1 = {"recall_100"}
METRICS_END2END = {"ndcg_cut_10", "map", "recip_rank"}
