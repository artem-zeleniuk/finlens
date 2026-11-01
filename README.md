# FinLens: multi-stage hybrid retrieval for financial question answering

FinLens is a two-stage search engine for financial passage retrieval, evaluated on the FiQA-2018 test set (57,638 passages, 648 test queries). Two competing first-stage strategies feed the same cross-encoder reranker, and all configurations are scored with pytrec_eval.

## Research question

People ask financial questions in everyday language, while the passages that answer them are written in technical financial terminology. This vocabulary mismatch caps what a lexical first stage can retrieve, and in a cascade the reranker can only reorder documents that the first stage has already found. First-stage recall is therefore a hard ceiling on end-to-end quality.

The project asks which first-stage strategy raises recall more in this domain:

- **neural hybrid fusion**, which adds a dense bi-encoder next to BM25 and merges the two ranked lists with Reciprocal Rank Fusion (Pipeline A), or
- **enriched lexical retrieval**, which expands the query with Bo1 pseudo-relevance feedback terms and stays within BM25 (Pipeline B).

Both are followed by the same BERT cross-encoder reranker. The design report expected that the bi-encoder, trained on MS MARCO rather than financial text, might transfer poorly and that query expansion might therefore do better.

## Answer

- Dense retrieval raises recall, query expansion barely does. Recall@100 rises from 0.559 for BM25 to 0.706 for the dense bi-encoder. Bo1 expansion barely moves it (0.563).
- The best end-to-end configuration is Dense plus cross-encoder, with nDCG@10 = 0.380. This is about 50% above the BM25 baseline (0.253).
- RRF fusion did not beat dense retrieval alone. The hybrid essentially matches the dense Recall@100 (0.7055 against 0.7061 in the log) and has a lower nDCG@10, both before reranking (0.359 against 0.369) and after it (0.370 against 0.380).
- The cross-encoder helps every first stage, for example BM25 rises from 0.253 to 0.347 nDCG@10, but it does not close the gap left by a weaker candidate pool.
- Choosing the Bo1 feedback documents with the dense retriever (DenseBo1) did not help either. Its nDCG@10, MAP and MRR@10 equal those of plain BM25 to four decimals, which suggests that the dense feedback changed the expanded queries very little. We did not investigate this further.

The expectation stated in the design report is not supported. Despite its MS MARCO training, the zero-shot bi-encoder transfers well to FiQA, and semantic matching outperforms lexical enrichment.

## Pipelines

| Pipeline | Stage 1 (top 100 candidates) | Stage 2 (top 10 results) |
|----------|------------------------------|--------------------------|
| A, Hybrid | BM25 and dense bi-encoder, fused by RRF (k = 60) | Cross-encoder |
| B, Query expansion | BM25 first pass, Bo1 expansion (3 feedback documents, 10 expansion terms), BM25 second pass | Cross-encoder |
| B variant, DenseBo1 | Dense retriever selects the 3 feedback documents, then Bo1 and BM25 as in Pipeline B | Cross-encoder |

Each Stage 1 retriever is also evaluated on its own as a baseline. All hyperparameters are set in `config.py`.

## Results

All numbers come from the logged run in `results/experiment_log_20260407_202643.txt`, rounded to three decimals. The log also reports MAP and MRR@10 for every configuration.

| Config | Stage 1 | Stage 2 | Recall@100 | nDCG@10 |
|--------|---------|---------|-----------|---------|
| BM25 | Lexical | None | 0.559 | 0.253 |
| Dense | Dense | None | 0.706 | 0.369 |
| Hybrid (RRF) | BM25 + Dense | None | 0.706 | 0.359 |
| BM25+Bo1 | Expanded lexical | None | 0.563 | 0.244 |
| DenseBo1 | Dense feedback + Bo1 | None | 0.560 | 0.253 |
| BM25+CE | Lexical | Cross-encoder | 0.559 (a) | 0.347 |
| Dense+CE | Dense | Cross-encoder | 0.706 (a) | 0.380 |
| Hybrid+CE | BM25 + Dense | Cross-encoder | 0.706 (a) | 0.370 |
| BM25+Bo1+CE | Expanded lexical | Cross-encoder | 0.563 (a) | 0.345 |
| DenseBo1+CE | Dense feedback + Bo1 | Cross-encoder | 0.560 (a) | 0.347 |

(a) Not measured separately. The cross-encoder only reorders the 100 Stage 1 candidates, so the Recall@100 of its candidate pool is by construction the value of the corresponding Stage 1 configuration.

## Setup

### Prerequisites

- Python 3.10 or newer
- Java 11 or newer (required by PyTerrier)
- conda (recommended)

### Installation

```bash
# Create and activate an environment
conda create -n ir_search python=3.10 -y
conda activate ir_search

# Install dependencies
pip install -r requirements.txt

# Optional: GPU acceleration for FAISS (requires CUDA)
pip install faiss-gpu

# Optional: the AI summaries in the demo use the OpenAI client
pip install openai
```

### Data preparation and indexing

```bash
# Step 1: download FiQA via ir_datasets into data/
python prepare_dataset.py

# Step 2: build the BM25 inverted index and the FAISS dense index
python build_indexes.py
```

This creates:

- `indexes/bm25_index/`, a PyTerrier inverted index
- `indexes/faiss_index/`, a FAISS IndexFlatIP over 384-dimensional embeddings (exact search)

## Usage

### Run the evaluation

```bash
python run_experiments.py
```

This runs every configuration in the table above, plus two extra variants that fuse the DenseBo1 run with the dense run by RRF (reported in the log as DenseBo1+Hyb and DenseBo1+Hyb+CE). It prints Recall@100, nDCG@10, MAP and MRR@10 and writes a timestamped log to `results/`. In the logged run, each cross-encoder configuration took between 230 and 242 seconds.

### Interactive demo

```bash
conda activate ir_search
streamlit run demo.py
```

The FinLens web interface opens at `http://localhost:8501`. It offers:

- selection of Pipeline A or Pipeline B
- a compare mode that shows both pipelines side by side
- a rank journey view that follows each document through the stages
- optional AI summaries of the top results (retrieval-augmented generation). These need your own OpenAI API key, which you enter in the sidebar at run time. No key is stored in this repository.

## Models

| Component | Model | Details |
|-----------|-------|---------|
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` | 384 dimensions, 6 layers, zero-shot |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Zero-shot, same MiniLM family |
| BM25 | PyTerrier | k1 = 1.2, b = 0.75 |

All neural components are used zero-shot, without fine-tuning on financial data. The models are downloaded from the Hugging Face Hub on first use.

## Data

The project uses the FiQA-2018 collection in its BEIR version (`beir/fiqa/test` in ir_datasets). The dataset is not redistributed in this repository. `prepare_dataset.py` downloads it and writes a local snapshot to `data/`, dropping empty passages, and in the logged run 57,600 passages were indexed. The `data/` and `indexes/` folders are generated locally and are excluded from version control.

The pipelines are assembled from the modules in `src/` by `run_experiments.py` (evaluation) and `demo.py` (interface). All hyperparameters are in `config.py`.

## Documents

- [`docs/design_report.pdf`](docs/design_report.pdf): design report with the literature review, the search engine design and the research question.
- [`docs/slides.pdf`](docs/slides.pdf): presentation of the implementation and the results.
