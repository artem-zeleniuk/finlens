"""FinLens — Financial Passage Search Engine.

Interactive Streamlit demo for the IR Search Engine project.
Run:  streamlit run demo.py --server.port 8501
"""

from __future__ import annotations

import html as html_lib
import time
from typing import Dict, List

import pandas as pd
import streamlit as st

#  Page config 
st.set_page_config(
    page_title="FinLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#  Custom CSS 
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.finlens-header {
    text-align: center;
    padding: 2rem 0 0.5rem 0;
}
.finlens-logo {
    font-size: 2.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #1a73e8, #00c9a7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}
.finlens-sub {
    color: #6b7280;
    font-size: 1rem;
    margin-top: -0.3rem;
}

div[data-testid="stTextInput"] input {
    font-size: 1.1rem !important;
    padding: 0.8rem 1.2rem !important;
    border-radius: 12px !important;
    border: 2px solid #e5e7eb !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #1a73e8 !important;
    box-shadow: 0 0 0 3px rgba(26,115,232,0.15) !important;
}

.ai-summary {
    background: linear-gradient(135deg, #f0f7ff 0%, #eef9f5 100%);
    border-left: 4px solid #1a73e8;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0 1.5rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
}
.ai-summary-title {
    font-weight: 600;
    color: #1a73e8;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
}

/*  Result card (Google-style)  */
.result-header {
    margin-top: 1.2rem;
    margin-bottom: 0.3rem;
}
.result-rank {
    display: inline-block;
    background: #1a73e8;
    color: white;
    font-weight: 600;
    font-size: 0.75rem;
    padding: 0.1rem 0.5rem;
    border-radius: 5px;
    margin-right: 0.5rem;
    vertical-align: middle;
}
.result-docid {
    color: #1a73e8;
    font-size: 0.95rem;
    font-weight: 600;
    vertical-align: middle;
}
.result-score {
    color: #6b7280;
    font-size: 0.8rem;
    margin-left: 0.8rem;
    vertical-align: middle;
}
.result-snippet {
    color: #374151;
    font-size: 0.9rem;
    line-height: 1.55;
    margin-bottom: 0.2rem;
}
.result-divider {
    border: none;
    border-top: 1px solid #f3f4f6;
    margin: 0.5rem 0 0 0;
}

/*  Pipeline badges  */
.pipeline-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.4rem;
}
.badge-bm25   { background: #fef3c7; color: #92400e; }
.badge-dense  { background: #dbeafe; color: #1e40af; }
.badge-hybrid { background: #d1fae5; color: #065f46; }
.badge-ce     { background: #ede9fe; color: #5b21b6; }
.badge-bo1    { background: #fce7f3; color: #9d174d; }

.timing-chip {
    display: inline-block;
    background: #f3f4f6;
    color: #6b7280;
    font-size: 0.75rem;
    padding: 0.15rem 0.6rem;
    border-radius: 12px;
    margin-left: 0.5rem;
}

.metric-box {
    text-align: center;
    padding: 0.8rem;
    background: #f9fafb;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1a73e8;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 0.2rem;
}

/*  Rank bars  */
.rank-bar-row {
    display: flex;
    align-items: center;
    margin: 0.25rem 0;
    font-size: 0.8rem;
}
.rank-bar-label {
    width: 110px;
    color: #6b7280;
    font-weight: 500;
    flex-shrink: 0;
}
.rank-bar-bg {
    flex: 1;
    height: 10px;
    background: #f3f4f6;
    border-radius: 5px;
    overflow: hidden;
    margin: 0 0.6rem;
}
.rank-bar-fill {
    height: 100%;
    border-radius: 5px;
}
.rank-bar-value {
    width: 70px;
    text-align: right;
    color: #374151;
    font-weight: 600;
    flex-shrink: 0;
}

.block-container { padding-top: 1rem !important; }
</style>
""",
    unsafe_allow_html=True,
)


#  Lazy-load components (cached) 
@st.cache_resource(show_spinner="Loading BM25 index...")
def load_bm25():
    from src.retrieval_bm25 import BM25Retriever
    return BM25Retriever()


@st.cache_resource(show_spinner="Loading dense encoder + FAISS...")
def load_dense():
    from src.retrieval_dense import DenseRetriever
    d = DenseRetriever()
    d.load_index()
    return d


@st.cache_resource(show_spinner="Loading cross-encoder reranker...")
def load_reranker():
    from src.reranker import CrossEncoderReranker
    return CrossEncoderReranker()


@st.cache_resource(show_spinner="Loading query expander...")
def load_query_expander():
    from src.query_expansion import QueryExpander
    return QueryExpander()


@st.cache_resource(show_spinner="Loading corpus...")
def load_corpus():
    import re
    from src.data_loader import load_local_corpus
    corpus = load_local_corpus()
    # Filter out ~89 non-English spam docs in FiQA
    return {
        k: v for k, v in corpus.items()
        if len(v) == 0 or len(re.findall(r'[a-zA-Z]', v)) / len(v) > 0.4
    }


#  Helpers 
def _build_rank_map(df: pd.DataFrame) -> dict[str, int]:
    """Build {docno: 1-indexed rank} from results for qid='0'."""
    if df.empty or "score" not in df.columns:
        return {}
    sub = df[df["qid"] == "0"] if "qid" in df.columns else df
    if sub.empty:
        return {}
    ranked = sub.sort_values("score", ascending=False).reset_index(drop=True)
    return {str(r.docno): i + 1 for i, r in enumerate(ranked.itertuples(index=False))}


#  Pipeline runners 
def run_pipeline_a(query: str, top_k: int = 10) -> dict:
    from src.fusion import rrf_fuse
    bm25 = load_bm25()
    dense = load_dense()
    reranker = load_reranker()
    corpus = load_corpus()
    timings = {}
    q = {"0": query}

    t0 = time.perf_counter()
    bm25_res = bm25.retrieve_batch(q)
    timings["BM25"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dense_res = dense.retrieve_batch(q)
    timings["Dense"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    hybrid_res = rrf_fuse([bm25_res, dense_res])
    timings["RRF Fusion"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reranked = reranker.rerank_batch(q, hybrid_res, corpus, top_k=top_k)
    timings["Cross-Encoder"] = time.perf_counter() - t0

    n = lambda df: len(df[df["qid"] == "0"])
    rank_maps = {
        "BM25": (_build_rank_map(bm25_res), n(bm25_res)),
        "Dense": (_build_rank_map(dense_res), n(dense_res)),
        "RRF Fusion": (_build_rank_map(hybrid_res), n(hybrid_res)),
        "Cross-Encoder": (_build_rank_map(reranked), n(reranked)),
    }
    return {
        "results": reranked, "timings": timings, "rank_maps": rank_maps,
        "badges": ["bm25", "dense", "hybrid", "ce"],
    }


def run_pipeline_b(query: str, top_k: int = 10) -> dict:
    qe = load_query_expander()
    reranker = load_reranker()
    corpus = load_corpus()
    timings = {}
    q = {"0": query}

    t0 = time.perf_counter()
    expanded_res = qe.retrieve_batch(q)
    timings["BM25 + Bo1"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    expanded_query = qe.expand_query(query)
    timings["Query Expansion"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reranked = reranker.rerank_batch(q, expanded_res, corpus, top_k=top_k)
    timings["Cross-Encoder"] = time.perf_counter() - t0

    n = lambda df: len(df[df["qid"] == "0"])
    rank_maps = {
        "BM25 + Bo1": (_build_rank_map(expanded_res), n(expanded_res)),
        "Cross-Encoder": (_build_rank_map(reranked), n(reranked)),
    }
    return {
        "results": reranked, "timings": timings, "rank_maps": rank_maps,
        "expanded_query": expanded_query,
        "badges": ["bo1", "bm25", "ce"],
    }


#  RAG generation 
def generate_rag_answer(query: str, docs: List[dict]) -> str | None:
    try:
        from openai import OpenAI
        api_key = st.session_state.get("api_key", "")
        base_url = st.session_state.get("api_base_url", "https://api.openai.com/v1")
        model = st.session_state.get("llm_model", "gpt-4o-mini")
        if not api_key:
            return None
        client = OpenAI(api_key=api_key, base_url=base_url)
        context = ""
        for i, doc in enumerate(docs[:5], 1):
            context += f"[{i}] {doc['text'][:800]}\n\n"
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "You are a financial expert assistant. Answer the user's question "
                    "based ONLY on the provided documents. Cite sources using [1], [2], etc. "
                    "Be concise (3-5 sentences). If the documents don't contain enough info, say so."
                )},
                {"role": "user", "content": f"Question: {query}\n\nDocuments:\n{context}"},
            ],
            temperature=0.3, max_tokens=300,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"RAG error: {e}"


#  UI components 
SNIPPET_WORDS = 150

def render_badges(badges: list):
    badge_map = {
        "bm25": ("BM25", "badge-bm25"),
        "dense": ("Dense", "badge-dense"),
        "hybrid": ("Hybrid RRF", "badge-hybrid"),
        "ce": ("Cross-Encoder", "badge-ce"),
        "bo1": ("Bo1 Expansion", "badge-bo1"),
    }
    html = ""
    for b in badges:
        label, cls = badge_map.get(b, (b, "badge-bm25"))
        html += f'<span class="pipeline-badge {cls}">{label}</span> &rarr; '
    st.markdown(html.rstrip(" &rarr; "), unsafe_allow_html=True)


def render_timings(timings: dict):
    chips = " ".join(
        f'<span class="timing-chip">{stage}: {t * 1000:.0f}ms</span>'
        for stage, t in timings.items()
    )
    total = sum(timings.values())
    chips += f' <span class="timing-chip">Total: {total:.2f}s</span>'
    st.markdown(chips, unsafe_allow_html=True)


def render_result(rank: int, docno: str, score: float, text: str,
                  rank_maps: dict[str, tuple[dict[str, int], int]]):
    """Render one result: Google-style header + snippet + expander for details."""
    # Header
    st.markdown(
        f'<div class="result-header">'
        f'<span class="result-rank">#{rank}</span>'
        f'<span class="result-docid">Document {docno}</span>'
        f'<span class="result-score">CE score: {score:.4f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Snippet (first N words)
    words = text.split()
    if len(words) > SNIPPET_WORDS:
        snippet = " ".join(words[:SNIPPET_WORDS]) + " ..."
    else:
        snippet = text
    st.markdown(
        f'<div class="result-snippet">{html_lib.escape(snippet)}</div>',
        unsafe_allow_html=True,
    )

    # Expander with full text (if longer) + ranking details
    has_more_text = len(words) > SNIPPET_WORDS
    expander_label = "Full document & ranking details" if has_more_text else "Ranking details"

    with st.expander(expander_label):
        # Full text if truncated
        if has_more_text:
            st.markdown(
                f"<div style='color:#374151;line-height:1.6;font-size:0.9rem;'>"
                f"{html_lib.escape(text)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("---")

        # Rank journey bars
        st.markdown("**Rank journey through pipeline:**")

        colors = {
            "BM25": "#f59e0b", "Dense": "#3b82f6", "RRF Fusion": "#10b981",
            "BM25 + Bo1": "#ec4899", "Cross-Encoder": "#8b5cf6",
        }

        bars = ""
        for stage, (rmap, pool) in rank_maps.items():
            color = colors.get(stage, "#6b7280")
            r = rmap.get(docno)
            if r is not None:
                pct = max((1 - (r - 1) / pool) * 100, 3) if pool > 0 else 0
                bars += f"""<div class="rank-bar-row">
                    <span class="rank-bar-label">{stage}</span>
                    <div class="rank-bar-bg">
                        <div class="rank-bar-fill" style="width:{pct:.0f}%;background:{color};"></div>
                    </div>
                    <span class="rank-bar-value">#{r} / {pool}</span>
                </div>"""
            else:
                bars += f"""<div class="rank-bar-row">
                    <span class="rank-bar-label">{stage}</span>
                    <div class="rank-bar-bg"></div>
                    <span class="rank-bar-value" style="color:#9ca3af;">—</span>
                </div>"""

        st.markdown(bars, unsafe_allow_html=True)

    # Subtle divider
    st.markdown('<hr class="result-divider">', unsafe_allow_html=True)


def get_result_docs(results: pd.DataFrame, corpus: dict) -> List[dict]:
    docs = []
    for row in results.itertuples(index=False):
        docs.append({
            "rank": int(row.rank) + 1,
            "docno": str(row.docno),
            "score": float(row.score),
            "text": corpus.get(str(row.docno), ""),
        })
    return docs


#  Main app 
def main():
    # Header
    st.markdown(
        '<div class="finlens-header">'
        '<div class="finlens-logo">FinLens</div>'
        '<div class="finlens-sub">AI-Powered Financial Document Search</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    #  Sidebar 
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Results to show", 5, 20, 10)
        st.divider()
        st.subheader("AI Summary (RAG)")
        st.caption(
            "Enter your OpenAI API key to enable RAG-powered answers. "
            "The system retrieves relevant documents, then an LLM generates "
            "a concise answer with citations."
        )
        st.text_input("OpenAI API Key", type="password", key="api_key",
                       placeholder="sk-...")
        with st.expander("Advanced LLM settings"):
            st.text_input("API Base URL", value="https://api.openai.com/v1", key="api_base_url")
            st.text_input("Model", value="gpt-4o-mini", key="llm_model")

    #  Search bar 
    col_l, col_search, col_r = st.columns([1, 6, 1])
    with col_search:
        query = st.text_input(
            "search",
            placeholder="Ask a financial question, e.g.: How do I save for retirement?",
            label_visibility="collapsed",
            key="query_input",
        )

    #  Pipeline selector (always visible, under search) 
    pipe_col, compare_col = st.columns([5, 1])
    with pipe_col:
        selected = st.selectbox(
            "Choose retrieval pipeline",
            [
                "Pipeline A: Hybrid (BM25 + Dense + RRF + CE)",
                "Pipeline B: Query Expansion (BM25 + Bo1 + CE)",
            ],
            key="pipeline_select",
        )
    with compare_col:
        st.markdown("<br>", unsafe_allow_html=True)
        show_compare = st.toggle("Compare", key="show_compare")

    pipe_id = "a" if "Hybrid" in selected else "b"

    if not query:
        # Landing page
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                '<div class="metric-box"><div class="metric-value">57,638</div>'
                '<div class="metric-label">Financial Documents</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div class="metric-box"><div class="metric-value">3-Stage</div>'
                '<div class="metric-label">Retrieval Pipeline</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                '<div class="metric-box"><div class="metric-value">AI-Powered</div>'
                '<div class="metric-label">Neural Reranking + RAG</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Try these queries:")
        sc = st.columns(3)
        samples = [
            "How do I save for retirement?",
            "What is a leveraged buyout?",
            "Best way to diversify investment portfolio",
        ]

        def _set_query(s: str):
            st.session_state["query_input"] = s

        for col, sample in zip(sc, samples):
            with col:
                st.button(sample, use_container_width=True,
                          on_click=_set_query, args=(sample,))
        return

    #  Load corpus 
    corpus = load_corpus()

    #  Single pipeline mode 
    if not show_compare:
        if pipe_id == "a":
            with st.spinner("Searching with Pipeline A..."):
                result = run_pipeline_a(query, top_k=top_k)
        else:
            with st.spinner("Searching with Pipeline B..."):
                result = run_pipeline_b(query, top_k=top_k)

        render_badges(result["badges"])
        render_timings(result["timings"])

        docs = get_result_docs(result["results"], corpus)

        # Expanded query for Pipeline B
        if "expanded_query" in result:
            with st.expander("Expanded query (Bo1)", expanded=True):
                st.code(result["expanded_query"], language=None)

        # Placeholder for RAG — will be filled AFTER results render
        rag_placeholder = st.empty()

        # Results render immediately
        if not docs:
            st.warning("No results found for this query.")
        for doc in docs:
            render_result(doc["rank"], doc["docno"], doc["score"], doc["text"],
                          result["rank_maps"])

        # Now fill RAG placeholder (results already visible)
        if st.session_state.get("api_key") and docs:
            with rag_placeholder.container():
                with st.spinner("Generating AI summary..."):
                    t_rag = time.perf_counter()
                    summary = generate_rag_answer(query, docs)
                    t_rag = time.perf_counter() - t_rag
                if summary:
                    st.markdown(
                        f'<div class="ai-summary">'
                        f'<div class="ai-summary-title">AI Summary — powered by RAG'
                        f'<span class="timing-chip" style="margin-left:0.8rem;">'
                        f'LLM: {t_rag:.1f}s</span></div>'
                        f'{summary}</div>',
                        unsafe_allow_html=True,
                    )
        elif not st.session_state.get("api_key") and docs:
            rag_placeholder.info(
                "Add an OpenAI API key in the sidebar to enable AI-generated answers (RAG)."
            )

    #  Compare mode 
    else:
        with st.spinner("Running both pipelines..."):
            result_a = run_pipeline_a(query, top_k=top_k)
            result_b = run_pipeline_b(query, top_k=top_k)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Pipeline A: Hybrid")
            render_badges(result_a["badges"])
            render_timings(result_a["timings"])
            docs_a = get_result_docs(result_a["results"], corpus)
            for doc in docs_a[:5]:
                render_result(doc["rank"], doc["docno"], doc["score"], doc["text"],
                              result_a["rank_maps"])

        with col_b:
            st.markdown("#### Pipeline B: Expansion")
            render_badges(result_b["badges"])
            render_timings(result_b["timings"])
            if "expanded_query" in result_b:
                with st.expander("Expanded query"):
                    st.code(result_b["expanded_query"], language=None)
            docs_b = get_result_docs(result_b["results"], corpus)
            for doc in docs_b[:5]:
                render_result(doc["rank"], doc["docno"], doc["score"], doc["text"],
                              result_b["rank_maps"])

        # Overlap
        st.divider()
        set_a = {d["docno"] for d in docs_a}
        set_b = {d["docno"] for d in docs_b}
        overlap = set_a & set_b
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">{len(overlap)}</div>'
                f'<div class="metric-label">Shared documents</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">{len(set_a - set_b)}</div>'
                f'<div class="metric-label">Only in Pipeline A</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">{len(set_b - set_a)}</div>'
                f'<div class="metric-label">Only in Pipeline B</div></div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
