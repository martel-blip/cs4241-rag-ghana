"""
app.py
------
Streamlit UI for the CS4241 RAG project.

Tabs:
  1. Chat               -> ask a question, see answer + citations + feedback
  2. Retrieval inspector -> show the BM25/dense/hybrid breakdown per query
  3. Failure-case demo   -> side-by-side: dense-only vs BM25-only vs hybrid
  4. Evaluation          -> run the adversarial test set
  5. Feedback report     -> boosted/penalized chunks

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud: see README.md for steps.

Author: index 10022200110
"""

from __future__ import annotations

import logging
import os
import sys

import streamlit as st

# Make sibling modules importable when launched from anywhere
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from modules import (
    Embedder,
    FeedbackStore,
    GroqClient,
    PromptBuilder,
    RAGPipeline,
    apply_reputation_boost,
    evaluate,
)

# Existing modules provided by the student (per the prompt).
from data_loader import load_all      # noqa: E402
from chunker import chunk_all         # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("app")


# =============================================================================
# Cached setup
# =============================================================================
@st.cache_resource(show_spinner="Loading corpus, building chunks, embedding…")
def build_pipeline() -> tuple[RAGPipeline, list[dict], dict]:
    """Run all one-time stages. Cached so Streamlit reruns are instant."""
    docs = load_all()
    chunks = chunk_all(docs)

    # Adapter: the pipeline expects each chunk to have a 'chunk_id' key,
    # but some chunker variants use 'id'. Alias it here so downstream code
    # doesn't need to care which variant is in use.
    for c in chunks:
        if "chunk_id" not in c and "id" in c:
            c["chunk_id"] = c["id"]

    # Detect whether a Groq key is configured; fall back to stub if not.
    has_key = (
        ("GROQ_API_KEY" in st.secrets if hasattr(st, "secrets") else False)
        or bool(os.getenv("GROQ_API_KEY"))
    )
    llm = GroqClient(stub=not has_key)

    pipeline = RAGPipeline(
        embedder=Embedder(),
        llm=llm,
        prompt_builder=PromptBuilder(),
        top_k=5,
    )
    timings = pipeline.setup(chunks)
    return pipeline, chunks, {"timings": timings, "stub": not has_key}


@st.cache_resource
def get_feedback_store() -> FeedbackStore:
    return FeedbackStore(path="feedback.json")


# =============================================================================
# UI
# =============================================================================
st.set_page_config(
    page_title="Ghana RAG • CS4241",
    page_icon="🇬🇭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }

    /* Subtitle styling */
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Card-like containers */
    .card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    /* Example query buttons */
    .example-btn {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border: 1px solid #0ea5e9;
        border-radius: 8px;
        color: #0c4a6e;
        font-weight: 500;
        padding: 0.75rem 1rem;
        margin: 0.25rem;
        transition: all 0.2s ease;
        text-align: left;
        width: 100%;
    }
    .example-btn:hover {
        background: linear-gradient(135deg, #e0f2fe, #bae6fd);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.2);
    }

    /* Sidebar improvements */
    .sidebar-header {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }

    /* Chat messages */
    .chat-user {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-assistant {
        background: #f1f5f9;
        color: #334155;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
    }

    /* Feedback buttons */
    .feedback-btn {
        border-radius: 20px;
        padding: 8px 16px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .feedback-btn:hover {
        transform: scale(1.05);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        font-weight: 600;
    }

    /* Status indicators */
    .status-success {
        color: #059669;
        font-weight: 600;
    }
    .status-warning {
        color: #d97706;
        font-weight: 600;
    }
    .status-error {
        color: #dc2626;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Main title with custom styling
st.markdown('<h1 class="main-title">🇬🇭 Ghana Elections + 2025 Budget — RAG</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">CS4241 project • index 10022200110 • Groq + custom NumPy vector store + hybrid retrieval</p>', unsafe_allow_html=True)

with st.spinner("🔄 Initializing pipeline…"):
    pipeline, chunks, setup_info = build_pipeline()
feedback = get_feedback_store()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h3 style="margin: 0; color: white;">🚀 System Status</h3>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline metrics in cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: bold; color: #3b82f6;">{len(chunks)}</div>
            <div style="color: #64748b; font-size: 0.9rem;">Chunks Indexed</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: bold; color: #10b981;">{pipeline.vector_store.dim or 0}</div>
            <div style="color: #64748b; font-size: 0.9rem;">Vector Dim</div>
        </div>
        """, unsafe_allow_html=True)

    # LLM status
    stub = setup_info.get("stub", False)
    status_icon = "🧪" if stub else "🚀"
    status_color = "status-warning" if stub else "status-success"
    st.markdown(f'<p class="{status_color}">{status_icon} <strong>LLM mode:</strong> {"stub (no GROQ_API_KEY)" if stub else "Groq API"}</p>', unsafe_allow_html=True)

    with st.expander("⏱️ Setup timings"):
        st.json(setup_info["timings"])

    st.markdown("---")

    st.markdown("### ⚙️ Retrieval Settings")
    top_k = st.slider("Top-k final", 1, 10, 5, help="Number of chunks to retrieve and rank")
    use_feedback_boost = st.toggle("Apply feedback reputation boost", value=True, help="Boost chunks based on user feedback")

    st.markdown("---")

    st.markdown("""
    <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
        <h4 style="margin: 0 0 0.5rem 0; color: #0c4a6e;">📚 Data Sources</h4>
        <p style="margin: 0; font-size: 0.9rem; color: #374151;">
            • Ghana Presidential Election Results (2012/2016/2020)<br>
            • 2025 Budget Statement and Economic Policy
        </p>
    </div>
    """, unsafe_allow_html=True)


tab_chat, tab_retr, tab_fail, tab_eval, tab_fb = st.tabs([
    "💬 Chat", "🔎 Retrieval Inspector", "⚖️ Failure Demo",
    "✅ Evaluation", "📈 Feedback Report",
])


# -----------------------------------------------------------------------------
# Tab 1: Chat
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Tab 1: Chat
# -----------------------------------------------------------------------------
with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []

    # Only show example queries when there's no history yet (ChatGPT-style)
    if not st.session_state.history:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top: 0; color: #1e40af;">💡 Example Queries</h3>
            <p style="color: #64748b; margin-bottom: 1rem;">Click any example to get started:</p>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(2)
        example_queries = [
            "Who won the 2020 presidential election in Ghana?",
            "What was the NDC vote share in Greater Accra in 2016?",
            "What are the key economic policies in the 2025 Budget Statement?",
            "How much is allocated to education in the 2025 budget?",
        ]
        for idx, query in enumerate(example_queries):
            col = cols[idx % 2]
            if col.button(
                f"📌 {query[:50]}…" if len(query) > 50 else f"📌 {query}",
                key=f"ex-{idx}",
                help=f"Try: {query}"
            ):
                st.session_state.query_from_example = query

    # Render chat history above the input
    for past in st.session_state.history:
        st.markdown(f'<div class="chat-user">{past["q"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-assistant">{past["a"]}</div>', unsafe_allow_html=True)

    # Chat input LAST — Streamlit pins it to the bottom of the viewport
    question = st.chat_input("💭 Ask something about Ghana elections or the 2025 Budget…")

    # Pick up example-query click, if any
    if "query_from_example" in st.session_state:
        question = st.session_state.query_from_example
        del st.session_state.query_from_example

    if question:
        st.markdown(f'<div class="chat-user">{question}</div>', unsafe_allow_html=True)

        with st.spinner("🔄 Retrieving + generating…"):
            result = pipeline.ask(question, top_k=top_k)
            if use_feedback_boost:
                apply_reputation_boost(result.retrieval, feedback)

        st.markdown(f'<div class="chat-assistant">{result.answer}</div>', unsafe_allow_html=True)

        # Grounding warning
        g = result.grounding
        if g["suspicious"]:
            st.markdown("""
            <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 12px; margin: 12px 0;">
                ⚠️ <strong>Grounding Warning:</strong> No citations in the answer despite context being provided — possible hallucination.
            </div>
            """, unsafe_allow_html=True)
        elif g["unknown_tags"]:
            st.markdown(f"""
            <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 12px; margin: 12px 0;">
                ⚠️ <strong>Citation Warning:</strong> Answer cited unknown tags: {g['unknown_tags']}
            </div>
            """, unsafe_allow_html=True)

        # Citations
        if result.citations:
            with st.expander(f"📚 Citations ({len(result.citations)})", expanded=False):
                for c in result.citations:
                    st.markdown(f"""
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin: 8px 0;">
                        <strong style="color: #3b82f6;">[{c['tag']}]</strong> ·
                        <em style="color: #64748b;">{c['source']}</em> ·
                        <code style="background: #e5e7eb; padding: 2px 4px; border-radius: 4px;">{c['chunk_id']}</code> ·
                        score <code>{c['score']:.3f}</code> · via {c['rank_sources']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(c["text"][:500] + ("…" if len(c["text"]) > 500 else ""))
                    st.markdown("---")

        # Performance metrics
        with st.expander("⏱️ Performance Metrics"):
            t = result.timings
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Retrieval", f"{round(t.retrieve, 3)}s")
                st.metric("Generation", f"{round(t.generate, 3)}s")
            with col2:
                st.metric("Prompt Build", f"{round(t.prompt, 3)}s")
                st.metric("Verification", f"{round(t.verify, 3)}s")
            with col3:
                st.metric("Total", f"{round(t.total, 3)}s")
                st.metric("Model", result.llm.model or "unknown")

            if result.llm.prompt_tokens is not None and result.llm.completion_tokens is not None:
                st.metric("Tokens", f"{result.llm.prompt_tokens} → {result.llm.completion_tokens}")

        # Feedback buttons
        st.markdown("### 💬 Was this helpful?")
        col1, col2, _ = st.columns([1, 1, 5])
        cited_ids = [c["chunk_id"] for c in result.citations
                     if c["tag"] in g["cited_tags"]]
        if not cited_ids:
            cited_ids = [c["chunk_id"] for c in result.citations]

        if col1.button("👍 Helpful", key=f"up-{len(st.session_state.history)}",
                     help="This answer was accurate and useful"):
            feedback.record(cited_ids, rating=+1, question=question)
            st.success("✅ Thanks for the feedback! This will improve future responses.")

        if col2.button("👎 Not helpful", key=f"down-{len(st.session_state.history)}",
                     help="This answer was inaccurate or unhelpful"):
            feedback.record(cited_ids, rating=-1, question=question)
            st.error("📝 Thanks for the feedback! We'll use this to improve the system.")

        # Append to history AFTER rendering, so this turn is not rendered twice
        st.session_state.history.append({"q": question, "a": result.answer})


# -----------------------------------------------------------------------------
# Tab 2: Retrieval inspector
# -----------------------------------------------------------------------------
with tab_retr:
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0; color: #1e40af;">🔎 Retrieval Inspector</h3>
        <p style="color: #64748b; margin-bottom: 1rem;">Analyze how the retrieval system ranks and scores chunks for any query.</p>
    </div>
    """, unsafe_allow_html=True)

    q = st.text_input("Query", value="Who won the Greater Accra Region in 2020?",
                     placeholder="Enter a question to inspect retrieval results…")

    if st.button("🔍 Inspect Retrieval", key="inspect-btn", type="primary"):
        with st.spinner("Analyzing retrieval…"):
            hits = pipeline.retriever.retrieve(q, top_k=top_k)
            if use_feedback_boost:
                apply_reputation_boost(hits, feedback)

        st.markdown(f"### 📊 Top {len(hits)} Results")

        for i, r in enumerate(hits, 1):
            score_color = "#10b981" if r.score > 0.7 else "#f59e0b" if r.score > 0.5 else "#ef4444"

            with st.expander(
                f"#{i} [{r.chunk_id}] {r.source} · Score: {r.score:.3f} · "
                f"Dense: {r.dense_score:.3f} · BM25: {r.bm25_score:.3f} · Via: {r.rank_sources}",
                expanded=i <= 3  # Expand top 3 results
            ):
                st.markdown(f"""
                <div style="background: #f8fafc; border-left: 4px solid {score_color}; padding: 12px; margin: 8px 0; border-radius: 4px;">
                    <strong style="color: {score_color};">Relevance Score: {r.score:.3f}</strong><br>
                    <small style="color: #64748b;">
                        Dense similarity: {r.dense_score:.3f} |
                        BM25 score: {r.bm25_score:.3f} |
                        Ranking method: {r.rank_sources}
                    </small>
                </div>
                """, unsafe_allow_html=True)
                st.write(r.text)


# -----------------------------------------------------------------------------
# Tab 3: Failure-case demo
# -----------------------------------------------------------------------------
with tab_fail:
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0; color: #1e40af;">⚖️ Retrieval Method Comparison</h3>
        <p style="color: #64748b; margin-bottom: 1rem;">
            Compare different retrieval strategies side-by-side. BM25 excels at exact matches,
            dense retrieval handles paraphrases, and hybrid methods combine the best of both.
        </p>
    </div>
    """, unsafe_allow_html=True)

    q = st.text_input(
        "Demo query",
        value="What percent did the NDC get in Volta Region in 2020?",
        key="fail-q",
        placeholder="Try queries with exact terms vs. paraphrases…"
    )

    if st.button("🚀 Run Comparison", key="fail-btn", type="primary"):
        with st.spinner("Running retrieval comparison…"):
            report = pipeline.retriever.demo_failure_case(q, top_k=3)

        st.markdown("### 📈 Retrieval Method Results")

        methods = ["dense_only", "bm25_only", "hybrid_no_rerank", "hybrid_full"]
        method_names = ["Dense Only", "BM25 Only", "Hybrid (No Rerank)", "Hybrid + Rerank"]
        method_descriptions = [
            "Semantic similarity only",
            "Keyword matching only",
            "Combined scores, no final reranking",
            "Combined + dense reranking"
        ]

        cols = st.columns(4)
        for col, method, name, desc in zip(cols, methods, method_names, method_descriptions):
            with col:
                st.markdown(f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin: 8px 0; text-align: center;">
                    <h4 style="margin: 0 0 8px 0; color: #1e40af;">{name}</h4>
                    <p style="margin: 0; font-size: 0.8rem; color: #64748b;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

                for hit in report[method]:
                    score_color = "#10b981" if hit['score'] > 0.5 else "#f59e0b" if hit['score'] > 0.3 else "#ef4444"
                    st.markdown(f"""
                    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px; margin: 4px 0;">
                        <code style="background: {score_color}20; color: {score_color}; padding: 2px 4px; border-radius: 4px; font-size: 0.8rem;">
                            {hit['chunk_id']}
                        </code>
                        <span style="color: {score_color}; font-weight: bold;">{hit['score']}</span>
                        <br><small style="color: #64748b;">{hit['source']}</small>
                        <p style="margin: 4px 0; font-size: 0.85rem;">{hit['preview'][:100]}…</p>
                    </div>
                    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Tab 4: Evaluation
# -----------------------------------------------------------------------------
with tab_eval:
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0; color: #1e40af;">✅ Adversarial Evaluation</h3>
        <p style="color: #64748b; margin-bottom: 1rem;">
            Test the system against a curated set of challenging questions including
            ground-truth queries, out-of-scope questions, and adversarial cases.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🧪 Run Full Evaluation", key="eval-btn", type="primary"):
        with st.spinner("Running comprehensive evaluation…"):
            report = evaluate(pipeline)

        # Summary metrics — keys emitted by modules.evaluate.evaluate()
        summary = report["summary"]
        cases = report["cases"]

        n_total = summary["n_cases"]
        correct_rate = summary["overall_correct_rate"]      # already a fraction in [0,1]
        hallucination_rate = summary["hallucination_rate"]  # fraction of OOS/adv that weren't refused
        refused_cases = sum(1 for c in cases if c.get("is_refusal"))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Questions", n_total)
        with col2:
            st.metric("Correct", f"{correct_rate * 100:.1f}%")
        with col3:
            st.metric("Hallucinations", f"{hallucination_rate * 100:.1f}%")
        with col4:
            refusal_pct = (refused_cases / n_total * 100) if n_total else 0
            st.metric("Refusals", f"{refusal_pct:.1f}%")

        # Per-category breakdown
        st.markdown("### 📊 Per-Category Breakdown")
        for cat, stats in summary["per_category"].items():
            st.markdown(
                f"**{cat}** · n={stats['n']} · "
                f"correct={stats['correct']*100:.0f}% · "
                f"refused={stats['refused']*100:.0f}% · "
                f"grounded={stats['grounded']*100:.0f}%"
            )

        # Detailed per-case results
        with st.expander("📋 Detailed Results", expanded=True):
            for case in cases:
                if case.get("correct"):
                    status_label, status_color = "Correct", "#10b981"
                elif case.get("is_refusal"):
                    status_label, status_color = "Refused", "#6b7280"
                elif case.get("category") in ("out_of_scope", "adversarial"):
                    status_label, status_color = "Hallucination", "#f59e0b"
                else:
                    status_label, status_color = "Incorrect", "#ef4444"

                st.markdown(f"""
                <div style="background: #f8fafc; border-left: 4px solid {status_color}; padding: 12px; margin: 8px 0; border-radius: 4px;">
                    <strong style="color: {status_color};">{status_label}</strong> ·
                    <em>{case["category"]}</em><br>
                    <strong>Q:</strong> {case["question"]}<br>
                    <strong>A:</strong> {case["answer"][:200]}{"…" if len(case["answer"]) > 200 else ""}<br>
                    <small style="color: #64748b;">Cited: {case.get("cited_tags", [])} · Latency: {case.get("latency_s", 0):.2f}s</small>
                </div>
                """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Tab 5: Feedback report
# -----------------------------------------------------------------------------
with tab_fb:
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0; color: #1e40af;">📈 Chunk Reputation System</h3>
        <p style="color: #64748b; margin-bottom: 1rem;">
            Monitor how user feedback influences chunk reputation scores.
            Helpful chunks get boosted, unhelpful ones get penalized.
        </p>
    </div>
    """, unsafe_allow_html=True)

    summary = feedback.summary(top_n=5)

    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunks Rated", summary["total_chunks_rated"])
    with col2:
        combined = summary["top_boosted"] + summary["top_penalized"]
        avg_reputation = (sum(rep for _, rep, _, _ in combined) / len(combined)) if combined else 0.0
        st.metric("Avg Reputation", f"{avg_reputation:+.2f}")
    with col3:
        total_feedback = sum(pos + neg for _, _, pos, neg in combined)
        st.metric("Total Feedback", total_feedback)

    # Top chunks
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏆 Most Helpful Chunks")
        if summary["top_boosted"]:
            for cid, rep, pos, neg in summary["top_boosted"]:
                st.markdown(f"""
                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 8px; margin: 4px 0;">
                    <code style="color: #166534;">{cid}</code> ·
                    <strong style="color: #166534;">{rep:+.2f}</strong>
                    <small style="color: #166534;">(+{pos}/-{neg})</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No positive feedback yet")

    with col2:
        st.markdown("### ⚠️ Least Helpful Chunks")
        if summary["top_penalized"]:
            for cid, rep, pos, neg in summary["top_penalized"]:
                st.markdown(f"""
                <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; padding: 8px; margin: 4px 0;">
                    <code style="color: #991b1b;">{cid}</code> ·
                    <strong style="color: #991b1b;">{rep:+.2f}</strong>
                    <small style="color: #991b1b;">(+{pos}/-{neg})</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No negative feedback yet")

    if st.button("🔄 Reset All Feedback", key="reset-btn", type="secondary"):
        feedback.reset()
        st.success("✅ Feedback data cleared. The system will relearn from new interactions.")