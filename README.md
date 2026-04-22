# Ghana Elections + 2025 Budget — RAG

**CS4241 project · index 10022200110**

A retrieval-augmented QA system over two Ghanaian public-interest documents:

- Ghana Presidential Election results for 2012, 2016, 2020
- The Government of Ghana **2025 Budget Statement and Economic Policy**

Stack: **Groq (Llama 3.1 8B Instant) · sentence-transformers · custom NumPy vector store · hybrid BM25 + dense retrieval · Streamlit**, deployed on Streamlit Community Cloud.

---

## Feature summary

- **Ingestion** — 866 raw docs → 397 chunks (98 election year-region groups + 299 sentence-window PDF chunks, 300w / 50w overlap).
- **Embeddings** — `all-MiniLM-L6-v2`, 384 dims, L2-normalized, disk-cached.
- **Vector store** — from-scratch NumPy, cosine = dot product, `argpartition` top-k.
- **Hybrid retrieval** — dense + custom BM25, fused with Reciprocal Rank Fusion (k=60), then dense-cosine reranking on the pooled candidates.
- **Prompt & hallucination guard** — strict system prompt, numbered `[C1]` citations, post-hoc grounding audit flags suspicious answers.
- **Feedback-loop innovation** — Chunk Reputation Boosting: each thumbs-up/down updates per-chunk reputation and influences future retrieval, bounded to prevent runaway effects.
- **Adversarial evaluation** — ground-truth, numeric, out-of-scope, and false-premise cases with reported hallucination and refusal rates.
- **Streamlit UI** — Chat, Retrieval inspector, Failure-case demo, Evaluation, Feedback report.

---

## Repository layout

```
rag_project/
├── app.py                        # Streamlit UI
├── data_loader.py                # (your module) CSV + PDF loader
├── chunker.py                    # (your module) chunking with filters
├── modules/
│   ├── __init__.py
│   ├── embedder.py               # MiniLM wrapper, disk cache
│   ├── vector_store.py           # NumPy-only k-NN
│   ├── retriever.py              # BM25 + dense + RRF + rerank + failure-case demo
│   ├── prompt_builder.py         # grounded prompt + citation verifier
│   ├── llm_client.py             # Groq wrapper (+ offline stub)
│   ├── pipeline.py               # orchestrator, per-stage logging
│   ├── feedback.py               # Chunk Reputation Boosting
│   └── evaluate.py               # adversarial test set
├── tests/
│   ├── test_smoke.py             # unit tests (no network)
│   └── test_integration.py       # end-to-end (fake embedder offline)
├── docs/
│   └── architecture.md           # full architecture doc
├── .streamlit/
│   ├── config.toml               # theme + server settings
│   └── secrets.toml.example      # copy to secrets.toml and fill in
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Local setup

```bash
# 1. Clone and enter
git clone <your-repo-url> rag_project
cd rag_project

# 2. Create and activate a virtualenv (Python 3.10+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Supply your Groq API key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and paste your key

# 5. Ensure data is in place
# The app expects the two source files alongside app.py (or wherever
# your data_loader.py reads them from):
#   Ghana_Election_Result.csv
#   2025-Budget-Statement-and-Economic-Policy_v4.pdf

# 6. Run
streamlit run app.py
```

The app auto-falls-back to a **deterministic stub LLM** if `GROQ_API_KEY` is missing, so you can iterate on the UI offline.

---

## Run the tests

```bash
python -m unittest discover tests -v
```

15 tests, 1 documented expected failure (requires real MiniLM embeddings — passes at runtime, not in the offline integration harness).

---

## Deploy on Streamlit Community Cloud

1. **Push to GitHub.** Make sure `.streamlit/secrets.toml` is gitignored (it already is in this repo). Your repo root should contain `app.py`, `requirements.txt`, `modules/`, `data_loader.py`, `chunker.py`, and the two data files.

2. **Create the app.** Go to <https://share.streamlit.io>, click *New app*, pick your repo, set:
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **Python version:** 3.11 (match your local version)

3. **Add the secret.** *App settings → Secrets* → paste:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key_here"
   ```
   Save. The app will auto-redeploy.

4. **First boot.** Expect ~30–60s while MiniLM downloads (~80 MB). Subsequent boots reuse the cache and are much faster.

5. **Verify.** In the sidebar you should see:
   - `Chunks indexed: 397`
   - `Vector dim: 384`
   - `LLM mode: 🚀 Groq API` (not `stub`)

6. **Share.** The public URL is `https://<your-app>.streamlit.app`.

### Troubleshooting

| Symptom | Fix |
|---|---|
| Sidebar shows `🧪 stub` after setting the secret | Click *Rerun* or *Reboot app* in the cloud UI |
| `ModuleNotFoundError: No module named 'modules'` | Make sure `modules/__init__.py` is in the repo |
| `FileNotFoundError: Ghana_Election_Result.csv` | Commit the data files at the path your `data_loader.py` expects |
| Cold start times out | Community Cloud has a 10-min boot budget; MiniLM well within it |
| App OOM | Reduce `top_k`, or switch to a quantized model in `embedder.py` |

---

## Grading checklist (CS4241)

| Requirement | Where to look |
|---|---|
| Data ingestion + chunking | `data_loader.py`, `chunker.py` (397 chunks, hybrid strategy) |
| Embeddings | `modules/embedder.py` |
| **Custom** vector store (not FAISS/Chroma) | `modules/vector_store.py` |
| Retriever with hybrid search | `modules/retriever.py` (BM25 + dense + RRF + rerank) |
| Re-ranking step | `HybridRetriever.retrieve()` rerank block |
| Failure-case demo (where retrieval breaks) | `retriever.demo_failure_case()` + Streamlit *Failure-case demo* tab |
| Prompt builder | `modules/prompt_builder.py` |
| Hallucination guard | System prompt + `verify_citations()` + UI warning banner |
| Full pipeline with per-stage logging | `modules/pipeline.py`, `StageTimings` in every `RAGAnswer` |
| Streamlit UI | `app.py`, 5 tabs |
| Feedback-loop innovation | `modules/feedback.py`, Chunk Reputation Boosting |
| Adversarial evaluation | `modules/evaluate.py`, Streamlit *Evaluation* tab |
| Architecture doc | `docs/architecture.md` |
| Deploy | Streamlit Community Cloud, steps above |

---

## License & credits

Built for CS4241. Data: Ghana Electoral Commission presidential results; Ministry of Finance 2025 Budget Statement. Models: `sentence-transformers/all-MiniLM-L6-v2` (Apache 2.0), Groq-hosted Llama 3.1.
