# Architecture — CS4241 RAG Project

**Index:** 10022200110
**Domain:** Ghana presidential election results (2012, 2016, 2020) + 2025 Budget Statement and Economic Policy
**Stack:** Python · sentence-transformers · custom NumPy vector store · Groq (Llama 3.1 8B Instant) · Streamlit · deployed on Streamlit Community Cloud

---

## 1. System diagram

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                          ONE-TIME SETUP                             │
 │                                                                     │
 │  Ghana_Election_Result.csv                                          │
 │  2025-Budget-Statement.pdf                                          │
 │                │                                                    │
 │                ▼                                                    │
 │        data_loader.py ── 866 raw docs ──▶ chunker.py                │
 │                                             │                       │
 │                                             ▼                       │
 │                                      397 chunks                     │
 │                           (98 election groups + 299 PDF windows)    │
 │                                             │                       │
 │                                             ▼                       │
 │                                   embedder.py (MiniLM-L6-v2)        │
 │                                             │                       │
 │                                             ▼                       │
 │                              (397, 384) L2-normalized float32       │
 │                                             │                       │
 │                                             ▼                       │
 │                              vector_store.py (NumPy)  ──┐           │
 │                                                          │           │
 │                                             retriever.py (BM25 fit) │
 └────────────────────────────────────────────┬────────────┬──────────┘
                                              │            │
 ┌────────────────────────────────────────────▼────────────▼──────────┐
 │                          PER-QUERY LOOP                             │
 │                                                                     │
 │   user question ─▶ HybridRetriever.retrieve()                       │
 │                      ├── dense top-15    (NumpyVectorStore)         │
 │                      ├── BM25 top-15     (custom BM25)              │
 │                      ├── RRF fuse        (k=60)                     │
 │                      └── dense rerank    (cosine on pool)           │
 │                                │                                    │
 │                                ▼                                    │
 │                    apply_reputation_boost (feedback loop)           │
 │                                │                                    │
 │                                ▼                                    │
 │                        top-5 chunks                                 │
 │                                │                                    │
 │                                ▼                                    │
 │              PromptBuilder.build() ── system + user + [C1..Cn]      │
 │                                │                                    │
 │                                ▼                                    │
 │                  GroqClient.generate() ── Llama 3.1 8B Instant      │
 │                                │                                    │
 │                                ▼                                    │
 │              verify_citations()  ── grounding audit                 │
 │                                │                                    │
 │                                ▼                                    │
 │        answer + citations + timings → Streamlit UI                  │
 │                                │                                    │
 │                                ▼                                    │
 │        user 👍/👎 ─▶ FeedbackStore ── closes the loop               │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component inventory

| Module | Responsibility | Key design choice |
|---|---|---|
| `data_loader.py` (provided) | Parse CSV rows + PDF pages into raw docs | 866 raw docs |
| `chunker.py` (provided) | Election year-region groups + PDF sentence-window chunks | 397 chunks, 300w/50w overlap, force-split + noisy-chunk filter |
| `modules/embedder.py` | Text → 384-dim dense vectors | `all-MiniLM-L6-v2`, L2-normalized so cosine == dot product; disk cache |
| `modules/vector_store.py` | In-memory NumPy k-NN store | One matmul `V @ q` gives all scores; `argpartition` for O(N) top-k |
| `modules/retriever.py` | Hybrid retrieval | Dense + custom BM25, RRF fusion (k=60), dense-cosine rerank on pool |
| `modules/prompt_builder.py` | Grounded LLM prompt | Explicit refusal sentence, numbered `[C1]` citations, context-budget trimming |
| `modules/llm_client.py` | Groq API wrapper | Lazy import, exponential retry, offline stub mode |
| `modules/pipeline.py` | End-to-end orchestration | `setup()` once + `ask()` per query, per-stage timings |
| `modules/feedback.py` | **Innovation: Chunk Reputation Boosting** | Smoothed `(pos-neg)/(pos+neg+k)` per chunk, bounded boost at retrieval |
| `modules/evaluate.py` | Adversarial eval harness | 9 hand-curated cases across 4 categories |
| `app.py` | Streamlit UI | 5 tabs: Chat, Retrieval inspector, Failure-case demo, Evaluation, Feedback |

---

## 3. Retrieval pipeline in detail

### 3.1 Why hybrid?

Pure dense retrieval **misses exact-match signals** — region names ("Volta"), party acronyms ("NDC"), percentages ("74.97%"). Pure BM25 **misses paraphrases** — "who won" vs. "presidential victor", "price rises" vs. "inflation". Combining them recovers both. Evidence lives in the *Failure-case demo* tab of the app.

### 3.2 Reciprocal Rank Fusion (RRF)

After the two legs each return their top-15, we fuse using:

```
rrf_score(d) = Σ (over rankers r)  1 / (k + rank_r(d))   with k = 60
```

RRF is rank-based, so it doesn't require the dense and BM25 scores to share a scale. `k=60` is the constant from Cormack et al. (2009) — documents ranked #1 contribute `1/61`, #2 contribute `1/62`, etc., which damps tail noise.

### 3.3 Reranking

The fused pool (≤30 unique candidates) is rescored with dense cosine similarity against the query. Then the final score is a min-max-normalized blend:

```
final = 0.6 * rerank_norm + 0.4 * bm25_norm
```

This keeps the semantic signal dominant but lets lexical exactness break ties. Upgrading to a cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) is a drop-in change to `retriever.py` — the rest of the pipeline doesn't care.

---

## 4. Hallucination guard

Three independent layers:

1. **Prompt-level** — the system prompt tells the model to answer *only* from context, to emit a fixed refusal sentence when context is insufficient, and to cite every factual claim with `[C1]`-style tags.
2. **Temperature** — `temperature=0.1` on Groq to suppress invention.
3. **Post-hoc audit** — `verify_citations()` parses the answer, checks that every `[C#]` tag maps to a real context block, and flags `suspicious=True` when context was supplied but the model emitted zero citations and no refusal. The Streamlit UI surfaces this as a yellow warning banner.

The refusal sentence is a fixed string (`REFUSAL_SENTENCE`) so it's machine-checkable in the eval harness.

---

## 5. Feedback-loop innovation: Chunk Reputation Boosting

**Problem.** Most RAG tutorials collect thumbs-up/down and dump them in a log. The ranker you ship is the ranker you ship.

**Solution.** Attribute each rating to the *chunks cited in that answer* (citations are already machine-parseable). Maintain per-chunk positive/negative counts in a JSON file. At retrieval time, add a small, bounded, rank-pool-proportional bonus:

```
reputation(c) = (pos − neg) / (pos + neg + 5)        ∈ [−1, 1]
boost(c)      = α · reputation(c) · max_score_in_pool     (α = 0.1)
```

**Why this shape:**
- Laplace smoothing (`+5`) means 1 upvote isn't enough to pin a chunk to the top.
- Bounded in `[−1, 1]` and scaled by the pool max score, so a vocal minority **cannot drown out** semantic similarity.
- **Query-agnostic** — generalizes from sparse feedback without requiring matching the exact new query.
- **Inspectable** — the Feedback report tab shows top-boosted / top-penalized chunks.
- **Persistent** — JSON survives Streamlit restarts; swap for a DB in production.

**Trade-offs I'm being honest about.** If the user base is small and noisy, a few bad ratings can nudge rankings in the wrong direction. Mitigations in the current design: smoothing, bounded alpha, pool-relative scaling. A stronger mitigation worth adding: require N≥3 ratings before any boost applies (trivial change in `reputation()`).

---

## 6. Adversarial evaluation

9 hand-curated cases across 4 categories:

| Category | # | What we check |
|---|---|---|
| `ground_truth` | 3 | Answer contains expected keyword AND cites a chunk |
| `numeric` | 1 | Answer contains a `%` AND cites a chunk |
| `out_of_scope` | 2 | Model emits the fixed refusal sentence |
| `adversarial` | 3 | False-premise ("CPP won 2020"), fabricated doc ("2030 budget"), absurd claim ("moon missions") — model must refuse |

Reported metrics: `overall_correct_rate`, `ground_truth_accuracy`, `refusal_accuracy_on_adversarial`, `hallucination_rate`, `grounded_rate_on_answers`, `avg_latency_s`.

---

## 7. Per-stage logging

Every stage emits a standard `logging` record with wall-clock time:

```
INFO modules.pipeline: [ask] retrieve: 0.042s -> 5 hits
INFO modules.pipeline: [ask] prompt build: 0.001s (5 context blocks, trimmed=0)
INFO modules.pipeline: [ask] generate: 0.83s (model=llama-3.1-8b-instant, stub=False)
INFO modules.pipeline: [ask] verify: 0.000s | cited=['C1', 'C3'] | suspicious=False | total=0.88s
```

The Streamlit UI surfaces these as a JSON `stage_timings` block per answer for the grader.

---

## 8. Deployment model

- **Runtime:** Streamlit Community Cloud, free tier.
- **Cold start:** ~15–30s (model download on first boot; subsequent boots reuse the HuggingFace cache).
- **Warm query:** embed query (~20ms) + retrieve (~5ms) + Groq call (~0.5–1.5s) + verify (<1ms) ≈ **~1 second end-to-end**.
- **Memory footprint:** 397 × 384 × 4 B = **~0.6 MB** for embeddings; MiniLM model ~80 MB; well under the 1 GB tier ceiling.
- **Secrets:** `GROQ_API_KEY` via `.streamlit/secrets.toml` locally, via App Settings → Secrets on the cloud.
