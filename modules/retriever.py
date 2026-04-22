"""
retriever.py
------------
Hybrid retriever: BM25 (lexical) + dense vectors (semantic) + re-ranking.

Why hybrid?
-----------
Pure dense retrieval misses exact-match signals (e.g. region names, party
acronyms like "NDC", specific percentages).  Pure BM25 misses paraphrases
(e.g. "who won" vs. "presidential victor").  Combining them via reciprocal
rank fusion (RRF) captures both.

Pipeline (per query)
--------------------
1. Dense:   embed query -> top_k_dense from NumpyVectorStore
2. Lexical: tokenize query -> top_k_bm25 from BM25 index
3. Fuse:    reciprocal-rank-fuse the two lists into a candidate pool
4. Rerank:  recompute cosine(q, doc) on the pool, sort, return top_k_final

Failure-case hook
-----------------
`demo_failure_case` turns off one leg of the pipeline at a time and returns
the top-k under each config, so we can show WHY hybrid matters.

Author: index 10022200110
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .embedder import Embedder
from .vector_store import NumpyVectorStore, SearchHit

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tokenization – tiny, dependency-free
# -----------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9%]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is",
    "are", "was", "were", "be", "by", "with", "at", "as", "this", "that",
    "it", "its", "from", "which", "who", "what", "how", "why", "when",
    "where", "do", "does", "did", "has", "have", "had", "but", "not",
    "we", "you", "i", "our", "your", "their", "they", "he", "she",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, keep alphanumerics + %, drop short stopwords."""
    toks = _TOKEN_RE.findall(text.lower())
    return [t for t in toks if len(t) > 1 and t not in _STOP]


# -----------------------------------------------------------------------------
# BM25 – compact custom implementation (no rank_bm25 dependency)
# -----------------------------------------------------------------------------
class BM25:
    """
    Okapi BM25 with the standard k1=1.5, b=0.75 defaults.

    score(q, d) = sum_t IDF(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl))
    IDF(t)      = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = []
        self.doc_lens: np.ndarray | None = None
        self.avgdl: float = 0.0
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.tf: list[Counter] = []
        self.N: int = 0

    def fit(self, docs: list[str]) -> None:
        self.doc_tokens = [tokenize(d) for d in docs]
        self.N = len(self.doc_tokens)
        self.tf = [Counter(toks) for toks in self.doc_tokens]
        self.doc_lens = np.array([len(toks) for toks in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if self.N else 0.0

        df: dict[str, int] = {}
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.df = df
        self.idf = {
            t: float(np.log((self.N - c + 0.5) / (c + 0.5) + 1.0))
            for t, c in df.items()
        }
        logger.info("BM25 fit on %d docs, avgdl=%.1f, vocab=%d",
                    self.N, self.avgdl, len(df))

    def score_query(self, query: str) -> np.ndarray:
        """Return length-N array of BM25 scores."""
        if self.doc_lens is None:
            raise RuntimeError("BM25 not fit.")
        q_toks = tokenize(query)
        if not q_toks:
            return np.zeros(self.N, dtype=np.float32)

        scores = np.zeros(self.N, dtype=np.float32)
        for t in q_toks:
            idf_t = self.idf.get(t)
            if idf_t is None:
                continue
            for i, tf_i in enumerate(self.tf):
                f = tf_i.get(t, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-6))
                scores[i] += idf_t * (f * (self.k1 + 1) / denom)
        return scores

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.score_query(query)
        k = min(k, self.N)
        if k == 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


# -----------------------------------------------------------------------------
# Hybrid retriever
# -----------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    """One final, re-ranked retrieval result with provenance."""
    chunk_id: str
    text: str
    source: str
    metadata: dict[str, Any]
    score: float                      # final blended score
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    rank_sources: list[str] = field(default_factory=list)  # which legs found it


class HybridRetriever:
    """
    Combines:
      * dense retrieval over a NumpyVectorStore
      * BM25 lexical retrieval
      * reciprocal rank fusion to merge candidate pools
      * a final rerank on cosine similarity of the (fused) pool

    The 'rerank' step here is a simple cross-encoder-free rerank: we just
    recompute the dense similarity over the fused candidate set.  With a
    larger project budget you'd swap in a cross-encoder such as
    `cross-encoder/ms-marco-MiniLM-L-6-v2`; the interface wouldn't change.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: NumpyVectorStore,
        chunks: list[dict],
        k_dense: int = 15,
        k_bm25: int = 15,
        k_final: int = 5,
        rrf_k: int = 60,           # standard RRF constant
        dense_weight: float = 0.6, # weights applied AFTER rerank
        bm25_weight: float = 0.4,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunks = chunks
        self.k_dense = k_dense
        self.k_bm25 = k_bm25
        self.k_final = k_final
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        self.bm25 = BM25()
        self.bm25.fit([c["text"] for c in chunks])

    # -------- main entry point --------
    def retrieve(
        self,
        query: str,
        *,
        use_dense: bool = True,
        use_bm25: bool = True,
        use_rerank: bool = True,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Run the hybrid pipeline.  The three bool flags exist so we can build
        failure-case demos by disabling individual legs.
        """
        top_k = top_k or self.k_final
        if not (use_dense or use_bm25):
            raise ValueError("At least one of use_dense / use_bm25 must be True.")

        # --- Leg 1: dense ---
        dense_hits: list[SearchHit] = []
        if use_dense:
            q_vec = self.embedder.encode_one(query)
            dense_hits = self.vector_store.search(q_vec, top_k=self.k_dense)

        # --- Leg 2: BM25 ---
        bm25_hits: list[tuple[int, float]] = []
        if use_bm25:
            bm25_hits = self.bm25.top_k(query, k=self.k_bm25)

        # --- Fuse candidate pools via RRF ---
        candidates = self._rrf_fuse(dense_hits, bm25_hits)

        # --- Rerank (dense cosine on the pooled candidates) ---
        # Note: q_vec is guaranteed to exist here because this branch requires use_dense=True.
        if use_rerank and use_dense and candidates:
            cand_idx = np.array([i for i, _ in candidates], dtype=int)
            cand_vecs = self.vector_store.vectors[cand_idx]
            rerank_scores = cand_vecs @ (q_vec / (np.linalg.norm(q_vec) + 1e-12))
        else:
            rerank_scores = np.array([score for _, score in candidates], dtype=np.float32)

        # Pull per-leg scores for transparency
        dense_score_map = {h.index: h.score for h in dense_hits}
        bm25_score_map = {i: s for i, s in bm25_hits}
        dense_rank_set = {h.index for h in dense_hits}
        bm25_rank_set = {i for i, _ in bm25_hits}

        # Blend: normalize rerank score to [0,1] across candidates, then blend.
        # (BM25 scores are unbounded; we min-max them over the pool too.)
        results: list[RetrievalResult] = []
        rr_norm = _minmax(rerank_scores) if use_rerank else rerank_scores
        bm_arr = np.array([bm25_score_map.get(i, 0.0) for i, _ in candidates], dtype=np.float32)
        bm_norm = _minmax(bm_arr)

        for pos, (i, rrf_score) in enumerate(candidates):
            chunk = self.chunks[i]
            rank_sources = []
            if i in dense_rank_set:
                rank_sources.append("dense")
            if i in bm25_rank_set:
                rank_sources.append("bm25")

            blended = (
                self.dense_weight * float(rr_norm[pos])
                + self.bm25_weight * float(bm_norm[pos])
            )
            results.append(RetrievalResult(
                chunk_id=str(chunk.get("chunk_id", i)),
                text=chunk["text"],
                source=chunk.get("source", "?"),
                metadata=chunk,
                score=blended,
                dense_score=float(dense_score_map.get(i, 0.0)),
                bm25_score=float(bm25_score_map.get(i, 0.0)),
                rerank_score=float(rerank_scores[pos]) if len(rerank_scores) else 0.0,
                rank_sources=rank_sources,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # -------- RRF --------
    def _rrf_fuse(
        self,
        dense_hits: list[SearchHit],
        bm25_hits: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """
        Reciprocal Rank Fusion.

            rrf_score(d) = sum over rankers r of 1 / (k + rank_r(d))

        We use the standard k=60 from Cormack et al. (2009).
        """
        scores: dict[int, float] = {}
        for rank, hit in enumerate(dense_hits):
            scores[hit.index] = scores.get(hit.index, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, (i, _) in enumerate(bm25_hits):
            scores[i] = scores.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return fused

    # -------- failure-case demo --------
    def demo_failure_case(self, query: str, top_k: int = 3) -> dict:
        """
        Run the retriever in four configurations and return a side-by-side
        dict.  Lets us illustrate when hybrid wins vs. either leg alone.
        """
        def _short(results: list[RetrievalResult]) -> list[dict]:
            return [
                {
                    "chunk_id": r.chunk_id,
                    "source": r.source,
                    "score": round(r.score, 4),
                    "preview": r.text[:160].replace("\n", " ") + "...",
                }
                for r in results
            ]

        return {
            "query": query,
            "dense_only":  _short(self.retrieve(query, use_dense=True,  use_bm25=False, use_rerank=False, top_k=top_k)),
            "bm25_only":   _short(self.retrieve(query, use_dense=False, use_bm25=True,  use_rerank=False, top_k=top_k)),
            "hybrid_no_rerank": _short(self.retrieve(query, use_dense=True, use_bm25=True, use_rerank=False, top_k=top_k)),
            "hybrid_full": _short(self.retrieve(query, use_dense=True, use_bm25=True, use_rerank=True,  top_k=top_k)),
        }


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1].  Constant vectors -> all 0s."""
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)
