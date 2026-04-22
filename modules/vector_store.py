"""
vector_store.py
---------------
Custom NumPy-only vector store.

Rationale
---------
The CS4241 brief requires building a vector store "from scratch" rather than
pulling in FAISS or Chroma.  With only ~400 chunks and 384-dim embeddings the
entire matrix is under 1 MB, so a plain NumPy dot-product search is both
simple and fast (sub-millisecond for k-NN).

Because the embedder L2-normalizes every vector, cosine similarity is just
the dot product:  sim(q, d) = q . d.  No sqrt, no division.

Persistence
-----------
We save two files:
  <name>.npy   -> the (N, D) float32 matrix
  <name>.pkl   -> the list of metadata dicts (one per row), Python-pickled

This keeps the data human-inspectable with numpy and avoids binary formats
tied to external libraries.

Author: index 10022200110
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """One retrieval result."""
    score: float
    metadata: dict[str, Any]
    index: int  # position in the store

    def __repr__(self) -> str:  # readable in notebooks
        src = self.metadata.get("source", "?")
        chunk_id = self.metadata.get("chunk_id", "?")
        return f"SearchHit(score={self.score:.4f}, source={src}, id={chunk_id})"


class NumpyVectorStore:
    """
    Minimal in-memory vector store.

    Expected input:
        vectors:  (N, D) float32 np.ndarray, rows already L2-normalized.
        metadata: list of length N whose i-th element describes row i.
    """

    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None
        self.metadata: list[dict] = []
        self.dim: int | None = None

    # -------- construction --------
    def build(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        if vectors.shape[0] != len(metadata):
            raise ValueError(
                f"vectors ({vectors.shape[0]}) and metadata ({len(metadata)}) size mismatch"
            )
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        self.vectors = vectors
        self.metadata = list(metadata)
        self.dim = vectors.shape[1]
        logger.info("Vector store built: N=%d, D=%d", vectors.shape[0], vectors.shape[1])

    # -------- search --------
    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        """
        Return the top_k most similar stored vectors, by cosine similarity.

        Works because all vectors (and the query, ideally) are L2-normalized,
        so dot product == cosine similarity in [-1, 1].
        """
        if self.vectors is None:
            raise RuntimeError("Vector store is empty. Call build() first.")
        if query_vec.ndim != 1:
            raise ValueError(f"query_vec must be 1-D; got shape {query_vec.shape}")
        if query_vec.shape[0] != self.dim:
            raise ValueError(
                f"query dim {query_vec.shape[0]} != store dim {self.dim}"
            )

        # Safety: re-normalize the query in case caller forgot.
        q = query_vec.astype(np.float32)
        norm = np.linalg.norm(q) + 1e-12
        q = q / norm

        # Core: one matrix-vector product gives all cosine similarities.
        scores = self.vectors @ q  # shape (N,)

        # argpartition is O(N) vs sort's O(N log N); for small N the diff is
        # negligible but it keeps the door open to scaling the corpus later.
        k = min(top_k, scores.shape[0])
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]  # sort just the top k

        return [
            SearchHit(score=float(scores[i]), metadata=self.metadata[i], index=int(i))
            for i in top_idx
        ]

    def search_batch(self, query_matrix: np.ndarray, top_k: int = 5) -> list[list[SearchHit]]:
        """Batched variant for evaluation loops."""
        if self.vectors is None:
            raise RuntimeError("Vector store is empty.")
        # Normalize rows of the query matrix.
        norms = np.linalg.norm(query_matrix, axis=1, keepdims=True) + 1e-12
        Q = (query_matrix / norms).astype(np.float32)

        sims = Q @ self.vectors.T  # (B, N)
        k = min(top_k, sims.shape[1])
        all_hits = []
        for row in sims:
            top_idx = np.argpartition(-row, k - 1)[:k]
            top_idx = top_idx[np.argsort(-row[top_idx])]
            all_hits.append([
                SearchHit(score=float(row[i]), metadata=self.metadata[i], index=int(i))
                for i in top_idx
            ])
        return all_hits

    # -------- persistence --------
    def save(self, path_prefix: str) -> None:
        if self.vectors is None:
            raise RuntimeError("Nothing to save.")
        np.save(f"{path_prefix}.npy", self.vectors)
        with open(f"{path_prefix}.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("Saved store to %s.{npy,pkl}", path_prefix)

    def load(self, path_prefix: str) -> None:
        vec_path = f"{path_prefix}.npy"
        meta_path = f"{path_prefix}.pkl"
        if not (os.path.exists(vec_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(f"Missing {vec_path} or {meta_path}")
        self.vectors = np.load(vec_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        self.dim = self.vectors.shape[1]
        logger.info("Loaded store: N=%d, D=%d", self.vectors.shape[0], self.dim)

    # -------- diagnostics --------
    def __len__(self) -> int:
        return 0 if self.vectors is None else self.vectors.shape[0]

    def stats(self) -> dict:
        if self.vectors is None:
            return {"n": 0, "dim": None}
        return {
            "n": int(self.vectors.shape[0]),
            "dim": int(self.vectors.shape[1]),
            "bytes": int(self.vectors.nbytes),
            "mean_norm": float(np.mean(np.linalg.norm(self.vectors, axis=1))),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Tiny smoke test.
    rng = np.random.default_rng(0)
    V = rng.normal(size=(10, 384)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    store = NumpyVectorStore()
    store.build(V, [{"chunk_id": i, "source": "dummy"} for i in range(10)])
    hits = store.search(V[3], top_k=3)
    print("Top hit should be index 3:", hits[0])
    print("Stats:", store.stats())
