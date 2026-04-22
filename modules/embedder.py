"""
embedder.py
-----------
Converts text chunks into dense vector embeddings using a
sentence-transformers model.

Model choice
------------
We use `sentence-transformers/all-MiniLM-L6-v2`:
  * 384-dim vectors  -> small memory footprint (397 chunks x 384 x 4B = ~0.6 MB)
  * ~80 MB on disk   -> fits comfortably within Streamlit Community Cloud limits
  * Strong on short-to-medium English passages (our chunks are 300 words max)
  * CPU inference is fast enough for 397 chunks (<5s on a cloud CPU)

Author: index 10022200110
Course: CS4241
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

# Default model – chosen for CPU friendliness on Streamlit Community Cloud.
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = 32


class Embedder:
    """
    Thin wrapper around a SentenceTransformer model that:
      * lazy-loads the model (so Streamlit cold starts stay fast),
      * batches inputs,
      * L2-normalizes output vectors so cosine similarity == dot product,
      * can cache embeddings to disk as a .npy + .pkl pair.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_dir: str = "cache",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None  # lazy
        os.makedirs(self.cache_dir, exist_ok=True)

    # -------- model loading --------
    def _load_model(self) -> None:
        """Import + load sentence-transformers only on first real use."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. Install with "
                "`pip install sentence-transformers`."
            ) from e

        logger.info("Loading embedding model: %s", self.model_name)
        t0 = time.time()
        self._model = SentenceTransformer(self.model_name)
        logger.info("Model loaded in %.2fs", time.time() - t0)

    # -------- encoding --------
    def encode(
        self,
        texts: Iterable[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode an iterable of texts -> (N, D) float32 matrix with L2-normalized rows.
        """
        self._load_model()
        texts = list(texts)
        if not texts:
            return np.zeros((0, DEFAULT_EMBEDDING_DIM), dtype=np.float32)

        logger.info("Embedding %d texts (batch_size=%d)", len(texts), self.batch_size)
        t0 = time.time()
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so cosine == dot product
        ).astype(np.float32)
        logger.info("Embedded %d texts in %.2fs", len(texts), time.time() - t0)
        return vectors

    def encode_one(self, text: str) -> np.ndarray:
        """Convenience: encode a single query -> 1-D float32 vector of size D."""
        return self.encode([text])[0]

    # -------- caching --------
    def embed_chunks_with_cache(
        self,
        chunks: list[dict],
        cache_key: str = "chunks_v1",
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Embed a list of chunk dicts (each with a 'text' field) and cache
        the resulting vectors + metadata to disk.  Re-runs are instant.
        """
        vec_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
        meta_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if os.path.exists(vec_path) and os.path.exists(meta_path):
            logger.info("Loading cached embeddings from %s", vec_path)
            vectors = np.load(vec_path)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            if len(meta) == len(chunks) and vectors.shape[0] == len(chunks):
                return vectors, meta
            logger.warning("Cache size mismatch, recomputing.")

        texts = [c["text"] for c in chunks]
        vectors = self.encode(texts, show_progress=False)
        np.save(vec_path, vectors)
        with open(meta_path, "wb") as f:
            pickle.dump(chunks, f)
        logger.info("Wrote cache -> %s (+ metadata)", vec_path)
        return vectors, chunks


if __name__ == "__main__":
    # Smoke test
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    emb = Embedder()
    v = emb.encode(["Nana Akufo-Addo won Ashanti Region in 2020.",
                    "The 2025 Budget emphasizes debt sustainability."])
    print("Shape:", v.shape, "dtype:", v.dtype)
    print("Row norm (should be ~1.0):", float(np.linalg.norm(v[0])))
