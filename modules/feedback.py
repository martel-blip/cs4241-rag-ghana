"""
feedback.py
-----------
Feedback-loop innovation: Chunk Reputation Boosting (CRB).

Problem
-------
Standard RAG systems throw thumbs-up/down into a log and never close the
loop.  The ranking you see today is the ranking you saw at launch.

Idea
----
Every time a user rates an answer, we attribute that rating to the chunks
that were cited in it (citations are machine-parseable, see
prompt_builder.verify_citations).  Chunks that consistently support useful
answers accumulate positive "reputation"; chunks tied to bad answers
accumulate negative reputation.  At retrieval time we add a small
reputation bonus/penalty to each candidate's score.

This is intentionally:
  - *Query-agnostic* (so sparse feedback generalizes),
  - *Bounded* (we clip the bonus so a vocal minority can't drown out cosine),
  - *Persistent* (JSON on disk survives Streamlit restarts),
  - *Inspectable* (the UI can show top-boosted / top-penalized chunks).

Math
----
reputation(c) = (pos - neg) / (pos + neg + smoothing)      in [-1, 1]
boost(c)      = alpha * reputation(c) * max_blended_score  # scale-matched

where alpha is a small constant (default 0.1) and the max_blended_score
is computed per-query so the boost is proportional to the pool.

Storage
-------
JSON file keyed by chunk_id:
  {"chunk_id_42": {"pos": 3, "neg": 0, "last_seen": "2026-04-21T..."}}

Author: index 10022200110
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Iterable

logger = logging.getLogger(__name__)


class FeedbackStore:
    """Thread-safe JSON-backed counts of user feedback per chunk."""

    def __init__(self, path: str = "feedback.json", smoothing: float = 5.0) -> None:
        self.path = path
        self.smoothing = smoothing
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        self._load()

    # -------- I/O --------
    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self._data = json.load(f)
                logger.info("Loaded feedback for %d chunks from %s",
                            len(self._data), self.path)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to read %s (%s); starting fresh.", self.path, e)
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp, self.path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to persist feedback: %s", e)

    # -------- public API --------
    def record(
        self,
        chunk_ids: Iterable[str],
        rating: int,  # +1 or -1
        question: str = "",
    ) -> None:
        """
        Record a single user rating against the chunks that were cited in
        that answer.  `rating` should be +1 (helpful) or -1 (unhelpful).
        """
        if rating not in (1, -1):
            raise ValueError("rating must be +1 or -1")
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            for cid in chunk_ids:
                entry = self._data.setdefault(cid, {"pos": 0, "neg": 0,
                                                    "last_seen": stamp,
                                                    "last_q": ""})
                if rating == 1:
                    entry["pos"] += 1
                else:
                    entry["neg"] += 1
                entry["last_seen"] = stamp
                entry["last_q"] = question[:120]
            self._save()
        logger.info("Recorded rating=%+d for chunks=%s", rating, list(chunk_ids))

    def reputation(self, chunk_id: str) -> float:
        """
        Laplace-smoothed reputation in [-1, 1].
        Returns 0 for unseen chunks.
        """
        entry = self._data.get(chunk_id)
        if not entry:
            return 0.0
        pos, neg = entry["pos"], entry["neg"]
        return (pos - neg) / (pos + neg + self.smoothing)

    def summary(self, top_n: int = 5) -> dict:
        """Top-boosted + top-penalized chunks for UI display."""
        scored = [
            (cid, self.reputation(cid), e["pos"], e["neg"])
            for cid, e in self._data.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return {
            "total_chunks_rated": len(self._data),
            "top_boosted": scored[:top_n],
            "top_penalized": list(reversed(scored[-top_n:])) if len(scored) >= top_n else [],
        }

    def reset(self) -> None:
        with self._lock:
            self._data = {}
            if os.path.exists(self.path):
                os.remove(self.path)


# -----------------------------------------------------------------------------
# Retrieval-time boost helper
# -----------------------------------------------------------------------------
def apply_reputation_boost(
    results,  # list[RetrievalResult]
    store: FeedbackStore,
    alpha: float = 0.1,
):
    """
    Mutate `results` in place, adding a reputation-based bonus to each
    result's `score`.  Re-sorts by the new score.

    The bonus is scaled by the max observed score in the pool so it stays
    proportional and can never dominate the semantic signal.
    """
    if not results:
        return results
    max_score = max((r.score for r in results), default=1.0) or 1.0
    for r in results:
        rep = store.reputation(r.chunk_id)
        bonus = alpha * rep * max_score
        r.score = r.score + bonus
        # Record the bonus in metadata for UI transparency.
        if isinstance(r.metadata, dict):
            r.metadata = {**r.metadata, "_reputation": rep, "_boost": bonus}
    results.sort(key=lambda r: r.score, reverse=True)
    return results
