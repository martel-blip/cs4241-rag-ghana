"""
chunker.py
----------
Two chunking strategies — one per source type:

1. Election rows  -> group by (year, new_region). Every (year, region) pair
   becomes one chunk that summarizes all candidates for that region-year.
   Yields 98 chunks across 8 election years.

2. PDF pages      -> sentence-window chunking. We concatenate all page text,
   split into sentences, and slide a window of ~300 words with ~50-word
   overlap (stride 250). Oversized sentences are force-split. A noisy-chunk
   filter drops boilerplate (TOC dot-leaders, page numbers, excessive punct).

Target: 98 + ~300 = ~400 chunks.

Author: index 10022200110
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Iterable

logger = logging.getLogger(__name__)

WINDOW_WORDS = 300
OVERLAP_WORDS = 50
STRIDE_WORDS = WINDOW_WORDS - OVERLAP_WORDS
MIN_CHUNK_WORDS = 30


def _slug(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip().lower())
    return re.sub(r"[^a-z0-9_]", "", s)


def chunk_election(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group rows by (year, new_region) -> one chunk per group."""
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r["year"], r["new_region"])
        groups[key].append(r)

    chunks: list[dict[str, Any]] = []
    for (year, region), rs in sorted(groups.items()):
        def _votes_key(r):
            try:
                return -int(str(r["votes"]).replace(",", ""))
            except ValueError:
                return 0
        rs_sorted = sorted(rs, key=_votes_key)

        lines = [f"Ghana {year} presidential election results for {region}:"]
        for r in rs_sorted:
            lines.append(
                f"- {r['candidate']} ({r['party']}, {r['code']}): "
                f"{r['votes']} votes ({r['votes_pct']})."
            )
        text = "\n".join(lines)

        chunks.append({
            "text": text,
            "source": "Ghana_Election_Result.csv",
            "type": "election_group",
            "chunk_id": f"election_{year}_{_slug(region)}",
            "year": year,
            "region": region,
            "n_candidates": len(rs_sorted),
        })

    logger.info("Election: built %d year-region chunks from %d rows",
                len(chunks), len(rows))
    return chunks


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _force_split(sentence: str, max_words: int = WINDOW_WORDS) -> list[str]:
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _is_noisy(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 80:
        return True
    dots = stripped.count(".")
    if dots / max(len(stripped), 1) > 0.30:
        return True
    alpha = sum(c.isalpha() for c in stripped)
    if alpha / max(len(stripped), 1) < 0.22:
        return True
    real_words = [w for w in stripped.split() if any(c.isalpha() for c in w)]
    if len(real_words) < MIN_CHUNK_WORDS:
        return True
    return False


def chunk_pdf(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sentence-window chunking over concatenated PDF text."""
    sentences: list[tuple[str, int]] = []
    for page_doc in pages:
        page_no = page_doc["page"]
        for sent in _split_sentences(page_doc["text"]):
            for piece in _force_split(sent):
                sentences.append((piece, page_no))

    chunks: list[dict[str, Any]] = []
    i = 0
    kept = 0
    dropped = 0
    while i < len(sentences):
        window_words: list[str] = []
        start_page = sentences[i][1]
        pages_in_window: set[int] = set()
        j = i
        while j < len(sentences) and len(window_words) < WINDOW_WORDS:
            sent_text, sent_page = sentences[j]
            window_words.extend(sent_text.split())
            pages_in_window.add(sent_page)
            j += 1

        text = " ".join(window_words[:WINDOW_WORDS])

        if not _is_noisy(text):
            chunks.append({
                "text": text,
                "source": "2025-Budget-Statement-and-Economic-Policy_v4.pdf",
                "type": "pdf_sentence_window",
                "chunk_id": f"budget_p{start_page:03d}_c{kept:03d}",
                "page": start_page,
                "pages": sorted(pages_in_window),
                "n_words": len(text.split()),
            })
            kept += 1
        else:
            dropped += 1

        advanced_words = 0
        step = 0
        while i + step < j and advanced_words < STRIDE_WORDS:
            advanced_words += len(sentences[i + step][0].split())
            step += 1
        i += max(step, 1)

    logger.info("PDF: built %d chunks (dropped %d noisy) from %d pages",
                kept, dropped, len(pages))
    return chunks


def chunk_all(raw_docs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_docs = list(raw_docs)
    rows = [d for d in raw_docs if d.get("type") == "election_row"]
    pages = [d for d in raw_docs if d.get("type") == "pdf_page"]

    election_chunks = chunk_election(rows)
    pdf_chunks = chunk_pdf(pages)

    all_chunks = election_chunks + pdf_chunks
    logger.info("Total chunks: %d (election=%d, pdf=%d)",
                len(all_chunks), len(election_chunks), len(pdf_chunks))
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from data_loader import load_all
    docs = load_all()
    chunks = chunk_all(docs)
    print(f"Produced {len(chunks)} chunks")
    print("First election chunk:\n", chunks[0]["text"][:400])