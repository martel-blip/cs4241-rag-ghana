"""
data_loader.py
--------------
Loads raw documents from the two sources:
  1. Ghana_Election_Result.csv  -> one raw doc per row (615 rows)
  2. 2025-Budget-Statement-and-Economic-Policy_v4.pdf -> one raw doc per page (~251 pages)

Each raw doc is a dict with at least: 'text', 'source', 'type', plus source-specific
metadata (year/region/party for CSV rows, page number for PDF pages).

Author: index 10022200110
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "Ghana_Election_Result.csv")
PDF_PATH = os.path.join(DATA_DIR, "2025-Budget-Statement-and-Economic-Policy_v4.pdf")


def _normalize(s) -> str:
    """Normalize whitespace, strip non-breaking spaces."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).replace("\xa0", " ")).strip()


def load_election_csv(path: str = CSV_PATH) -> list[dict[str, Any]]:
    """Turn every CSV row into a raw doc with a natural-language sentence."""
    import pandas as pd

    if not os.path.exists(path):
        raise FileNotFoundError(f"Election CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    docs: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        year = int(row["Year"])
        old_region = _normalize(row["Old Region"])
        new_region = _normalize(row["New Region"])
        code = _normalize(row["Code"])
        candidate = _normalize(row["Candidate"])
        party = _normalize(row["Party"])
        votes = row["Votes"]
        pct = _normalize(row["Votes(%)"])

        try:
            votes_int = int(votes)
            votes_str = f"{votes_int:,}"
        except (TypeError, ValueError):
            votes_str = str(votes)

        text = (
            f"In the {year} Ghana presidential election, in {new_region} "
            f"(formerly {old_region}), candidate {candidate} of the {party} "
            f"party ({code}) received {votes_str} votes ({pct})."
        )

        docs.append({
            "text": text,
            "source": "Ghana_Election_Result.csv",
            "type": "election_row",
            "year": year,
            "old_region": old_region,
            "new_region": new_region,
            "candidate": candidate,
            "party": party,
            "code": code,
            "votes": votes_str,
            "votes_pct": pct,
            "row_index": int(i),
        })

    logger.info("Loaded %d rows from %s", len(docs), path)
    return docs


def load_budget_pdf(path: str = PDF_PATH) -> list[dict[str, Any]]:
    """One raw doc per page. Pages with no extractable text are skipped."""
    from pypdf import PdfReader

    if not os.path.exists(path):
        raise FileNotFoundError(f"Budget PDF not found: {path}")

    reader = PdfReader(path)
    docs: list[dict[str, Any]] = []
    skipped = 0
    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception as e:
            logger.warning("Could not extract page %d: %s", i + 1, e)
            skipped += 1
            continue

        text = _normalize(raw)
        if len(text) < 50:
            skipped += 1
            continue

        docs.append({
            "text": text,
            "source": "2025-Budget-Statement-and-Economic-Policy_v4.pdf",
            "type": "pdf_page",
            "page": i + 1,
        })

    logger.info("Loaded %d pages from %s (skipped %d)", len(docs), path, skipped)
    return docs


def load_all() -> list[dict[str, Any]]:
    """Return all raw docs from both sources."""
    csv_docs = load_election_csv()
    pdf_docs = load_budget_pdf()
    all_docs = csv_docs + pdf_docs
    logger.info("Total raw docs: %d (CSV=%d, PDF=%d)",
                len(all_docs), len(csv_docs), len(pdf_docs))
    return all_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    docs = load_all()
    print(f"Loaded {len(docs)} raw docs.")
    print("First CSV row:", docs[0]["text"])