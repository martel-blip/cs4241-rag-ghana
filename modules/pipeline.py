"""
pipeline.py
-----------
End-to-end RAG pipeline.

Stages (each timed + logged):
  1. load      -> data_loader.load_all()
  2. chunk     -> chunker.chunk_all()
  3. embed     -> Embedder.embed_chunks_with_cache()
  4. index     -> NumpyVectorStore.build()
  5. retrieve  -> HybridRetriever.retrieve()
  6. prompt    -> PromptBuilder.build()
  7. generate  -> GroqClient.generate()
  8. verify    -> prompt_builder.verify_citations()

Calling `RAGPipeline().setup()` runs stages 1-4 once.  Afterwards
`ask(question)` runs stages 5-8 per query and returns a structured result.

Author: index 10022200110
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .embedder import Embedder
from .llm_client import GroqClient, LLMResponse
from .prompt_builder import PromptBuilder, verify_citations
from .retriever import HybridRetriever, RetrievalResult
from .vector_store import NumpyVectorStore

logger = logging.getLogger(__name__)


@dataclass
class StageTimings:
    """Per-stage wall-clock timings (seconds)."""
    retrieve: float = 0.0
    prompt: float = 0.0
    generate: float = 0.0
    verify: float = 0.0
    total: float = 0.0


@dataclass
class RAGAnswer:
    """Everything the UI needs to render one interaction."""
    question: str
    answer: str
    citations: list[dict]           # the context blocks that were handed to the LLM
    retrieval: list[RetrievalResult]
    timings: StageTimings
    grounding: dict                  # output of verify_citations()
    llm: LLMResponse
    debug: dict = field(default_factory=dict)


class RAGPipeline:
    """
    Orchestrator.  Not thread-safe (Streamlit runs one session at a time).
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        llm: GroqClient | None = None,
        prompt_builder: PromptBuilder | None = None,
        top_k: int = 5,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.llm = llm or GroqClient()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.top_k = top_k

        self.vector_store: NumpyVectorStore | None = None
        self.retriever: HybridRetriever | None = None
        self.chunks: list[dict] = []

    # -------- one-time setup --------
    def setup(self, chunks: list[dict]) -> dict:
        """
        Embed + index a list of chunks.  Expects the output of chunker.chunk_all():
        each chunk is a dict with at least 'text', 'chunk_id', 'source'.
        Returns a dict of stage timings for the caller to log.
        """
        if not chunks:
            raise ValueError("No chunks provided to pipeline.setup()")

        timings: dict[str, float] = {}

        t0 = time.time()
        vectors, _ = self.embedder.embed_chunks_with_cache(chunks, cache_key="chunks_v1")
        timings["embed"] = time.time() - t0
        logger.info("[setup] embed stage: %.2fs (%d chunks)", timings["embed"], len(chunks))

        t0 = time.time()
        self.vector_store = NumpyVectorStore()
        self.vector_store.build(vectors, chunks)
        timings["index"] = time.time() - t0
        logger.info("[setup] index stage: %.2fs", timings["index"])

        t0 = time.time()
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            chunks=chunks,
            k_final=self.top_k,
        )
        timings["retriever_build"] = time.time() - t0
        logger.info("[setup] retriever build: %.2fs", timings["retriever_build"])

        self.chunks = chunks
        return timings

    # -------- per-query --------
    def ask(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """
        Run retrieval -> prompt -> generate -> verify for one question.
        """
        if self.retriever is None:
            raise RuntimeError("Pipeline not set up. Call setup(chunks) first.")

        timings = StageTimings()
        total_t0 = time.time()

        # --- retrieve ---
        t0 = time.time()
        hits = self.retriever.retrieve(question, top_k=top_k or self.top_k)
        timings.retrieve = time.time() - t0
        logger.info("[ask] retrieve: %.3fs -> %d hits", timings.retrieve, len(hits))

        # --- prompt ---
        t0 = time.time()
        built = self.prompt_builder.build(question, hits)
        timings.prompt = time.time() - t0
        logger.info("[ask] prompt build: %.3fs (%d context blocks, trimmed=%d)",
                    timings.prompt, len(built.context_blocks), built.trimmed)

        # --- generate ---
        t0 = time.time()
        llm_resp = self.llm.generate(built.system, built.user)
        timings.generate = time.time() - t0
        logger.info("[ask] generate: %.2fs (model=%s, stub=%s)",
                    timings.generate, llm_resp.model, llm_resp.stub)

        # --- verify / ground-check ---
        t0 = time.time()
        grounding = verify_citations(llm_resp.content, built.context_blocks)
        timings.verify = time.time() - t0
        timings.total = time.time() - total_t0

        logger.info(
            "[ask] verify: %.3fs | cited=%s | suspicious=%s | total=%.2fs",
            timings.verify, grounding["cited_tags"],
            grounding["suspicious"], timings.total,
        )

        return RAGAnswer(
            question=question,
            answer=llm_resp.content,
            citations=built.context_blocks,
            retrieval=hits,
            timings=timings,
            grounding=grounding,
            llm=llm_resp,
            debug={
                "trimmed_chunks": built.trimmed,
                "prompt_chars": len(built.user),
            },
        )
