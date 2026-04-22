"""modules package for the CS4241 RAG project."""

from .embedder import Embedder
from .evaluate import evaluate, save_report, TEST_SET
from .feedback import FeedbackStore, apply_reputation_boost
from .llm_client import GroqClient, LLMResponse
from .pipeline import RAGAnswer, RAGPipeline, StageTimings
from .prompt_builder import PromptBuilder, verify_citations, REFUSAL_SENTENCE
from .retriever import BM25, HybridRetriever, RetrievalResult
from .vector_store import NumpyVectorStore, SearchHit

__all__ = [
    "Embedder",
    "NumpyVectorStore",
    "SearchHit",
    "BM25",
    "HybridRetriever",
    "RetrievalResult",
    "PromptBuilder",
    "verify_citations",
    "REFUSAL_SENTENCE",
    "GroqClient",
    "LLMResponse",
    "RAGPipeline",
    "RAGAnswer",
    "StageTimings",
    "FeedbackStore",
    "apply_reputation_boost",
    "evaluate",
    "save_report",
    "TEST_SET",
]
