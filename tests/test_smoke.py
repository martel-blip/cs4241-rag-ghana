"""
Smoke tests for the RAG modules.  Runnable without network / Groq key.

    python -m unittest tests/test_smoke.py
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from modules.vector_store import NumpyVectorStore          # noqa: E402
from modules.retriever import BM25, tokenize               # noqa: E402
from modules.prompt_builder import PromptBuilder, verify_citations  # noqa: E402
from modules.feedback import FeedbackStore, apply_reputation_boost  # noqa: E402
from modules.retriever import RetrievalResult              # noqa: E402


class VectorStoreTest(unittest.TestCase):
    def test_self_retrieval(self):
        rng = np.random.default_rng(0)
        V = rng.normal(size=(20, 16)).astype(np.float32)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        store = NumpyVectorStore()
        store.build(V, [{"chunk_id": str(i), "source": "x"} for i in range(20)])
        for i in range(20):
            hits = store.search(V[i], top_k=1)
            self.assertEqual(hits[0].index, i)
            self.assertGreater(hits[0].score, 0.999)


class BM25Test(unittest.TestCase):
    def test_basic_ranking(self):
        docs = [
            "The NPP won the Ashanti Region in 2020.",
            "The 2025 Budget focuses on debt sustainability.",
            "Volta Region has historically favored the NDC.",
        ]
        bm = BM25()
        bm.fit(docs)
        top = bm.top_k("Ashanti 2020 NPP", k=3)
        self.assertEqual(top[0][0], 0)

    def test_tokenize_drops_stopwords(self):
        self.assertNotIn("the", tokenize("the quick brown fox"))


class PromptBuilderTest(unittest.TestCase):
    def test_empty_context_path(self):
        pb = PromptBuilder()
        built = pb.build("nonsense?", [])
        self.assertIn("no relevant context was retrieved", built.user)
        self.assertEqual(built.context_blocks, [])

    def test_citations_are_numbered(self):
        pb = PromptBuilder()
        r = RetrievalResult(
            chunk_id="c42", text="Ashanti went NPP in 2020.",
            source="elections.csv", metadata={}, score=0.9,
            rank_sources=["dense"],
        )
        built = pb.build("Who won Ashanti 2020?", [r])
        self.assertIn("[C1]", built.user)
        self.assertEqual(built.context_blocks[0]["tag"], "C1")

    def test_verify_citations_flags_hallucination(self):
        ctx = [{"tag": "C1"}, {"tag": "C2"}]
        # No citation emitted despite context being provided
        v = verify_citations("NPP won.", ctx)
        self.assertTrue(v["suspicious"])
        # Proper citation
        v2 = verify_citations("NPP won [C1].", ctx)
        self.assertFalse(v2["suspicious"])
        # Unknown tag
        v3 = verify_citations("NPP won [C9].", ctx)
        self.assertEqual(v3["unknown_tags"], ["C9"])


class FeedbackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = "test_feedback.json"
        if os.path.exists(self.tmp):
            os.remove(self.tmp)
        self.store = FeedbackStore(path=self.tmp)

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_reputation_bounded(self):
        self.store.record(["c1"], rating=+1)
        self.store.record(["c1"], rating=+1)
        self.assertGreater(self.store.reputation("c1"), 0.0)
        self.assertLessEqual(self.store.reputation("c1"), 1.0)

    def test_boost_resorts(self):
        r1 = RetrievalResult(chunk_id="c1", text="...", source="s", metadata={}, score=0.5)
        r2 = RetrievalResult(chunk_id="c2", text="...", source="s", metadata={}, score=0.6)
        # Give c1 a bunch of positive feedback so it overtakes c2
        for _ in range(20):
            self.store.record(["c1"], rating=+1)
        for _ in range(20):
            self.store.record(["c2"], rating=-1)
        out = apply_reputation_boost([r1, r2], self.store, alpha=0.5)
        self.assertEqual(out[0].chunk_id, "c1")


if __name__ == "__main__":
    unittest.main()
