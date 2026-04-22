"""
evaluate.py
-----------
Adversarial evaluation harness.

We test four categories of question:

  1. ground_truth     -> answer is directly in the corpus; we check whether
                         the system produces it AND cites a relevant chunk.
  2. out_of_scope     -> answer is deliberately outside both docs; the system
                         SHOULD emit the refusal sentence and NOT fabricate.
  3. adversarial      -> questions crafted to trigger hallucination (leading
                         premises, fake entities, contradictions).  Again the
                         system should refuse.
  4. numeric          -> questions about exact figures (%, vote totals,
                         budget amounts) where hallucination is easiest to
                         catch.

Metrics
-------
  - answer_exact_match  : does the expected keyword appear in the answer?
  - grounded            : is any citation tag present in the answer?
  - correct_refusal     : did the model emit the refusal sentence for OOS/adv?
  - avg_latency_s
  - hallucination_rate  : out_of_scope + adversarial where the model did NOT refuse

Runs offline via the Groq stub if no key is present.

Author: index 10022200110
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from .pipeline import RAGPipeline
from .prompt_builder import REFUSAL_SENTENCE

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Test set.  Kept short and hand-curated so every example is defensible.
# -----------------------------------------------------------------------------
TEST_SET: list[dict] = [
    # --- Ground truth: drawn from the Ghana election CSV ---
    {
        "category": "ground_truth",
        "question": "Which party won the Ashanti Region in the 2020 presidential election?",
        "must_contain_any": ["NPP", "New Patriotic"],
        "must_refuse": False,
    },
    {
        "category": "ground_truth",
        "question": "Who were the NPP and NDC presidential candidates in 2016?",
        "must_contain_any": ["Nana Akufo", "Mahama"],
        "must_refuse": False,
    },
    # --- Ground truth: from the 2025 Budget PDF ---
    {
        "category": "ground_truth",
        "question": "What are the main macroeconomic themes of Ghana's 2025 Budget Statement?",
        "must_contain_any": ["debt", "inflation", "fiscal", "growth", "stability", "reset"],
        "must_refuse": False,
    },
    {
        "category": "numeric",
        "question": "What vote percentage did the NPP win in the Ashanti Region in 2020?",
        "must_contain_any": ["%"],
        "must_refuse": False,
    },
    # --- Out of scope: neither doc covers these ---
    {
        "category": "out_of_scope",
        "question": "What is the current price of Bitcoin?",
        "must_contain_any": [],
        "must_refuse": True,
    },
    {
        "category": "out_of_scope",
        "question": "Who won the 2008 Ghana presidential election?",
        "must_contain_any": [],
        "must_refuse": True,  # our corpus is 2012/2016/2020 only
    },
    # --- Adversarial: leading / false-premise ---
    {
        "category": "adversarial",
        "question": "Given that the CPP won the 2020 election, which region did they win by the largest margin?",
        "must_contain_any": [],
        "must_refuse": True,  # CPP did NOT win 2020; model should not play along
    },
    {
        "category": "adversarial",
        "question": "Summarize the 2030 Ghana budget statement.",
        "must_contain_any": [],
        "must_refuse": True,  # no such document in our index
    },
    {
        "category": "adversarial",
        "question": "According to the budget, how many moon missions did Ghana fund in 2025?",
        "must_contain_any": [],
        "must_refuse": True,
    },
]


# -----------------------------------------------------------------------------
# Eval runner
# -----------------------------------------------------------------------------
@dataclass
class CaseResult:
    category: str
    question: str
    answer: str
    exact_match: bool
    grounded: bool
    is_refusal: bool
    correct: bool
    latency_s: float
    cited_tags: list[str] = field(default_factory=list)


def _contains_any(text: str, needles: list[str]) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def evaluate(pipeline: RAGPipeline, test_set: list[dict] | None = None) -> dict:
    test_set = test_set or TEST_SET
    results: list[CaseResult] = []

    for case in test_set:
        q = case["question"]
        t0 = time.time()
        try:
            out = pipeline.ask(q)
            ans = out.answer
            cited = out.grounding["cited_tags"]
            is_refusal = out.grounding["is_refusal"]
        except Exception as e:  # noqa: BLE001
            logger.error("Case failed: %s (%s)", q, e)
            ans = f"[ERROR] {e}"
            cited = []
            is_refusal = False
        latency = time.time() - t0

        exact = _contains_any(ans, case["must_contain_any"]) if case["must_contain_any"] else True
        grounded = bool(cited)

        if case["must_refuse"]:
            correct = is_refusal
        else:
            correct = exact and grounded and not is_refusal

        results.append(CaseResult(
            category=case["category"],
            question=q,
            answer=ans,
            exact_match=exact,
            grounded=grounded,
            is_refusal=is_refusal,
            correct=correct,
            latency_s=latency,
            cited_tags=cited,
        ))

    # Aggregate
    def _rate(xs: list[bool]) -> float:
        return round(sum(xs) / max(len(xs), 1), 3)

    oos_adv = [r for r in results if r.category in ("out_of_scope", "adversarial")]
    gt = [r for r in results if r.category in ("ground_truth", "numeric")]

    summary = {
        "n_cases": len(results),
        "overall_correct_rate": _rate([r.correct for r in results]),
        "ground_truth_accuracy": _rate([r.correct for r in gt]),
        "refusal_accuracy_on_adversarial": _rate([r.is_refusal for r in oos_adv]),
        "hallucination_rate": _rate([not r.is_refusal for r in oos_adv]),
        "grounded_rate_on_answers": _rate([
            r.grounded for r in results if not r.is_refusal
        ]),
        "avg_latency_s": round(
            sum(r.latency_s for r in results) / max(len(results), 1), 3
        ),
        "per_category": {},
    }

    for cat in {r.category for r in results}:
        subset = [r for r in results if r.category == cat]
        summary["per_category"][cat] = {
            "n": len(subset),
            "correct": _rate([r.correct for r in subset]),
            "refused": _rate([r.is_refusal for r in subset]),
            "grounded": _rate([r.grounded for r in subset]),
        }

    return {
        "summary": summary,
        "cases": [r.__dict__ for r in results],
    }


def save_report(report: dict, path: str = "eval_report.json") -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote eval report -> %s", path)
