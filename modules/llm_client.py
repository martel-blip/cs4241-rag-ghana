"""
llm_client.py
-------------
Thin wrapper around the Groq chat API.

We isolate the LLM call here so:
  - the pipeline stays testable offline (use `stub=True`),
  - we can swap providers later without touching downstream code,
  - retries / timeouts live in one place.

API key is read from:
  1. the `api_key` constructor argument (for tests),
  2. `st.secrets["GROQ_API_KEY"]` when running under Streamlit,
  3. the `GROQ_API_KEY` environment variable.

Author: index 10022200110
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Groq's current production model of choice for our use case: fast, cheap,
# reliable instruction following.
DEFAULT_MODEL = "llama-3.1-8b-instant"


@dataclass
class LLMResponse:
    content: str
    model: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    stub: bool = False


class GroqClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,  # low, to discourage invention
        max_tokens: int = 700,
        timeout_s: float = 30.0,
        stub: bool = False,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.stub = stub
        self._api_key = api_key or self._resolve_api_key()
        self._client = None  # lazy

    @staticmethod
    def _resolve_api_key() -> str | None:
        # Streamlit secrets (only if Streamlit is importable and configured)
        try:
            import streamlit as st  # type: ignore
            if "GROQ_API_KEY" in st.secrets:
                return st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
        return os.getenv("GROQ_API_KEY")

    def _ensure_client(self) -> None:
        if self.stub or self._client is not None:
            return
        if not self._api_key:
            raise RuntimeError(
                "No Groq API key found. Set GROQ_API_KEY env var or add it to "
                ".streamlit/secrets.toml as GROQ_API_KEY."
            )
        
        # Set environment variable so Groq can pick it up automatically
        os.environ["GROQ_API_KEY"] = self._api_key
        
        try:
            from groq import Groq  # type: ignore
            # Patch httpx to avoid proxies issue
            try:
                import httpx
                # Monkey patch to prevent proxies from being passed
                original_init = httpx.Client.__init__
                def patched_init(self, **kwargs):
                    kwargs.pop('proxies', None)
                    return original_init(self, **kwargs)
                httpx.Client.__init__ = patched_init
            except Exception:
                pass
            
            # Let Groq read from environment variable
            self._client = Groq()
        except ImportError as e:
            raise ImportError(
                "groq package not installed. Run `pip install groq`."
            ) from e

    # -------- main call --------
    def generate(
        self,
        system: str,
        user: str,
        max_retries: int = 2,
    ) -> LLMResponse:
        if self.stub:
            return self._stub_response(user)

        self._ensure_client()
        t0 = time.time()
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._client.chat.completions.create(  # type: ignore[union-attr]
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                return LLMResponse(
                    content=content.strip(),
                    model=self.model,
                    latency_s=time.time() - t0,
                    prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                    completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                )
            except Exception as e:  # noqa: BLE001 - we want to retry on any transient error
                last_err = e
                logger.warning("Groq call failed (attempt %d/%d): %s",
                               attempt + 1, max_retries + 1, e)
                time.sleep(1.5 * (attempt + 1))
        assert last_err is not None
        raise last_err

    # -------- offline fallback --------
    @staticmethod
    def _stub_response(user: str) -> LLMResponse:
        """
        Deterministic stub used in tests / when GROQ_API_KEY is missing.
        Produces a plausible-looking citation-bearing answer so UI work can
        proceed without burning API tokens.
        """
        # Pick up the first [C#] tag from the prompt so we can cite it.
        import re
        tags = re.findall(r"\[C(\d+)\]", user)
        first = f"[C{tags[0]}]" if tags else ""
        if "[C" not in user:
            answer = "I don't have enough information in the provided sources to answer that."
        else:
            answer = (
                f"[STUB ANSWER — no Groq key set.] Based on the retrieved context, "
                f"the relevant information appears in {first}. "
                f"Connect a real Groq API key to get a grounded model answer."
            )
        return LLMResponse(
            content=answer,
            model="stub",
            latency_s=0.0,
            stub=True,
        )
