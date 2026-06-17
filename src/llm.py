"""Provider-agnostic LLM client.

Supports Anthropic (Claude), OpenAI, and any OpenAI-compatible local endpoint
(Ollama / llama.cpp). A deterministic stub is used when USE_STUB_LLM=1 or no key
is configured, so the whole pipeline + demo run offline with no credentials.
"""
from __future__ import annotations

import os
from typing import List

from .config import env


class LLM:
    def __init__(self) -> None:
        self.provider = (env("LLM_PROVIDER", "anthropic") or "anthropic").lower()
        self.stub = env("USE_STUB_LLM", "0") == "1"
        self._client = None
        if not self.stub:
            self._init_client()

    # ------------------------------------------------------------------ setup
    def _init_client(self) -> None:
        try:
            if self.provider == "anthropic":
                key = env("ANTHROPIC_API_KEY")
                if not key or key.startswith("sk-ant-..."):
                    self.stub = True
                    return
                import anthropic
                self._client = anthropic.Anthropic(api_key=key)
                self.model = env("ANTHROPIC_MODEL", "claude-haiku-4-5")
            elif self.provider == "openai":
                key = env("OPENAI_API_KEY")
                if not key or key.startswith("sk-..."):
                    self.stub = True
                    return
                from openai import OpenAI
                self._client = OpenAI(api_key=key)
                self.model = env("OPENAI_MODEL", "gpt-4o-mini")
            elif self.provider == "local":
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=env("LOCAL_BASE_URL", "http://localhost:11434/v1"),
                    api_key=env("LOCAL_API_KEY", "local"),  # Groq/OpenAI-compatible key
                )
                self.model = env("LOCAL_MODEL", "llama3:8b")
            else:
                self.stub = True
        except Exception as e:  # pragma: no cover - missing sdk etc.
            print(f"[llm] falling back to stub ({e})")
            self.stub = True

    # --------------------------------------------------------------- generate
    def complete(self, prompt: str, system: str = "", temperature: float = 0.0,
                 max_tokens: int = 1024) -> str:
        if self.stub:
            return self._stub(prompt, system)
        if self.provider == "anthropic":
            msg = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, temperature=temperature,
                system=system or "You are a precise assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        # openai / local share the chat-completions API
        resp = self._client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system or "You are a precise assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content

    # ------------------------------------------------------------------ stub
    @staticmethod
    def _stub(prompt: str, system: str) -> str:
        """Deterministic, grounded-ish answer for offline runs.

        For MultiQuery it returns newline-separated rephrasings; for generation
        it echoes the first context snippet so the pipeline stays faithful.
        """
        low = prompt.lower()
        if "rephrase" in low or "alternative" in low or "variant" in low:
            base = prompt.strip().split("\n")[-1][:120]
            return "\n".join([
                f"What does the law say about {base}?",
                f"Procedure and process regarding {base}",
                f"Remedy or penalty related to {base}",
                f"Relevant precedent on {base}",
            ])
        # generation: pull the first context block as a grounded answer
        if "context:" in low:
            ctx = prompt.split("Context:", 1)[1]
            snippet = " ".join(ctx.split())[:400]
            return f"Based on the retrieved sources: {snippet}"
        return "[stub-llm] no provider configured; set keys in .env to enable real generation."


_LLM: LLM | None = None


def get_llm() -> LLM:
    global _LLM
    if _LLM is None:
        _LLM = LLM()
    return _LLM
