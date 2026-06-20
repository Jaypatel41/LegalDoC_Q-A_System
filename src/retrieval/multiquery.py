"""MultiQuery expansion — the accuracy multiplier.

For each user query, the LLM generates N rephrased variants covering different
legal framings (literal term, procedural angle, remedy/penalty angle, precedent
angle). Legal language has extreme synonym variance ("penalty for breach" vs
"damages for non-performance" sit far apart in embedding space but retrieve
complementary chunks), so expansion typically lifts context recall +12-15%.
"""
from __future__ import annotations

from typing import List

from ..config import CFG
from ..llm import get_llm

_PROMPT = """You are a legal search assistant. Generate {n} alternative rephrasings of \
the user's legal question. Each rephrasing must explore a different framing:
1. Literal / terminology framing
2. Procedural framing (how the process works)
3. Remedy or penalty framing
4. Precedent / case-law framing

Return ONLY the {n} rephrasings, one per line, no numbering, no extra text.

User question: {query}"""


def expand(query: str) -> List[str]:
    cfg = CFG["retrieval"]["multiquery"]
    if not cfg["enabled"]:
        return [query]
    n = int(cfg["n_variants"])
    llm = get_llm()
    raw = llm.complete(_PROMPT.format(n=n, query=query), temperature=0.3, max_tokens=300)
    variants = [ln.strip(" -•\t") for ln in raw.splitlines() if ln.strip()]
    variants = [v for v in variants if len(v) > 5][:n]
    # always include the original query
    out = [query] + variants
    # de-dup preserving order
    seen, uniq = set(), []
    for q in out:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(q)
    return uniq
