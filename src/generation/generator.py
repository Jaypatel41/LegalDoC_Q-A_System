"""Grounded answer generation. The prompt forces the model to answer ONLY from
the retrieved context and to cite sources; a stricter variant is used on the
faithfulness-triggered regeneration pass."""
from __future__ import annotations

from typing import List

from ..config import CFG
from ..llm import get_llm
from ..schema import Retrieved

_SYSTEM = ("You are an Indian-law legal assistant. Answer strictly and only from "
           "the provided context. Cite the source/section for every claim. If the "
           "context does not contain the answer, say so explicitly. Do not invent "
           "sections, case names, or numbers.")

_STRICT_SYSTEM = _SYSTEM + (" CRITICAL: every sentence must be directly supported "
                            "by the context. Omit anything not explicitly stated.")

_PROMPT = """Answer the legal question using only the context below.

Context:
{context}

Question: {query}

Answer (cite sources):"""


def _format_context(results: List[Retrieved], max_chars: int) -> str:
    blocks, total = [], 0
    for r in results:
        tag = f"[{r.chunk.store}:{r.chunk.title or r.chunk.source}]"
        block = f"{tag} {r.chunk.text}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def generate(query: str, results: List[Retrieved], strict: bool = False) -> str:
    gcfg = CFG["generation"]
    if not results:
        return gcfg["insufficient_msg"]
    context = _format_context(results, int(gcfg["max_context_chars"]))
    prompt = _PROMPT.format(context=context, query=query)
    llm = get_llm()
    return llm.complete(prompt, system=_STRICT_SYSTEM if strict else _SYSTEM,
                        temperature=float(gcfg["temperature"]), max_tokens=800)


def citations(results: List[Retrieved]) -> List[str]:
    seen, out = set(), []
    for r in results:
        c = f"{r.chunk.store}: {r.chunk.title or r.chunk.source}"
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out
