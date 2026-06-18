"""Store-aware chunking. The chunking strategy is the first measurable design
decision in the pipeline:
  - statute  -> split by "Section N" (preserves legal structure)
  - case_law -> split by paragraph with token overlap (argument blocks)
  - contract -> split by clause heading (clause-level retrieval beats fixed-size)
"""
from __future__ import annotations

import re
from typing import List

from ..config import CFG
from ..schema import Chunk

# rough token ~= word; overlap expressed in tokens in config
_SECTION_RE = re.compile(r"(?im)^\s*(section\s+\d+[A-Za-z]?\.?.*)$")
_CLAUSE_RE = re.compile(r"(?im)^\s*((?:clause\s+)?\d+(?:\.\d+)*\s*[\.\):-].*)$")


def _overlap_words(prev: str, n: int) -> str:
    if n <= 0 or not prev:
        return ""
    return " ".join(prev.split()[-n:])


def _chunk_by_regex(text: str, regex: re.Pattern, store: str, source: str,
                    overlap: int) -> List[Chunk]:
    matches = list(regex.finditer(text))
    chunks: List[Chunk] = []
    if not matches:
        return _chunk_paragraph(text, store, source, overlap)
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        title = m.group(1).strip()[:80]
        prefix = _overlap_words(chunks[-1].text, overlap) if chunks else ""
        full = (prefix + " " + body).strip() if prefix else body
        chunks.append(Chunk(id=f"{store}:{source}:{i}", text=full, store=store,
                            source=source, title=title))
    return chunks


def _chunk_paragraph(text: str, store: str, source: str, overlap: int) -> List[Chunk]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[Chunk] = []
    for i, para in enumerate(paras):
        prefix = _overlap_words(paras[i - 1], overlap) if i > 0 else ""
        full = (prefix + " " + para).strip() if prefix else para
        title = para.split("\n", 1)[0][:80]
        chunks.append(Chunk(id=f"{store}:{source}:{i}", text=full, store=store,
                            source=source, title=title))
    return chunks


def chunk_document(store: str, source: str, text: str) -> List[Chunk]:
    scfg = next(s for s in CFG["stores"] if s["name"] == store)
    strategy = scfg["chunk_strategy"]
    overlap = int(scfg.get("overlap_tokens", 0))
    if strategy == "section":
        return _chunk_by_regex(text, _SECTION_RE, store, source, overlap)
    if strategy == "clause":
        return _chunk_by_regex(text, _CLAUSE_RE, store, source, overlap)
    return _chunk_paragraph(text, store, source, overlap)
