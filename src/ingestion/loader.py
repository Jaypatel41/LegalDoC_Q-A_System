"""Load raw documents from data/seed/<store>/ . Supports .txt, .md and .pdf."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from ..config import CFG


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:  # pragma: no cover
        print(f"[loader] failed to read {path.name}: {e}")
        return ""


def load_store(store: str) -> List[Tuple[str, str]]:
    """Return [(filename, raw_text), ...] for one store."""
    root = Path(CFG["paths"]["seed_dir"]) / store
    out: List[Tuple[str, str]] = []
    if not root.exists():
        return out
    for path in sorted(root.iterdir()):
        if path.suffix.lower() in {".txt", ".md"}:
            out.append((path.name, path.read_text(encoding="utf-8", errors="ignore")))
        elif path.suffix.lower() == ".pdf":
            out.append((path.name, _read_pdf(path)))
    return out


def load_all() -> Dict[str, List[Tuple[str, str]]]:
    return {s["name"]: load_store(s["name"]) for s in CFG["stores"]}
