"""Central config loader. Reads config/config.yaml + .env once, exposes a dict-ish object."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv optional
    pass

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"


class Config(dict):
    """Dict with attribute access and dotted-path .get('a.b.c')."""

    __getattr__ = dict.get  # type: ignore[assignment]

    def path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


@lru_cache(maxsize=1)
def get_config() -> Config:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _wrap(raw)
    # resolve paths relative to project root
    for key, val in cfg["paths"].items():
        cfg["paths"][key] = str((ROOT / val).resolve())
    return cfg


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


CFG = get_config()
