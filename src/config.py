"""Load config.yaml into a plain dict.

Paths in the config are relative to the project root, so we resolve them
against the folder containing config.yaml. That way scripts work no matter
which directory you run them from.
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = path.resolve().parent  # remember where relative paths start
    return cfg


def resolve(cfg: dict, relative_path: str) -> Path:
    """Turn a path from the config into an absolute path."""
    p = Path(relative_path)
    return p if p.is_absolute() else cfg["_root"] / p
