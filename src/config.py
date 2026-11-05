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


def apply_overrides(section: dict, overrides: list[str] | None) -> dict:
    """Return a copy of a config section with KEY=VALUE overrides applied.

    Values are read as YAML, so use_pb=false, trend_pct=75, fade_target=vwap work.
    """
    out = dict(section)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set needs KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in out:
            raise ValueError(f"Unknown strategy setting {k!r}. Known: {sorted(out)}")
        out[k] = yaml.safe_load(v)
    return out
