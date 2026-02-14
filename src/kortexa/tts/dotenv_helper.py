from __future__ import annotations

from pathlib import Path
from typing import Optional


def _candidates_for(mode: str) -> list[str]:
    mode = mode.lower()
    if mode in {"dev", "development"}:
        return [
            ".env.development.local",
            ".env.development",
            ".env.dev.local",
            ".env.dev",
            ".env.local",
            ".env",
        ]
    # default to production
    return [
        ".env.production.local",
        ".env.production",
        ".env.prod.local",
        ".env.prod",
        ".env.local",
        ".env",
    ]


def resolve_env_file(mode: str) -> Optional[str]:
    """Return the first existing env file path for the given mode."""
    root = Path.cwd()
    for name in _candidates_for(mode):
        p = root / name
        if p.is_file():
            return str(p)
    return None


def load_env_file(path: str, override: bool = False) -> None:
    """Minimal .env loader: KEY=VALUE per line, ignoring comments."""
    import os

    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("export "):
            s = s[7:].lstrip()
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if (k not in os.environ) or override:
            os.environ[k] = v


def load_for_mode(mode: str, override: bool = False) -> Optional[str]:
    path = resolve_env_file(mode)
    if path:
        load_env_file(path, override=override)
    return path
