"""Minimal `.env` file loader (standard library only).

Auto-loads key=value pairs from a `.env` file in the current working
directory or the repository root into ``os.environ``. Existing process
environment variables always win (never overridden by default), so
exported vars, CI secrets, and shell overrides keep precedence.

Supported syntax (subset of python-dotenv / docker-compose):
- ``KEY=VALUE`` and ``export KEY=VALUE``
- Single- or double-quoted values (``KEY="a # b"``, ``KEY='x=y'``)
- Full-line comments (``# comment``) and blank lines
- Unquoted trailing comments: ``KEY=value # comment`` -> ``value``
- Empty values: ``KEY=`` -> ``""``

Lookup order (first file wins per key):
1. Explicit ``path`` argument, if given.
2. ``$ASR_ENV_FILE`` or ``$DOTENV_PATH`` (when pointing at a file).
3. ``./.env`` (current working directory).
4. ``<repo-root>/.env`` (parent of this ``src/`` package).

Usage:
    from src.dotenv import load_dotenv
    load_dotenv()  # safe to call multiple times; returns dict of loaded vars
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


def _strip_inline_comment(value: str) -> str:
    """Remove a trailing `` # comment`` from an unquoted value."""
    # Only a hash preceded by whitespace starts a comment, so values
    # like ``abc#def`` or URLs with fragments are preserved.
    idx = value.find(" #")
    if idx != -1:
        return value[:idx].rstrip()
    # A tab can also separate value from comment.
    idx = value.find("\t#")
    if idx != -1:
        return value[:idx].rstrip()
    return value


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        inner = value[1:-1]
        if value[0] == '"':
            # Minimal escape handling for double-quoted strings.
            inner = (
                inner.replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
        return inner
    return _strip_inline_comment(value).strip()


def parse_dotenv(path: Path | str) -> Dict[str, str]:
    """Parse a `.env` file into a dict without touching ``os.environ``."""
    result: Dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError:
        return result
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "a").replace("-", "").isalnum():
            continue
        if not (key[0].isalpha() or key[0] == "_"):
            continue
        if "-" in key:
            continue
        result[key] = _unquote(value.strip())
    return result


def _candidate_files(explicit: Optional[Path | str] = None):
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for env_name in ("ASR_ENV_FILE", "DOTENV_PATH"):
        env_path = (os.environ.get(env_name, "") or "").strip()
        if env_path:
            candidates.append(Path(env_path).expanduser())
    candidates.append(Path.cwd() / ".env")
    candidates.append(REPO_ROOT / ".env")
    # Deduplicate while preserving priority order.
    seen: set[str] = set()
    unique: list[Path] = []
    for cand in candidates:
        try:
            resolved = str(cand.resolve()) if cand.is_absolute() else str((Path.cwd() / cand).resolve())
        except Exception:
            resolved = str(cand)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(cand)
    return unique


def load_dotenv(
    path: Optional[Path | str] = None,
    override: bool = False,
) -> Dict[str, str]:
    """Load `.env` file(s) into ``os.environ``.

    Args:
        path: Optional explicit `.env` file (highest priority).
        override: When True, `.env` values replace existing env vars.
            Default False — real environment always wins.

    Returns:
        Dict of key=value pairs that were applied to ``os.environ``.
    """
    applied: Dict[str, str] = {}
    for cand in _candidate_files(path):
        try:
            if not cand.is_file():
                continue
        except Exception:
            continue
        for key, value in parse_dotenv(cand).items():
            if override or key not in os.environ:
                os.environ[key] = value
                applied[key] = value
            elif key not in applied:
                # Higher-priority file already set it via os.environ;
                # keep first-wins ordering across files.
                pass
    return applied


def find_dotenv() -> Optional[str]:
    """Return the first existing `.env` path in lookup order, or None."""
    for cand in _candidate_files():
        try:
            if cand.is_file():
                return str(cand)
        except Exception:
            continue
    return None


__all__ = ["load_dotenv", "parse_dotenv", "find_dotenv", "REPO_ROOT"]
