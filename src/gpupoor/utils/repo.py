"""Repository path helpers."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

_PYPROJECT_NAME_RE = re.compile(r"""^\s*name\s*=\s*["']gpupoor["']\s*$""", re.MULTILINE)


def _looks_like_repo_root(path: Path) -> bool:
    """Identify the repo root by its project identity, not docs.

    Checking for doc files like design.md breaks every time docs are
    renamed or moved. The only durable identity is the pyproject that
    declares this package (name = "gpupoor") plus the src/gpupoor tree.
    """
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return False
    if not (path / "src" / "gpupoor").is_dir():
        return False
    try:
        head = pyproject.read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError:
        return False
    return _PYPROJECT_NAME_RE.search(head) is not None


def _iter_repo_candidates() -> list[Path]:
    """Fallback search roots when GPUPOOR_ROOT is not set.

    1. This file's ancestry - the package that is actually running. If
       gpupoor is installed editable, this points at the authoritative
       checkout regardless of where the CLI was invoked.
    2. cwd - last-resort fallback for non-editable installs that still
       happen to sit inside a gpupoor checkout.

    cwd was previously first, which allowed a stray pyproject + src/gpupoor
    in the working directory to hijack path resolution even when the
    running package came from elsewhere.
    """
    return [Path(__file__).resolve().parent, Path.cwd()]


def _search_for_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        resolved = candidate.resolve()
        if _looks_like_repo_root(resolved):
            return resolved
    return None


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Locate the repository root without hard-coded parent counts.

    GPUPOOR_ROOT is treated as strict: if the operator set it, we trust
    it and refuse to silently fall back to auto-discovery on a mismatch.
    """
    env_root = os.environ.get("GPUPOOR_ROOT")
    if env_root:
        start = Path(env_root).expanduser()
        found = _search_for_root(start)
        if found is not None:
            return found
        raise RuntimeError(
            f"GPUPOOR_ROOT={env_root!r} is not a gpupoor checkout "
            "(no pyproject.toml declaring name = 'gpupoor' and sibling src/gpupoor)"
        )
    for start in _iter_repo_candidates():
        found = _search_for_root(start)
        if found is not None:
            return found
    raise RuntimeError("Could not locate the gpupoor repo root; set GPUPOOR_ROOT if running outside the repo")


def repo_path(*parts: str) -> Path:
    """Return a path inside the repository root."""
    return repo_root().joinpath(*parts)
