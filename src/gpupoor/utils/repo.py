"""Repository path helpers."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path


def _looks_like_repo_root(path: Path) -> bool:
    return (
        (path / "pyproject.toml").is_file()
        and (path / "src" / "gpupoor").is_dir()
        and (path / "design.md").is_file()
    )


def _iter_repo_candidates() -> list[Path]:
    env_root = os.environ.get("GPUPOOR_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend([Path.cwd(), Path(__file__).resolve().parent])
    return candidates


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Locate the repository root without hard-coded parent counts."""
    for start in _iter_repo_candidates():
        for candidate in (start, *start.parents):
            resolved = candidate.resolve()
            if _looks_like_repo_root(resolved):
                return resolved
    raise RuntimeError(
        "Could not locate the gpupoor repo root; set GPUPOOR_ROOT if running outside the repo"
    )


def repo_path(*parts: str) -> Path:
    """Return a path inside the repository root."""
    return repo_root().joinpath(*parts)
