"""Repository path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the checked-out repository root."""
    explicit = os.environ.get("GPUPOOR_REPO_ROOT")
    if explicit:
        return Path(explicit).resolve()

    candidates = [Path.cwd(), Path(__file__).resolve()]
    for start in candidates:
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "training").is_dir():
                return candidate
    raise RuntimeError("Could not locate the gpupoor repository root")


def repo_path(*parts: str) -> Path:
    """Return a path inside the repo root."""
    return repo_root().joinpath(*parts)

