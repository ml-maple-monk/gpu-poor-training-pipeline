"""Helpers for loading secrets and environment variables from sidecar files."""

from __future__ import annotations

import os
from pathlib import Path


def load_hf_token(token_file: Path) -> dict[str, str]:
    """Return an ``env`` overlay dict carrying ``HF_TOKEN`` if available.

    Resolution order matches the legacy duplicates in ops.smoke and
    services.emulator:

    1. If ``HF_TOKEN`` is already set in the parent process environment,
       return an empty dict so the child inherits it unchanged (no override).
    2. Otherwise, if *token_file* exists, read it, strip whitespace, and
       return ``{"HF_TOKEN": <value>}``.
    3. Otherwise return an empty dict. The caller is responsible for deciding
       whether an absent token is fatal.
    """
    if os.environ.get("HF_TOKEN"):
        return {}
    if token_file.is_file():
        return {"HF_TOKEN": token_file.read_text(encoding="utf-8").strip()}
    return {}
