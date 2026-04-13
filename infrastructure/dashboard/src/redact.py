"""redact.py — Redact known secret env vars from displayed strings."""

from __future__ import annotations

import os
import re

# Env var names that should never appear in displayed log lines
_SECRET_ENV_NAMES = (
    "DSTACK_SERVER_ADMIN_TOKEN",
    "HF_TOKEN",
    "HF_ACCESS_TOKEN",
    "HUGGINGFACE_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "WANDB_API_KEY",
    "OPENAI_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)

_PLACEHOLDER = "[REDACTED]"


def _build_patterns() -> list[tuple[re.Pattern, str]]:
    patterns = []
    for name in _SECRET_ENV_NAMES:
        val = os.environ.get(name)
        if val and len(val) > 4:
            # Escape the literal value for use in regex
            escaped = re.escape(val)
            patterns.append((re.compile(escaped), _PLACEHOLDER))
    return patterns


# Build once at import; re-call redact() lazily rebuilds if env changes
_PATTERNS: list[tuple[re.Pattern, str]] = []


def redact(text: str) -> str:
    """Redact secret env var values from *text*. Returns the cleaned string."""
    global _PATTERNS
    if not _PATTERNS:
        _PATTERNS = _build_patterns()
    result = text
    for pattern, replacement in _PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def redact_lines(lines: list[str]) -> list[str]:
    return [redact(line) for line in lines]
