"""errors.py — SourceStatus enum for collector health reporting."""

from __future__ import annotations

from enum import Enum


class SourceStatus(Enum):
    OK = "ok"
    STALE = "stale"
    ERROR = "error"
    UNKNOWN = "unknown"
