"""Operational helpers for doctoring, smoke tests, and secret management."""

from gpupoor.ops.doctor import (
    _resolve_max_clock_skew,
    check_doc_anchors,
    fix_wsl_clock,
    run_preflight,
)
from gpupoor.ops.secrets import detect_secret_leaks, leak_scan, parse_secrets, parse_secrets_payload
from gpupoor.ops.smoke import run_smoke

# `_resolve_max_clock_skew` stays in __all__ despite the leading underscore:
# tests/test_maintenance.py accesses it as ``ops._resolve_max_clock_skew`` to
# cover the private fallback/override precedence. Remove from __all__ once that
# test moves onto the public surface (doctor(max_clock_skew_seconds=...) etc.).
__all__ = [
    "_resolve_max_clock_skew",
    "check_doc_anchors",
    "detect_secret_leaks",
    "fix_wsl_clock",
    "leak_scan",
    "parse_secrets",
    "parse_secrets_payload",
    "run_preflight",
    "run_smoke",
]
