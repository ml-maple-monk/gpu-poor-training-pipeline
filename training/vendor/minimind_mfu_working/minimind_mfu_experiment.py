#!/usr/bin/env python3
"""Compatibility shim for the packaged MiniMind MFU experiment CLI."""

from __future__ import annotations

from minimind_local.experiment_cli import *  # noqa: F403
from minimind_local.experiment_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
