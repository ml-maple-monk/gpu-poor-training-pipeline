"""Resolve the dashboard's dstack project alias without hardcoding `main`."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

_DEFAULT_DSTACK_PROJECT = "dashboard"


def _config_path(home: str | None = None) -> Path:
    base = Path(home or os.environ.get("HOME", "~")).expanduser()
    return base / ".dstack" / "config.yml"


def _parse_project_names(text: str) -> tuple[str | None, str | None]:
    """Return (default_project_name, first_project_name) from a dstack config.

    Uses ``yaml.safe_load`` instead of hand-rolled line parsing so
    structural YAML quirks (quoting, block scalars, multi-document files,
    BOMs) match what dstack itself sees. Malformed YAML and
    unexpected shapes degrade gracefully to ``(None, None)`` so the
    caller can fall through to the dashboard-local default alias.
    """
    if not text.strip():
        return None, None
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        return None, None
    if not isinstance(document, dict):
        return None, None
    projects = document.get("projects")
    if not isinstance(projects, list):
        return None, None

    default_name: str | None = None
    first_name: str | None = None
    for entry in projects:
        if not isinstance(entry, dict):
            continue
        raw_name = entry.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) else None
        if not name:
            continue
        if first_name is None:
            first_name = name
        if default_name is None and entry.get("default") is True:
            default_name = name
    return default_name, first_name


def infer_dstack_project(*, environ: dict[str, str] | None = None, home: str | None = None) -> str:
    env = environ or os.environ
    explicit = env.get("DSTACK_PROJECT", "").strip()
    if explicit:
        return explicit

    config_path = _config_path(home)
    if config_path.is_file():
        try:
            text = config_path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        default_name, first_name = _parse_project_names(text)
        if default_name:
            return default_name
        if first_name:
            return first_name

    return _DEFAULT_DSTACK_PROJECT
