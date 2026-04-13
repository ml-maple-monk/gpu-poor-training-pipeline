"""Resolve the dashboard's dstack project alias without hardcoding `main`."""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_DSTACK_PROJECT = "dashboard"


def _config_path(home: str | None = None) -> Path:
    base = Path(home or os.environ.get("HOME", "~")).expanduser()
    return base / ".dstack" / "config.yml"


def _parse_project_names(text: str) -> tuple[str | None, str | None]:
    default_name: str | None = None
    first_name: str | None = None
    current_name: str | None = None
    current_is_default = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            if current_name and current_is_default and default_name is None:
                default_name = current_name
            current_name = None
            current_is_default = False
            line = line[2:].strip()
        if line.startswith("name:"):
            current_name = line.partition(":")[2].strip().strip("'\"")
            if current_name and first_name is None:
                first_name = current_name
            if current_name and current_is_default and default_name is None:
                default_name = current_name
            continue
        if line.startswith("default:"):
            value = line.partition(":")[2].strip().lower()
            current_is_default = value == "true"
            if current_name and current_is_default and default_name is None:
                default_name = current_name

    if current_name and current_is_default and default_name is None:
        default_name = current_name
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
