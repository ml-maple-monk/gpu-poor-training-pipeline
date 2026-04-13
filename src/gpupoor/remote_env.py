"""Shared remote environment helpers."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

from gpupoor.paths import repo_path


DEFAULT_VCR_IMAGE_BASE = "vccr.io/f53909d3-a071-4826-8635-a62417ffc867/verda-minimind"


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file."""
    data: dict[str, str] = {}
    if not path.is_file():
        return data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        data[key.strip()] = value.strip().strip("'\"")
    return data


def load_remote_settings() -> dict[str, str]:
    settings = parse_env_file(repo_path(".env.remote"))
    for key, value in os.environ.items():
        settings[key] = value
    settings.setdefault("VCR_IMAGE_BASE", DEFAULT_VCR_IMAGE_BASE)
    settings.setdefault("VCR_LOGIN_REGISTRY", settings["VCR_IMAGE_BASE"].rsplit("/", 1)[0])
    return settings


def require_remote_settings(settings: dict[str, str]) -> None:
    missing = [key for key in ("VCR_USERNAME", "VCR_PASSWORD") if not settings.get(key)]
    if missing:
        missing_display = ", ".join(missing)
        raise RuntimeError(
            f"Missing remote registry settings: {missing_display}. "
            "Provide them via env vars or .env.remote."
        )


def find_dstack_bin() -> str:
    candidates = [
        os.environ.get("DSTACK_BIN"),
        str(Path.home() / ".dstack-cli-venv" / "bin" / "dstack"),
        shutil.which("dstack"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if not os.access(candidate, os.X_OK):
            continue
        result = subprocess.run([candidate, "--version"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            return candidate
    raise RuntimeError("No working dstack CLI found")
