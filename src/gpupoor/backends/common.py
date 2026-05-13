"""Shared utilities for remote training backends."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback (Windows)
    fcntl = None  # type: ignore[assignment]

from gpupoor.config import (
    DEFAULT_HF_DATASET_REPO,
    DEFAULT_HF_PRETOKENIZED_DATASET_FILENAME,
    DEFAULT_REMOTE_IMAGE_TAG,
    BackendConfig,
    RunConfig,
)
from gpupoor.utils import repo_path
from gpupoor.utils.http import http_ok  # re-export for backend consumers
from gpupoor.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

_DEFAULT_REMOTE_IMAGE_TAG = DEFAULT_REMOTE_IMAGE_TAG
_DEFAULT_HF_DATASET_REPO = DEFAULT_HF_DATASET_REPO
_DEFAULT_HF_PRETOKENIZED_DATASET_FILENAME = DEFAULT_HF_PRETOKENIZED_DATASET_FILENAME

__all__ = [
    "http_ok",
    "cached_remote_image_metadata_path",
    "read_required_secret",
    "git_short_sha",
    "git_has_tracked_changes",
    "read_cached_remote_image_metadata",
    "read_cached_remote_image_tag",
    "verify_mlflow",
    "remote_image_tag",
    "remote_worker_env",
    "track_run",
    "kill_tunnel",
]


def cached_remote_image_metadata_path() -> Path:
    return repo_path(".tmp", "remote-image-tag.json")


def read_required_secret(filename: str) -> str:
    secret_file = repo_path(filename)
    if not secret_file.is_file():
        raise FileNotFoundError(f"Required secret file missing: {secret_file}")
    return secret_file.read_text(encoding="utf-8").strip()


def git_short_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_path()), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def git_has_tracked_changes() -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_path()), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def read_cached_remote_image_metadata(settings: dict[str, str]) -> dict[str, object] | None:
    metadata_path = cached_remote_image_metadata_path()
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("vcr_image_base") != settings.get("VCR_IMAGE_BASE"):
        return None
    return payload


def read_cached_remote_image_tag(settings: dict[str, str]) -> str | None:
    payload = read_cached_remote_image_metadata(settings)
    if payload is None:
        return None
    image_tag = payload.get("image_tag")
    if not isinstance(image_tag, str) or not image_tag:
        return None
    return image_tag


def verify_mlflow(health_url: str, *, timeout_seconds: int) -> None:
    if not http_ok(health_url, timeout_seconds=timeout_seconds):
        raise RuntimeError(f"MLflow is not responding at {health_url}")


def remote_image_tag(
    backend: BackendConfig,
    *,
    skip_build: bool,
    dry_run: bool,
    settings: dict[str, str],
    cached_tag: str | None = None,
) -> str:
    if dry_run and not skip_build:
        return "dryrun0"
    if skip_build:
        return backend.remote_image_tag or cached_tag or settings.get("REMOTE_IMAGE_TAG", _DEFAULT_REMOTE_IMAGE_TAG)
    return git_short_sha()


def remote_worker_env(
    config: RunConfig,
    settings: Mapping[str, str],
    *,
    run_config_b64: str,
    profile: str,
    out_dir: str,
    hf_token: str = "",
    connector_env: Mapping[str, str] | None = None,
    mlflow_tracking_uri: str = "",
    hf_dataset_filename_default: str | None = None,
) -> dict[str, str]:
    """Build the env contract shared by remote and local-emulator workers."""
    env = dict(connector_env or {})
    env.pop("GPUPOOR_RUN_CONFIG", None)
    hf_dataset_repo = settings.get("HF_DATASET_REPO", _DEFAULT_HF_DATASET_REPO)
    injected = {
        "GPUPOOR_RUN_CONFIG_B64": run_config_b64,
        "VERDA_PROFILE": profile,
        "REMOTE_RUN_NAME": config.name,
        "OUT_DIR": out_dir,
        "HF_TOKEN": hf_token,
        "HF_DATASET_REPO": hf_dataset_repo,
        "HF_DATASET_FILENAME": settings.get(
            "HF_DATASET_FILENAME",
            hf_dataset_filename_default or Path(config.recipe.dataset_path).name,
        ),
        "HF_PRETOKENIZED_DATASET_REPO": settings.get(
            "HF_PRETOKENIZED_DATASET_REPO",
            hf_dataset_repo,
        ),
        "HF_PRETOKENIZED_DATASET_FILENAME": settings.get(
            "HF_PRETOKENIZED_DATASET_FILENAME",
            _DEFAULT_HF_PRETOKENIZED_DATASET_FILENAME,
        ),
        "R2_TOKENIZED_DATASET_URI": settings.get(
            "R2_TOKENIZED_DATASET_URI",
            config.remote.r2_tokenized_dataset_uri,
        ),
        "R2_TOKENIZED_DATASET_MAX_FILES": settings.get(
            "R2_TOKENIZED_DATASET_MAX_FILES",
            str(config.remote.r2_tokenized_dataset_max_files),
        ),
        "R2_TOKENIZED_DATASET_DIR": settings.get("R2_TOKENIZED_DATASET_DIR", config.remote.r2_tokenized_dataset_dir),
        "R2_TOKENIZER_URI": settings.get("R2_TOKENIZER_URI", config.remote.r2_tokenizer_uri),
        "R2_TOKENIZER_DIR": settings.get("R2_TOKENIZER_DIR", config.remote.r2_tokenizer_dir),
    }
    env.update({key: value for key, value in injected.items() if value})
    if mlflow_tracking_uri:
        env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    return env


def track_run(run_id: str) -> None:
    """Append a run/pod ID to the .run-ids sidecar file under an advisory lock."""
    if not run_id:
        return
    run_ids_file = repo_path(".run-ids")
    with run_ids_file.open("a", encoding="utf-8") as handle:
        if fcntl is not None and hasattr(fcntl, "flock"):
            fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(f"{run_id}\n")


def kill_tunnel() -> None:
    """Terminate the Cloudflare tunnel process tracked by .cf-tunnel.pid."""
    pid_file = repo_path(".cf-tunnel.pid")
    if not pid_file.is_file():
        return
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        pid = 0
    if pid:
        should_kill = True
        if platform.system() == "Linux":
            comm_path = Path(f"/proc/{pid}/comm")
            try:
                comm = comm_path.read_text(encoding="utf-8").strip()
            except OSError:
                should_kill = False
            else:
                if comm != "cloudflared":
                    log.info("WARN: .cf-tunnel.pid %s is '%s', not cloudflared; skipping kill", pid, comm)
                    should_kill = False
        if should_kill:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
    for suffix in (".cf-tunnel.pid", ".cf-tunnel.url", ".cf-tunnel.log"):
        path = repo_path(suffix)
        if path.exists():
            path.unlink()
