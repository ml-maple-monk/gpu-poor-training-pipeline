# WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION WITHOUT EXPLICIT USER APPROVAL.
# All concrete code must follow this abstraction; do not modify this code unless explicitly asked.
"""Portable MLflow runtime contract for remote training workers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

PORTABLE_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_BUNDLE_DIR = "/workspace/mlflow-bundle"
ARTIFACT_MODE_PORTABLE = "portable"
_DEFAULT_SYNC_PREFIX = "mlflow-bundles"


@dataclass(frozen=True, slots=True)
class PortableMlflowRuntime:
    """Environment contract for one self-contained MLflow bundle."""

    run_name: str
    bundle_dir: str = DEFAULT_BUNDLE_DIR
    tracking_uri: str = PORTABLE_TRACKING_URI
    sync_uri: str = ""

    def to_env(self) -> dict[str, str]:
        env = {
            "GPUPOOR_PORTABLE_MLFLOW": "1",
            "MLFLOW_TRACKING_URI": self.tracking_uri,
            "MLFLOW_BUNDLE_DIR": self.bundle_dir,
            "MLFLOW_BUNDLE_RUN_NAME": self.run_name,
            "GPUPOOR_CONNECTOR_ARTIFACT_MODE": ARTIFACT_MODE_PORTABLE,
        }
        if self.sync_uri:
            env["MLFLOW_BUNDLE_SYNC_URI"] = self.sync_uri
        return env


def sanitize_bundle_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-")
    return cleaned or "gpupoor-run"


def bundle_sync_uri(artifact_destination: str, run_name: str) -> str:
    """Derive a durable R2/S3 bundle destination from the configured artifact root."""
    parsed = urlparse(artifact_destination)
    if parsed.scheme != "s3" or not parsed.netloc:
        return ""
    prefix = parsed.path.strip("/")
    parts = [part for part in (prefix, _DEFAULT_SYNC_PREFIX, sanitize_bundle_name(run_name)) if part]
    return f"s3://{parsed.netloc}/{'/'.join(parts)}"


def runtime_from_artifact_env(
    *,
    run_name: str,
    artifact_env: dict[str, str],
    bundle_dir: str = DEFAULT_BUNDLE_DIR,
) -> PortableMlflowRuntime:
    return PortableMlflowRuntime(
        run_name=sanitize_bundle_name(run_name),
        bundle_dir=bundle_dir,
        sync_uri=bundle_sync_uri(artifact_env.get("MLFLOW_ARTIFACTS_DESTINATION", ""), run_name),
    )
