"""Verda cloud backend stub — HTTP API not yet configured."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpupoor.config import RemoteConfig, RunConfig
    from gpupoor.deployer import ConnectionBundle

_NOT_CONFIGURED = (
    "Verda direct API is not yet configured. "
    "Provide the Verda API base URL and credentials, then implement this backend."
)


def fetch_offers(config: RemoteConfig) -> list[dict]:
    raise NotImplementedError(_NOT_CONFIGURED)


def submit_job(config: RunConfig, image_ref: str, env: dict[str, str], **kwargs) -> str:
    raise NotImplementedError(_NOT_CONFIGURED)


def poll_job_status(job_id: str) -> tuple[str, int | None]:
    raise NotImplementedError(_NOT_CONFIGURED)


def stream_logs(job_id: str, **kwargs) -> None:
    raise NotImplementedError(_NOT_CONFIGURED)


def teardown_job(job_id: str) -> None:
    raise NotImplementedError(_NOT_CONFIGURED)


def launch_remote(
    config: RunConfig,
    *,
    skip_build: bool | None = None,
    dry_run: bool = False,
    connection_bundle: ConnectionBundle | None = None,
) -> None:
    raise NotImplementedError(_NOT_CONFIGURED)
