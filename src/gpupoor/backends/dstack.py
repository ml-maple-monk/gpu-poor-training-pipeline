"""dstack-backed remote launch backend."""

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

from gpupoor import ops
from gpupoor.config import (
    DEFAULT_DSTACK_APPLY_TIMEOUT_BUFFER,
    DEFAULT_DSTACK_HEALTH_RECHECK_TIMEOUT,
    DEFAULT_DSTACK_MIN_RESTART_WAIT,
    DEFAULT_DSTACK_OFFER_QUERY_TIMEOUT,
    DEFAULT_DSTACK_OFFER_TIMEOUT,
    DEFAULT_DSTACK_PROVIDER_MAX_OFFERS,
    DEFAULT_DSTACK_RENDERED_TASK_PATH,
    DEFAULT_DSTACK_RUN_START_POLL_INTERVAL,
    DEFAULT_DSTACK_TARGETED_MAX_OFFERS,
    DEFAULT_DSTACK_TASK_DURATION_BUFFER_MINUTES,
    DEFAULT_HF_DATASET_REPO,
    DEFAULT_HF_PRETOKENIZED_DATASET_FILENAME,
    DEFAULT_REMOTE_IMAGE_TAG,
    DEFAULT_REMOTE_OUTPUT_DIR,
    DEFAULT_REMOTE_RUN_START_TIMEOUT_SECONDS,
    BackendConfig,
    RunConfig,
    find_dstack_bin,
    image_base_requires_registry_auth,
    load_remote_settings,
    merged_toml_b64,
    require_remote_settings,
    validate_dstack_image_base,
)
from gpupoor.services import portable_mlflow
from gpupoor.subprocess_utils import CommandError, bash_script, run_command
from gpupoor.utils import repo_path
from gpupoor.utils.http import http_ok
from gpupoor.utils.logging import get_logger

if TYPE_CHECKING:
    from gpupoor.deployer import ConnectionBundle

log = get_logger(__name__)

_MIN_RESTART_WAIT_SECONDS = DEFAULT_DSTACK_MIN_RESTART_WAIT
_HEALTH_RECHECK_TIMEOUT_SECONDS = DEFAULT_DSTACK_HEALTH_RECHECK_TIMEOUT
_DEFAULT_REMOTE_IMAGE_TAG = DEFAULT_REMOTE_IMAGE_TAG
_TASK_DURATION_BUFFER_MINUTES = DEFAULT_DSTACK_TASK_DURATION_BUFFER_MINUTES
_DEFAULT_OFFER_TIMEOUT_SECONDS = DEFAULT_DSTACK_OFFER_TIMEOUT
_OFFER_QUERY_TIMEOUT_SECONDS = DEFAULT_DSTACK_OFFER_QUERY_TIMEOUT
_DEFAULT_PROVIDER_MAX_OFFERS = DEFAULT_DSTACK_PROVIDER_MAX_OFFERS
_DEFAULT_TARGETED_MAX_OFFERS = DEFAULT_DSTACK_TARGETED_MAX_OFFERS
_RUN_START_POLL_INTERVAL_SECONDS = DEFAULT_DSTACK_RUN_START_POLL_INTERVAL
_DEFAULT_REMOTE_OUTPUT_DIR = DEFAULT_REMOTE_OUTPUT_DIR
_DEFAULT_HF_DATASET_REPO = DEFAULT_HF_DATASET_REPO
_DEFAULT_HF_PRETOKENIZED_DATASET_FILENAME = DEFAULT_HF_PRETOKENIZED_DATASET_FILENAME
_DSTACK_APPLY_TIMEOUT_BUFFER_SECONDS = DEFAULT_DSTACK_APPLY_TIMEOUT_BUFFER

__all__ = [
    "ensure_dstack_server",
    "fetch_offers",
    "fetch_targeted_offers",
    "http_ok",
    "launch_remote",
    "teardown_remote_state",
]


def cached_remote_image_metadata_path() -> Path:
    return repo_path(".tmp", "remote-image-tag.json")


def fleet_config_path() -> Path:
    return repo_path("dstack", "config", "fleet.dstack.yml")


def expected_training_base_image_base(settings: dict[str, str]) -> str:
    return settings.get("TRAINING_BASE_IMAGE_BASE", f"{settings['VCR_IMAGE_BASE']}-base")


def expected_fleet_name() -> str:
    path = fleet_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Required dstack fleet config missing: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("'\"")
            if name:
                return name
    raise RuntimeError(f"Could not find fleet name in {path}")


def dstack_server_restart_marker() -> Path:
    return Path.home() / ".dstack" / "server" / ".restart-required"


def configured_backends() -> tuple[str, ...]:
    config_path = Path.home() / ".dstack" / "server" / "config.yml"
    if not config_path.is_file():
        return ()

    backends: list[str] = []
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("- type:"):
            continue
        backend = line.split(":", 1)[1].strip()
        if backend and backend not in backends:
            backends.append(backend)
    return tuple(backends)


def stop_dstack_server(dstack_bin: str) -> bool:
    proc = subprocess.run(
        ["pgrep", "-f", f"{dstack_bin} server"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError("Failed to enumerate dstack server processes")

    stopped = False
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        os.kill(pid, 15)
        stopped = True
    return stopped


def restart_dstack_server_if_needed(
    dstack_bin: str,
    *,
    health_url: str,
    health_timeout_seconds: int,
    start_timeout_seconds: int,
    dry_run: bool,
) -> None:
    marker = dstack_server_restart_marker()
    if not marker.exists():
        return
    if dry_run:
        print(f"[DRY-RUN] Would restart dstack server because {marker} exists")
        return
    if http_ok(health_url, timeout_seconds=health_timeout_seconds) and stop_dstack_server(dstack_bin):
        deadline = time.monotonic() + max(_MIN_RESTART_WAIT_SECONDS, health_timeout_seconds)
        while time.monotonic() < deadline:
            if not http_ok(health_url, timeout_seconds=_HEALTH_RECHECK_TIMEOUT_SECONDS):
                break
            time.sleep(0.25)
    marker.unlink(missing_ok=True)
    ensure_dstack_server(
        dstack_bin,
        health_url=health_url,
        health_timeout_seconds=health_timeout_seconds,
        start_timeout_seconds=start_timeout_seconds,
        dry_run=dry_run,
    )


def ensure_dstack_server(
    dstack_bin: str,
    *,
    health_url: str,
    health_timeout_seconds: int,
    start_timeout_seconds: int,
    dry_run: bool,
) -> None:
    if http_ok(health_url, timeout_seconds=health_timeout_seconds):
        log.info("dstack server already running")
        return

    log_file = repo_path(".dstack-server.log")
    if dry_run:
        print(f"[DRY-RUN] Would run: {dstack_bin} server >> {log_file} 2>&1 &")
        return

    log.info("dstack server not running; starting it in background")
    with log_file.open("ab") as handle:
        # dstack >=0.20.16 prompts `Update the main project in ~/.dstack/config.yml?`
        # on first start. With no TTY the interactive `input()` raises EOFError and
        # the server crashes before binding to port 3000. Feed "y\n" on stdin so
        # startup is non-interactive, then close stdin so the daemon doesn't wait
        # for more input.
        process = subprocess.Popen(
            [dstack_bin, "server"],
            stdin=subprocess.PIPE,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            if process.stdin is None:
                raise RuntimeError("dstack server Popen did not return stdin")
            process.stdin.write(b"y\n")
            process.stdin.close()
        except (BrokenPipeError, OSError, RuntimeError):
            # Don't leak the Popen handle if the stdin handshake fails; the
            # subsequent health poll still decides whether startup succeeded.
            pass

    # Wall-clock deadline instead of iteration count. The old
    # `for _ in range(start_timeout_seconds)` implicitly assumed each
    # iteration costs ~1s, but http_ok's internal timeout (several seconds
    # on registry stalls) stretched the real wait well past the knob's name.
    # Honor the knob literally: stop probing once start_timeout_seconds of
    # wall-clock time has elapsed, regardless of how many probes fit in it.
    deadline = time.monotonic() + start_timeout_seconds
    while time.monotonic() < deadline:
        if http_ok(health_url, timeout_seconds=health_timeout_seconds):
            log.info("dstack server healthy")
            return
        time.sleep(1)
    raise RuntimeError(f"dstack server did not become healthy; check {log_file}")


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
    if payload.get("training_base_image_base") != expected_training_base_image_base(settings):
        return None
    return payload


def _metadata_has_no_attestation(payload: Mapping[str, object]) -> bool:
    return payload.get("provenance_attestation") is False and payload.get("build_tool") == "docker-buildx"


def read_cached_remote_image_tag(settings: dict[str, str]) -> str | None:
    payload = read_cached_remote_image_metadata(settings)
    if payload is None or not _metadata_has_no_attestation(payload):
        return None

    image_tag = payload.get("image_tag")
    if not isinstance(image_tag, str) or not image_tag:
        return None
    return image_tag


def verify_no_attestation_image_metadata(settings: dict[str, str], image_tag: str) -> None:
    payload = read_cached_remote_image_metadata(settings)
    metadata_path = cached_remote_image_metadata_path()
    if payload is None:
        raise RuntimeError(
            f"Remote image tag '{image_tag}' has no matching no-attestation metadata at {metadata_path}. "
            "Rebuild with training/scripts/build-and-push.sh or set backend.remote_image_tag to a "
            "locally cached no-attestation tag."
        )
    if payload.get("image_tag") != image_tag:
        raise RuntimeError(
            f"Remote image tag '{image_tag}' does not match cached metadata tag '{payload.get('image_tag')}'. "
            "Rebuild with training/scripts/build-and-push.sh or choose the cached tag."
        )
    if not _metadata_has_no_attestation(payload):
        raise RuntimeError(
            f"Remote image tag '{image_tag}' was not proven to be built without provenance attestations. "
            "dstack 0.20.x can fail on OCI attestation manifests; rebuild with "
            "training/scripts/build-and-push.sh so metadata records provenance_attestation=false."
        )


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
    """Build the env contract shared by remote dstack and local-emulator workers."""
    env = dict(connector_env or {})
    env.pop("GPUPOOR_RUN_CONFIG", None)
    hf_dataset_repo = settings.get("HF_DATASET_REPO", _DEFAULT_HF_DATASET_REPO)
    injected = {
        "GPUPOOR_RUN_CONFIG_B64": run_config_b64,
        "VERDA_PROFILE": profile,
        "DSTACK_RUN_NAME": config.name,
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


def task_max_duration(time_cap_seconds: int) -> str:
    if time_cap_seconds <= 0:
        raise ValueError("time_cap_seconds must be positive")
    # Give the in-container timeout a clean
    # head start so the SIGTERM handler in train_pretrain.py can call
    # _mlflow_helper.finish(status='KILLED') before dstack's max_duration fires
    # as last-resort safety. 2-minute buffer covers SIGTERM grace (30s) plus
    # MLflow finalize over a slow Cloudflare tunnel.
    minutes = max(_TASK_DURATION_BUFFER_MINUTES, (time_cap_seconds + 59) // 60 + _TASK_DURATION_BUFFER_MINUTES)
    return f"{minutes}m"


def render_task(settings: dict[str, str], config: RunConfig, image_sha: str) -> Path:
    rendered_task = repo_path(*Path(DEFAULT_DSTACK_RENDERED_TASK_PATH).parts)
    rendered_task.parent.mkdir(parents=True, exist_ok=True)
    render_env = dict(settings)
    render_env["IMAGE_SHA"] = image_sha
    render_env["TASK_NAME"] = config.name
    render_env["TASK_MAX_DURATION"] = task_max_duration(config.recipe.time_cap_seconds)
    # Task/GPU overrides: unset fields fall back to shell defaults so the
    # baseline example stays unchanged while targeted runs (e.g. B300) can
    # pick their own instance type from TOML.
    render_env.update(config.remote.to_env())
    bash_script(
        repo_path("dstack", "scripts", "render-pretrain-task.sh"),
        str(rendered_task),
        env=render_env,
    )
    return rendered_task


def _offer_command(
    dstack_bin: str,
    *,
    max_offers: int,
    backend: str | None = None,
    spot_policy: str | None = "auto",
) -> list[str]:
    command = [
        dstack_bin,
        "offer",
        "--json",
        "--max-offers",
        str(max_offers),
    ]
    if spot_policy == "auto":
        command.append("--spot-auto")
    elif spot_policy == "spot":
        command.append("--spot")
    elif spot_policy == "on-demand":
        command.append("--on-demand")
    elif spot_policy:
        command.extend(["--spot-policy", spot_policy])
    if backend:
        command.extend(["--backend", backend])
    return command


def _load_offer_payload(command: list[str], *, timeout: int = _DEFAULT_OFFER_TIMEOUT_SECONDS) -> dict[str, object]:
    output = run_command(command, capture_output=True, quiet=True, timeout=timeout).stdout
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError("dstack offer returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("dstack offer JSON must be an object")
    return payload


def provider_offer_diagnostics(
    dstack_bin: str,
    *,
    max_offers: int = _DEFAULT_PROVIDER_MAX_OFFERS,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    offers: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for backend in configured_backends():
        try:
            payload = _load_offer_payload(
                _offer_command(dstack_bin, max_offers=max_offers, backend=backend),
                timeout=_OFFER_QUERY_TIMEOUT_SECONDS,
            )
            offers.extend(offer for offer in payload.get("offers", []) if isinstance(offer, dict))
            diagnostics.append(
                {
                    "backend": backend,
                    "status": "ok",
                    "total_offers": payload.get("total_offers", 0),
                    "visible_offers": len(payload.get("offers", [])),
                }
            )
        except subprocess.TimeoutExpired:
            diagnostics.append({"backend": backend, "status": "timeout"})
        except Exception as exc:
            diagnostics.append({"backend": backend, "status": "error", "reason": str(exc)})
    return offers, diagnostics


def fetch_offers(dstack_bin: str, *, max_offers: int = _DEFAULT_PROVIDER_MAX_OFFERS) -> dict[str, object]:
    payload = _load_offer_payload(_offer_command(dstack_bin, max_offers=max_offers))
    provider_offers, diagnostics = provider_offer_diagnostics(dstack_bin, max_offers=max_offers)
    if provider_offers:
        payload["offers"] = provider_offers
        payload["total_offers"] = sum(
            int(item.get("total_offers", 0)) for item in diagnostics if item.get("status") == "ok"
        )
    payload["provider_diagnostics"] = diagnostics
    return payload


def fetch_targeted_offers(
    dstack_bin: str,
    *,
    backend: str,
    gpu: str,
    count: int,
    mode: str,
    regions: tuple[str, ...] = (),
    max_price: float | None = None,
    max_offers: int = _DEFAULT_TARGETED_MAX_OFFERS,
) -> dict[str, object]:
    command = _offer_command(
        dstack_bin,
        max_offers=max_offers,
        backend=backend or None,
        spot_policy=None,
    )
    # NOTE:
    # dstack 0.20.16 advertises `--gpu`, but the installed CLI currently
    # rejects real GPU-name filters such as `H100`, `H100:1..`, and
    # `RTX5090`. Query the backend/region/price/mode slice and let the seeker
    # apply GPU/count matching client-side over the returned offers.
    _ = (gpu, count)
    if mode == "spot":
        command.append("--spot")
    elif mode == "on-demand":
        command.append("--on-demand")
    elif mode:
        command.extend(["--spot-policy", mode])
    if max_price is not None:
        command.extend(["--max-price", str(max_price)])
    for region in regions:
        command.extend(["--region", region])
    return _load_offer_payload(command, timeout=_OFFER_QUERY_TIMEOUT_SECONDS)


def dstack_has_run(dstack_bin: str, run_name: str) -> bool:
    """Return True if dstack ps reports a run with the given name.

    Filtering by name avoids trusting runs[0] as "the run we just
    launched". The dstack account may be shared across concurrent
    launches and the CLI's run ordering is not contractually stable.
    """
    if not run_name:
        return False
    command = [dstack_bin, "ps", "--json"]
    try:
        output = subprocess.check_output(command, text=True)
    except subprocess.CalledProcessError as exc:
        raise CommandError(command, exc.returncode) from exc
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        # Surface as CommandError so the caller sees a uniform failure type
        # and does not have to catch JSON-layer details at every call site.
        raise CommandError(command, 0) from exc
    runs = data.get("runs", []) if isinstance(data, dict) else data
    for run in runs:
        candidate = run.get("run_name") or (run.get("run_spec") or {}).get("run_name") or ""
        if candidate == run_name:
            return True
    return False


def dstack_run_status_triplet(dstack_bin: str, run_name: str) -> tuple[str, str, str]:
    output = subprocess.check_output([dstack_bin, "ps", "--json"], text=True)
    data = json.loads(output)
    runs = data.get("runs", []) if isinstance(data, dict) else data
    for run in runs:
        candidate = run.get("run_name") or (run.get("run_spec") or {}).get("run_name") or ""
        if candidate != run_name:
            continue
        latest = run.get("latest_job_submission") or {}
        return (
            str(run.get("status") or ""),
            str(latest.get("status") or ""),
            str(latest.get("termination_reason") or ""),
        )
    return ("", "", "")


def verify_fleet_ready(dstack_bin: str, *, fleet_name: str | None = None, timeout_seconds: int = 30) -> None:
    expected_name = fleet_name or expected_fleet_name()
    try:
        completed = run_command(
            [dstack_bin, "fleet", "get", "--json", expected_name],
            capture_output=True,
            quiet=True,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        raise RuntimeError(
            f"dstack fleet '{expected_name}' is not ready or dstack server is unreachable. "
            f"Run `gpupoor dstack fleet-apply` using {fleet_config_path()} before launching tasks."
        ) from exc

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"dstack fleet '{expected_name}' returned invalid JSON. "
            f"Run `gpupoor dstack fleet-apply` using {fleet_config_path()} before launching tasks."
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"dstack fleet '{expected_name}' JSON must be an object")

    actual_name = payload.get("name")
    if actual_name is None and isinstance(payload.get("fleet"), dict):
        actual_name = payload["fleet"].get("name")
    if actual_name != expected_name:
        raise RuntimeError(
            f"dstack fleet readiness check expected '{expected_name}' but got '{actual_name}'. "
            f"Run `gpupoor dstack fleet-apply` using {fleet_config_path()}."
        )

    status_values = []
    for key in ("status", "state"):
        value = payload.get(key)
        if isinstance(value, str):
            status_values.append(value)
    fleet_payload = payload.get("fleet")
    if isinstance(fleet_payload, dict):
        for key in ("status", "state"):
            value = fleet_payload.get(key)
            if isinstance(value, str):
                status_values.append(value)
    terminal = {"terminated", "deleting", "failed", "error"}
    for value in status_values:
        if value.strip().lower() in terminal:
            raise RuntimeError(
                f"dstack fleet '{expected_name}' is in terminal state '{value}'. "
                f"Run `gpupoor dstack fleet-apply` using {fleet_config_path()} before launching tasks."
            )


def wait_for_run_start(
    dstack_bin: str,
    run_name: str,
    *,
    max_wait: int = DEFAULT_REMOTE_RUN_START_TIMEOUT_SECONDS,
) -> None:
    log.info("Waiting for run '%s' to leave startup states", run_name)
    elapsed = 0
    while elapsed < max_wait:
        run_status, job_status, termination_reason = dstack_run_status_triplet(dstack_bin, run_name)
        if run_status == "running" or job_status == "running":
            log.info("Run '%s' is running", run_name)
            return
        if run_status == "provisioning" or job_status == "provisioning":
            log.info("Run '%s' is provisioning (pulling image, ~3-10 min)... [%ds]", run_name, elapsed)
            time.sleep(_RUN_START_POLL_INTERVAL_SECONDS)
            elapsed += _RUN_START_POLL_INTERVAL_SECONDS
            continue
        if run_status in {"pending", "submitted"} and termination_reason == "failed_to_start_due_to_no_capacity":
            log.info(
                "Run '%s' is waiting after a no-capacity offer; polling for the next submission",
                run_name,
            )
            time.sleep(_RUN_START_POLL_INTERVAL_SECONDS)
            elapsed += _RUN_START_POLL_INTERVAL_SECONDS
            continue
        if run_status in {"terminated", "failed", "stopped", "completed"} or job_status in {
            "terminated",
            "failed",
            "stopped",
            "completed",
        }:
            raise RuntimeError(
                f"Run '{run_name}' reached terminal job status '{job_status}' "
                f"before steady-state attach ({termination_reason or 'none'})"
            )
        time.sleep(_RUN_START_POLL_INTERVAL_SECONDS)
        elapsed += _RUN_START_POLL_INTERVAL_SECONDS
    raise RuntimeError(f"Run '{run_name}' did not reach RUNNING within {max_wait}s")


def track_run(run_name: str) -> None:
    if not run_name:
        return
    run_ids_file = repo_path(".run-ids")
    # Hold an exclusive advisory lock while appending so concurrent launches
    # (two `dstack apply` invocations racing to tag the .run-ids sidecar)
    # don't shred each other's lines. `with open(...)` on close releases the
    # lock implicitly via file descriptor close. Guard fcntl for non-POSIX
    # where flock is unavailable; on Windows the lock degrades to best-effort
    # (same as prior behavior), but POSIX deployments get real ordering.
    with run_ids_file.open("a", encoding="utf-8") as handle:
        if fcntl is not None and hasattr(fcntl, "flock"):
            fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(f"{run_name}\n")


def kill_tunnel() -> None:
    pid_file = repo_path(".cf-tunnel.pid")
    if not pid_file.is_file():
        return
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        pid = 0
    if pid:
        should_kill = True
        # On Linux, confirm the PID still belongs to cloudflared before
        # signalling. PIDs recycle; a stale .cf-tunnel.pid could name any
        # unrelated process (shell, editor, build) and we must not SIGTERM it.
        if platform.system() == "Linux":
            comm_path = Path(f"/proc/{pid}/comm")
            try:
                comm = comm_path.read_text(encoding="utf-8").strip()
            except OSError:
                # /proc entry gone -> PID no longer exists; nothing to kill.
                should_kill = False
            else:
                if comm != "cloudflared":
                    # Original print went to stdout (no file=sys.stderr); preserve
                    # that via log.info so captured-stdout callers still see it.
                    # The "WARN:" marker stays in the message text.
                    log.info(
                        "WARN: .cf-tunnel.pid %s is '%s', not cloudflared; skipping kill",
                        pid,
                        comm,
                    )
                    should_kill = False
        # On non-Linux (e.g. macOS), /proc is not available. Fall through and
        # trust the pid file; this matches prior behavior on those platforms.
        if should_kill:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
    for suffix in (".cf-tunnel.pid", ".cf-tunnel.url", ".cf-tunnel.log"):
        path = repo_path(suffix)
        if path.exists():
            path.unlink()


def teardown_remote_state() -> None:
    kill_tunnel()
    dstack_bin = find_dstack_bin()
    run_ids_file = repo_path(".run-ids")
    if not run_ids_file.is_file():
        return
    for raw_line in run_ids_file.read_text(encoding="utf-8").splitlines():
        run_name = raw_line.strip()
        if not run_name:
            continue
        run_command([dstack_bin, "stop", run_name, "-y"], check=False)
    run_ids_file.unlink()


def launch_remote(
    config: RunConfig,
    *,
    skip_build: bool | None = None,
    dry_run: bool = False,
    configure_server: bool = True,
    connection_bundle: ConnectionBundle | None = None,
) -> None:
    if config.backend.kind != "dstack":
        raise ValueError("launch_remote requires backend.kind='dstack'")

    settings = load_remote_settings(config.remote)
    validate_dstack_image_base(settings["VCR_IMAGE_BASE"])
    if image_base_requires_registry_auth(settings["VCR_IMAGE_BASE"]):
        require_remote_settings(settings)
    dstack_bin = find_dstack_bin()

    ops.run_preflight(remote=True, doctor=config.doctor, remote_config=config.remote)
    if configure_server:
        if dry_run:
            print("[DRY-RUN] Would configure dstack server")
        else:
            bash_script(repo_path("dstack", "scripts", "setup-config.sh"))
    restart_dstack_server_if_needed(
        dstack_bin,
        health_url=config.remote.dstack_server_health_url,
        health_timeout_seconds=config.remote.health_timeout_seconds,
        start_timeout_seconds=config.remote.dstack_server_start_timeout_seconds,
        dry_run=dry_run,
    )
    ensure_dstack_server(
        dstack_bin,
        health_url=config.remote.dstack_server_health_url,
        health_timeout_seconds=config.remote.health_timeout_seconds,
        start_timeout_seconds=config.remote.dstack_server_start_timeout_seconds,
        dry_run=dry_run,
    )

    rendered_task = None
    launched_remote_run = False
    try:
        use_skip_build = config.backend.skip_build if skip_build is None else skip_build
        cached_image_tag = None

        if config.recipe.prepare_data:
            if dry_run:
                print("[DRY-RUN] Would prepare and upload the pretokenized dataset artifact")
            else:
                bash_script(
                    repo_path("training", "scripts", "prepare-data.sh"),
                    env={**os.environ, **settings, "UPLOAD_PRETOKENIZED_DATASET": "1"},
                )
        else:
            log.info("Skipping dataset preparation (prepare_data=false)")

        if not use_skip_build:
            head_image_tag = git_short_sha()
            if not dry_run and not git_has_tracked_changes():
                cached_image_tag = read_cached_remote_image_tag(settings)
                if cached_image_tag == head_image_tag:
                    use_skip_build = True
                    log.info("Reusing previously published remote image tag '%s'", cached_image_tag)
            if dry_run:
                print("[DRY-RUN] Would build and push the remote image")
            elif not use_skip_build:
                bash_script(repo_path("training", "scripts", "build-and-push.sh"), env={**os.environ, **settings})
        else:
            log.info("Skipping remote image build")

        image_sha = remote_image_tag(
            config.backend,
            skip_build=use_skip_build,
            dry_run=dry_run,
            settings=settings,
            cached_tag=cached_image_tag,
        )
        connector_env = (
            connection_bundle.to_runtime_env()
            if connection_bundle is not None
            else portable_mlflow.runtime_from_artifact_env(run_name=config.name, artifact_env={}).to_env()
        )
        mlflow_url = connector_env.get("MLFLOW_TRACKING_URI", portable_mlflow.PORTABLE_TRACKING_URI)

        log.info("Config: %s", config.source)
        log.info("Backend: %s", config.backend.kind)
        log.info("MLFLOW_URL=%s", mlflow_url)
        log.info("IMAGE_SHA=%s", image_sha)
        log.info("VCR_IMAGE_BASE=%s", settings["VCR_IMAGE_BASE"])

        if dry_run:
            print("[DRY-RUN] Would verify remote image metadata proves provenance_attestation=false")
            print(f"[DRY-RUN] Would verify dstack fleet '{expected_fleet_name()}' is ready")
            print(f"[DRY-RUN] Would render task with IMAGE_SHA={image_sha}")
            print("[DRY-RUN] Would call dstack apply with HF_TOKEN and MLflow env")
            return

        verify_no_attestation_image_metadata(settings, image_sha)
        verify_fleet_ready(dstack_bin)
        rendered_task = render_task(settings, config, image_sha)
        apply_env = remote_worker_env(
            config,
            settings,
            run_config_b64=merged_toml_b64(config),
            profile="remote",
            out_dir=settings.get("OUT_DIR", _DEFAULT_REMOTE_OUTPUT_DIR),
            hf_token=read_required_secret("hf_token"),
            connector_env=connector_env,
            mlflow_tracking_uri=mlflow_url,
        )
        # `dstack apply` can hang indefinitely on registry auth or
        # network stalls; without a timeout the CLI freezes with no
        # liveness signal. Budget: the existing run-start window plus a
        # 60s buffer covers dstack's own internal retries without
        # inventing a new knob.
        apply_cmd = [dstack_bin, "apply", "-f", str(rendered_task), "-y", "-d", "--force"]
        apply_timeout = config.remote.run_start_timeout_seconds + _DSTACK_APPLY_TIMEOUT_BUFFER_SECONDS
        run_command(apply_cmd, env=apply_env, timeout=apply_timeout)

        run_name = config.name
        if dstack_has_run(dstack_bin, run_name):
            track_run(run_name)
            wait_for_run_start(dstack_bin, run_name, max_wait=config.remote.run_start_timeout_seconds)
            launched_remote_run = True
            # Poll logs via REST API (no SSH) until the run finishes.
            # dstack 0.20.17 'logs' dumps current output and exits;
            # we loop with --since to get incremental updates.
            log.info("Streaming logs for run '%s' (Ctrl+C to detach)...", run_name)
            try:
                last_since = "0s"
                while True:
                    try:
                        run_command(
                            [dstack_bin, "logs", run_name, "--since", last_since],
                            timeout=60,
                            quiet=True,
                        )
                    except CommandError:
                        pass  # logs command may fail if run just finished
                    status, _, _ = dstack_run_status_triplet(dstack_bin, run_name)
                    if status not in {"running", "provisioning", "submitted", "pending"}:
                        log.info("Run '%s' finished with status: %s", run_name, status)
                        break
                    last_since = f"{_RUN_START_POLL_INTERVAL_SECONDS}s"
                    time.sleep(_RUN_START_POLL_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                log.info("Detached from log stream (run '%s' continues on RunPod)", run_name)
        else:
            # Original print went to stdout (no file=sys.stderr); preserve via
            # log.info so stream routing stays the same. "WARN:" stays in text.
            log.info(
                "WARN: dstack apply reported success but run '%s' is not visible in dstack ps; skipping track/wait",
                run_name,
            )
    finally:
        if rendered_task and rendered_task.exists():
            rendered_task.unlink()
        if launched_remote_run:
            log.info("Remote worker owns portable MLflow bundle finalization")
