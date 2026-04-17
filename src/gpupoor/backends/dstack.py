"""dstack-backed remote launch backend."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback (Windows)
    fcntl = None  # type: ignore[assignment]

from gpupoor import ops
from gpupoor.config import (
    BackendConfig,
    RunConfig,
    find_dstack_bin,
    load_remote_settings,
    require_remote_settings,
    DEFAULT_DSTACK_RENDERED_TASK_PATH,
    DEFAULT_DSTACK_TUNNEL_JOIN_TIMEOUT,
    DEFAULT_DSTACK_MIN_RESTART_WAIT,
    DEFAULT_DSTACK_HEALTH_RECHECK_TIMEOUT,
    DEFAULT_DSTACK_TASK_SIGTERM_GRACE,
    DEFAULT_DSTACK_TASK_DURATION_BUFFER_MINUTES,
    DEFAULT_DSTACK_OFFER_TIMEOUT,
    DEFAULT_DSTACK_OFFER_QUERY_TIMEOUT,
    DEFAULT_DSTACK_PROVIDER_MAX_OFFERS,
    DEFAULT_DSTACK_TARGETED_MAX_OFFERS,
    DEFAULT_DSTACK_RUN_START_POLL_INTERVAL,
    DEFAULT_DSTACK_APPLY_TIMEOUT_BUFFER,
    DEFAULT_DSTACK_FINAL_TUNNEL_JOIN_TIMEOUT,
    DEFAULT_DSTACK_DRY_RUN_MLFLOW_URL,
    DEFAULT_REMOTE_IMAGE_TAG,
    DEFAULT_REMOTE_OUTPUT_DIR,
    DEFAULT_REMOTE_DATASET_PATH,
    DEFAULT_HF_DATASET_REPO,
    DEFAULT_HF_PRETOKENIZED_DATASET_FILENAME,
    DEFAULT_REMOTE_RUN_START_TIMEOUT_SECONDS,
)
from gpupoor.runtime_config import merged_toml_b64
from gpupoor.subprocess_utils import CommandError, bash_script, run_command
from gpupoor.utils import repo_path
from gpupoor.utils.http import http_ok
from gpupoor.utils.logging import get_logger

if TYPE_CHECKING:
    from gpupoor.connector import ConnectionBundle

log = get_logger(__name__)

_TUNNEL_JOIN_TIMEOUT = DEFAULT_DSTACK_TUNNEL_JOIN_TIMEOUT
_MIN_RESTART_WAIT_SECONDS = DEFAULT_DSTACK_MIN_RESTART_WAIT
_HEALTH_RECHECK_TIMEOUT_SECONDS = DEFAULT_DSTACK_HEALTH_RECHECK_TIMEOUT
_DEFAULT_REMOTE_IMAGE_TAG = DEFAULT_REMOTE_IMAGE_TAG
_TASK_SIGTERM_GRACE_SECONDS = DEFAULT_DSTACK_TASK_SIGTERM_GRACE
_TASK_DURATION_BUFFER_MINUTES = DEFAULT_DSTACK_TASK_DURATION_BUFFER_MINUTES
_DEFAULT_OFFER_TIMEOUT_SECONDS = DEFAULT_DSTACK_OFFER_TIMEOUT
_OFFER_QUERY_TIMEOUT_SECONDS = DEFAULT_DSTACK_OFFER_QUERY_TIMEOUT
_DEFAULT_PROVIDER_MAX_OFFERS = DEFAULT_DSTACK_PROVIDER_MAX_OFFERS
_DEFAULT_TARGETED_MAX_OFFERS = DEFAULT_DSTACK_TARGETED_MAX_OFFERS
_RUN_START_POLL_INTERVAL_SECONDS = DEFAULT_DSTACK_RUN_START_POLL_INTERVAL
_DRY_RUN_MLFLOW_URL = DEFAULT_DSTACK_DRY_RUN_MLFLOW_URL
_CONTAINER_REMOTE_DATASET_PATH = DEFAULT_REMOTE_DATASET_PATH
_DEFAULT_REMOTE_OUTPUT_DIR = DEFAULT_REMOTE_OUTPUT_DIR
_DEFAULT_HF_DATASET_REPO = DEFAULT_HF_DATASET_REPO
_DEFAULT_HF_PRETOKENIZED_DATASET_FILENAME = DEFAULT_HF_PRETOKENIZED_DATASET_FILENAME
_DSTACK_APPLY_TIMEOUT_BUFFER_SECONDS = DEFAULT_DSTACK_APPLY_TIMEOUT_BUFFER
_FINAL_TUNNEL_JOIN_TIMEOUT_SECONDS = DEFAULT_DSTACK_FINAL_TUNNEL_JOIN_TIMEOUT

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


def expected_training_base_image_base(settings: dict[str, str]) -> str:
    return settings.get("TRAINING_BASE_IMAGE_BASE", f"{settings['VCR_IMAGE_BASE']}-base")


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


def read_cached_remote_image_tag(settings: dict[str, str]) -> str | None:
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
                "Run '%s' is retrying after a no-capacity offer; waiting for the next submission",
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
    require_remote_settings(settings)
    dstack_bin = find_dstack_bin()

    ops.run_preflight(remote=True, doctor=config.doctor, remote_config=config.remote)
    if configure_server:
        bash_script(repo_path("dstack", "scripts", "setup-config.sh"))
    restart_dstack_server_if_needed(
        dstack_bin,
        health_url=config.remote.dstack_server_health_url,
        health_timeout_seconds=config.remote.health_timeout_seconds,
        start_timeout_seconds=config.remote.dstack_server_start_timeout_seconds,
        dry_run=dry_run,
    )
    verify_mlflow(config.remote.mlflow_health_url, timeout_seconds=config.remote.health_timeout_seconds)
    ensure_dstack_server(
        dstack_bin,
        health_url=config.remote.dstack_server_health_url,
        health_timeout_seconds=config.remote.health_timeout_seconds,
        start_timeout_seconds=config.remote.dstack_server_start_timeout_seconds,
        dry_run=dry_run,
    )

    tunnel_thread: threading.Thread | None = None
    tunnel_exception: BaseException | None = None
    started_tunnel = False

    rendered_task = None
    launched_remote_run = False
    try:
        if dry_run:
            print("[DRY-RUN] Would start the MLflow Cloudflare tunnel")
        elif connection_bundle is None:
            existing_tunnel_url = repo_path(".cf-tunnel.url")
            existing_tunnel_pid = repo_path(".cf-tunnel.pid")
            if existing_tunnel_url.exists() and existing_tunnel_pid.exists():
                log.info("Reusing existing Cloudflare tunnel (found .cf-tunnel.url and .cf-tunnel.pid)")
                started_tunnel = True
            else:

                def _run_tunnel() -> None:
                    nonlocal tunnel_exception
                    try:
                        bash_script(repo_path("infrastructure", "mlflow", "scripts", "run-tunnel.sh"))
                    except Exception as exc:
                        tunnel_exception = exc

                tunnel_thread = threading.Thread(target=_run_tunnel, daemon=True)
                tunnel_thread.start()
                started_tunnel = True

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

        # Wait for the background tunnel thread to finish before reading the URL.
        if tunnel_thread is not None:
            tunnel_thread.join(timeout=_TUNNEL_JOIN_TIMEOUT)
            if tunnel_thread.is_alive():
                raise RuntimeError("Tunnel startup timed out")
            if tunnel_exception is not None:
                raise tunnel_exception

        image_sha = remote_image_tag(
            config.backend,
            skip_build=use_skip_build,
            dry_run=dry_run,
            settings=settings,
            cached_tag=cached_image_tag,
        )
        if dry_run:
            mlflow_url = _DRY_RUN_MLFLOW_URL
        elif connection_bundle is not None:
            mlflow_url = connection_bundle.mlflow_tracking_uri
        else:
            mlflow_url = repo_path(".cf-tunnel.url").read_text(encoding="utf-8").strip()

        log.info("Config: %s", config.source)
        log.info("Backend: %s", config.backend.kind)
        log.info("MLFLOW_URL=%s", mlflow_url)
        log.info("IMAGE_SHA=%s", image_sha)
        log.info("VCR_IMAGE_BASE=%s", settings["VCR_IMAGE_BASE"])

        if dry_run:
            print(f"[DRY-RUN] Would render task with IMAGE_SHA={image_sha}")
            print("[DRY-RUN] Would call dstack apply with HF_TOKEN and MLflow env")
            return

        rendered_task = render_task(settings, config, image_sha)
        apply_env = {
            "HF_TOKEN": read_required_secret("hf_token"),
            "GPUPOOR_RUN_CONFIG_B64": merged_toml_b64(config),
            "VERDA_PROFILE": "remote",
            "DSTACK_RUN_NAME": config.name,
            "OUT_DIR": settings.get("OUT_DIR", _DEFAULT_REMOTE_OUTPUT_DIR),
            "MLFLOW_TRACKING_URI": mlflow_url,
            "HF_DATASET_REPO": settings.get("HF_DATASET_REPO", _DEFAULT_HF_DATASET_REPO),
            "HF_DATASET_FILENAME": settings.get("HF_DATASET_FILENAME", Path(config.recipe.dataset_path).name),
            "HF_PRETOKENIZED_DATASET_REPO": settings.get(
                "HF_PRETOKENIZED_DATASET_REPO",
                settings.get("HF_DATASET_REPO", _DEFAULT_HF_DATASET_REPO),
            ),
            "HF_PRETOKENIZED_DATASET_FILENAME": settings.get(
                "HF_PRETOKENIZED_DATASET_FILENAME",
                _DEFAULT_HF_PRETOKENIZED_DATASET_FILENAME,
            ),
            **(connection_bundle.to_runtime_env() if connection_bundle is not None else {}),
        }
        # `dstack apply` can hang indefinitely on registry auth or
        # network stalls; without a timeout the CLI freezes with no
        # liveness signal. Budget: the existing run-start window plus a
        # 60s buffer covers dstack's own internal retries without
        # inventing a new knob.
        apply_cmd = [dstack_bin, "apply", "-f", str(rendered_task), "-y", "-d"]
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
        if tunnel_thread is not None and tunnel_thread.is_alive():
            tunnel_thread.join(timeout=_FINAL_TUNNEL_JOIN_TIMEOUT_SECONDS)
        if started_tunnel:
            if launched_remote_run:
                log.info("Keeping Cloudflare tunnel alive until teardown so remote MLflow stays reachable")
            else:
                kill_tunnel()
