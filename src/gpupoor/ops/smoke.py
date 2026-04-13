"""Smoke test helpers for the local emulator pipeline."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from gpupoor.config import DoctorConfig, SmokeConfig
from gpupoor.ops.doctor import run_preflight
from gpupoor.ops.secrets import leak_scan
from gpupoor.subprocess_utils import run_command
from gpupoor.utils import repo_path


class SmokeReporter:
    """Collect per-probe results while preserving a simple terminal report."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def pass_probe(self, label: str, message: str) -> None:
        print(f"PROBE {label} PASS: {message}")
        self.passed += 1

    def fail_probe(self, label: str, message: str) -> None:
        print(f"PROBE {label} FAIL: {message}")
        self.failed += 1


def run_smoke(config: SmokeConfig | None = None, *, doctor: DoctorConfig | None = None) -> None:
    settings = config or SmokeConfig()
    reporter = SmokeReporter()
    compose = _compose_command(cpu=settings.cpu)

    print("--- Preflight ---")
    run_preflight(remote=False, doctor=doctor)

    runtime_env = _runtime_env()
    local_image = _build_local_image(settings.base_image)

    print("--- Build & Run ---")
    try:
        run_command([*compose, "up", "--build", "-d"], env=runtime_env)
        _wait_for_health(settings.health_port, expected=200, timeout_seconds=settings.health_timeout_seconds)
        _probe_uid_gid(compose, reporter)
        _probe_non_root_write(compose, reporter)
        _probe_sigterm_latency(compose, reporter, timeout_seconds=settings.sigterm_timeout_seconds)
        run_command([*compose, "up", "-d"], env=runtime_env, check=False)
        time.sleep(2)
        _probe_env_leak(compose, reporter)
        _probe_degraded_gating(runtime_env, reporter, settings)
        _probe_data_wait_timeout(local_image, reporter, timeout_seconds=settings.data_wait_timeout_seconds)

        print("--- Leak Scan ---")
        try:
            leak_scan()
        except RuntimeError as exc:
            print(exc)
            reporter.failed += 1
    finally:
        run_command([*compose, "down", "-v", "--remove-orphans"], check=False)

    print("")
    print(f"=== Smoke Results: {reporter.passed} passed, {reporter.failed} failed ===")
    if reporter.failed:
        raise RuntimeError("Smoke FAILED")


def _runtime_env() -> dict[str, str]:
    if os.environ.get("HF_TOKEN"):
        return {}
    token_file = repo_path("hf_token")
    if token_file.is_file():
        return {"HF_TOKEN": token_file.read_text(encoding="utf-8").strip()}
    return {}


def _compose_command(*, cpu: bool, extra_files: list[Path] | None = None) -> list[str]:
    command = [
        "docker",
        "compose",
        "-f",
        str(repo_path("infrastructure", "local-emulator", "compose", "docker-compose.yml")),
    ]
    if cpu:
        command.extend(
            [
                "-f",
                str(repo_path("infrastructure", "local-emulator", "compose", "docker-compose.cpu.yml")),
            ]
        )
    for extra_file in extra_files or []:
        command.extend(["-f", str(extra_file)])
    return command


def _build_local_image(base_image: str) -> str:
    local_tag = subprocess.check_output(
        ["git", "-C", str(repo_path()), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()
    local_image = f"verda-local:{local_tag or 'local'}"
    run_command(
        [
            "docker",
            "build",
            "--build-arg",
            f"BASE_IMAGE={base_image}",
            "-f",
            str(repo_path("infrastructure", "local-emulator", "docker", "Dockerfile")),
            "-t",
            local_image,
            str(repo_path()),
        ]
    )
    return local_image


def _wait_for_health(port: int, *, expected: int, timeout_seconds: int) -> None:
    url = f"http://127.0.0.1:{port}/health"
    last_code = "000"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == expected:
                    return
                last_code = str(response.status)
        except urllib.error.HTTPError as exc:
            last_code = str(exc.code)
            if exc.code == expected:
                return
        except Exception:
            last_code = "000"
        time.sleep(1)
    raise RuntimeError(f"{url} did not return {expected} within {timeout_seconds}s (last={last_code})")


def _probe_uid_gid(compose: list[str], reporter: SmokeReporter) -> None:
    print("--- Probe A: /data UID/GID ---")
    completed = subprocess.run(
        [*compose, "exec", "verda-local", "stat", "-c", "%u:%g", "/data"],
        check=False,
        capture_output=True,
        text=True,
    )
    uid_gid = completed.stdout.strip()
    if uid_gid == "1000:1000":
        reporter.pass_probe("A", "/data UID:GID = 1000:1000")
        return
    reporter.fail_probe("A", f"/data UID:GID = '{uid_gid}' (expected 1000:1000)")


def _probe_non_root_write(compose: list[str], reporter: SmokeReporter) -> None:
    print("--- Probe B: non-root write ---")
    completed = subprocess.run(
        [
            *compose,
            "exec",
            "-u",
            "verda",
            "verda-local",
            "sh",
            "-c",
            "touch /data/.probe && rm /data/.probe",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        reporter.pass_probe("B", "non-root write to /data OK")
        return
    reporter.fail_probe("B", "non-root write to /data failed")


def _probe_sigterm_latency(compose: list[str], reporter: SmokeReporter, *, timeout_seconds: int) -> None:
    print("--- Probe C: SIGTERM latency ---")
    start = time.time()
    subprocess.run([*compose, "kill", "-s", "TERM"], check=False, capture_output=True, text=True)
    deadline = time.time() + timeout_seconds + 5
    exited = False
    while time.time() < deadline:
        completed = subprocess.run(
            [*compose, "ps", "--status", "exited", "-q"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.stdout.strip():
            exited = True
            break
        time.sleep(1)
    latency_ms = int((time.time() - start) * 1000)
    budget_ms = timeout_seconds * 1000
    if exited and latency_ms <= budget_ms:
        reporter.pass_probe("C", f"SIGTERM latency {latency_ms}ms (<={budget_ms}ms)")
        return
    reporter.fail_probe("C", f"SIGTERM latency {latency_ms}ms exceeds {budget_ms}ms")


def _probe_env_leak(compose: list[str], reporter: SmokeReporter) -> None:
    print("--- Probe D: trust-zone leak ---")
    completed = subprocess.run(
        [*compose, "exec", "verda-local", "env"],
        check=False,
        capture_output=True,
        text=True,
    )
    leaks = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.startswith("VERDA_CLIENT_ID=") or line.startswith("VERDA_CLIENT_SECRET=")
    ]
    if not leaks:
        reporter.pass_probe("D", "no VERDA_CLIENT_* in container env")
        return
    reporter.fail_probe("D", f"LEAK: {'; '.join(leaks)}")


def _probe_degraded_gating(runtime_env: dict[str, str], reporter: SmokeReporter, settings: SmokeConfig) -> None:
    print("--- Probe E: degraded gating ---")
    with tempfile.TemporaryDirectory(prefix="gpupoor-smoke.") as temp_dir:
        strict_override = Path(temp_dir) / "strict.override.yml"
        strict_override.write_text(
            "\n".join(
                [
                    "services:",
                    "  verda-local:",
                    "    environment:",
                    '      ALLOW_DEGRADED: "0"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        strict_compose = _compose_command(cpu=True, extra_files=[strict_override])
        degraded_compose = _compose_command(cpu=True)

        strict_env = {"APP_PORT": str(settings.strict_port), **runtime_env}
        degraded_env = {"APP_PORT": str(settings.degraded_port), **runtime_env}
        try:
            run_command([*strict_compose, "up", "-d", "--build"], env=strict_env)
            _wait_for_health(settings.strict_port, expected=503, timeout_seconds=settings.health_timeout_seconds)
            run_command([*strict_compose, "down", "-v", "--remove-orphans"], check=False, env=strict_env)

            run_command([*degraded_compose, "up", "-d", "--build"], env=degraded_env)
            _wait_for_health(
                settings.degraded_port,
                expected=200,
                timeout_seconds=settings.health_timeout_seconds,
            )
            reporter.pass_probe("E", "strict->503, degraded->200")
        except RuntimeError as exc:
            reporter.fail_probe("E", str(exc))
        finally:
            run_command([*strict_compose, "down", "-v", "--remove-orphans"], check=False, env=strict_env)
            run_command([*degraded_compose, "down", "-v", "--remove-orphans"], check=False, env=degraded_env)


def _probe_data_wait_timeout(local_image: str, reporter: SmokeReporter, *, timeout_seconds: int) -> None:
    print("--- Probe F: /data wait timeout ---")
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-e",
            f"WAIT_DATA_TIMEOUT={timeout_seconds}",
            "--mount",
            "type=bind,source=/tmp,target=/data,readonly",
            local_image,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        reporter.pass_probe("F", f"/data timeout exits non-zero (code={completed.returncode})")
        return
    reporter.fail_probe("F", "/data timeout did not exit non-zero")
