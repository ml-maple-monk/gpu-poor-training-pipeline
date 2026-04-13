"""Package-owned maintenance helpers that replace top-level scripts."""

from __future__ import annotations

import base64
import os
from pathlib import Path
import re
import secrets as py_secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from gpupoor.config import DoctorConfig, RemoteConfig, SmokeConfig, find_dstack_bin, load_remote_settings
from gpupoor.paths import repo_path, repo_root
from gpupoor.subprocess_utils import run_command


ANCHOR_PATTERN = re.compile(r"doc-anchor:\s*([\w-]+)")
CLIENT_ID_PATTERN = re.compile(r"^[Cc]lien[dt]\s*ID\s*:\s*(.+)$", re.MULTILINE)
SECRET_PATTERN = re.compile(r"^[Ss]ecret:\s*(.+)$", re.MULTILINE)


class PreflightReporter:
    """Collects warnings and failures while preserving shell-like output."""

    def __init__(self) -> None:
        self.failed = False

    def fail(self, message: str) -> None:
        print(f"PREFLIGHT FAIL: {message}", file=sys.stderr)
        self.failed = True

    def warn(self, message: str) -> None:
        print(f"PREFLIGHT WARN: {message}", file=sys.stderr)


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


def _mode_octal(path: Path) -> str:
    return oct(path.stat().st_mode & 0o777)[2:]


def _read_windows_utc_timestamp() -> str:
    return subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-Command", "[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _absolute_delta(lhs: int, rhs: int) -> int:
    return abs(lhs - rhs)


def _write_mode_600(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def parse_secrets_payload(payload: str) -> tuple[str, str]:
    secret_match = SECRET_PATTERN.search(payload)
    client_match = CLIENT_ID_PATTERN.search(payload)
    secret = secret_match.group(1).strip() if secret_match else ""
    client_id = client_match.group(1).strip() if client_match else ""
    if not secret:
        raise RuntimeError("could not parse Secret from the secrets file")
    if not client_id:
        raise RuntimeError("could not parse ClientID/CliendID from the secrets file")
    return client_id, secret


def parse_secrets(secrets_file: str | Path | None = None, *, output_dir: str | Path | None = None) -> None:
    source = repo_path("secrets") if secrets_file is None else Path(secrets_file).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"secrets file not found: {source}")

    client_id, secret = parse_secrets_payload(source.read_text(encoding="utf-8"))
    target_dir = Path(output_dir or repo_root()).resolve()
    inference_token = f"local-dev-{py_secrets.token_urlsafe(16)}"

    _write_mode_600(
        target_dir / ".env.inference",
        "\n".join(
            [
                "# LOCAL-ONLY bearer token for /infer — NOT a Verda-issued token.",
                "# Delete this file and re-run gpupoor parse-secrets to rotate.",
                f"VERDA_INFERENCE_TOKEN={inference_token}",
                "",
            ]
        ),
    )
    _write_mode_600(
        target_dir / ".env.mgmt",
        "\n".join(
            [
                f"VERDA_CLIENT_ID={client_id}",
                f"VERDA_CLIENT_SECRET={secret}",
                "",
            ]
        ),
    )
    print("Written: .env.inference (mode 600)")
    print("Written: .env.mgmt (mode 600)")


def collect_doc_anchors(paths: list[Path]) -> set[str]:
    anchors: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            candidates = (candidate for candidate in path.rglob("*") if candidate.is_file())
        else:
            candidates = [path]
        for candidate in candidates:
            try:
                text = candidate.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            anchors.update(ANCHOR_PATTERN.findall(text))
    return anchors


def check_doc_anchors(*, root: str | Path | None = None) -> None:
    base = Path(root or repo_root()).resolve()
    defined = collect_doc_anchors(
        [
            base / "infrastructure",
            base / "training",
            base / "dstack",
            base / "src",
        ]
    )
    referenced = collect_doc_anchors(
        [
            base / "README.md",
            base / "TROUBLESHOOTING.md",
            base / "training" / "docs" / "README.md",
            base / "infrastructure" / "dashboard" / "docs" / "README.md",
            base / "infrastructure" / "local-emulator" / "docs" / "README.md",
            base / "infrastructure" / "mlflow" / "docs" / "README.md",
            base / "dstack" / "docs" / "README.md",
        ]
    )
    if not referenced:
        print("[anchor-check] No anchor references found in READMEs — nothing to verify.")
        return

    missing = sorted(referenced - defined)
    if missing:
        rendered = "\n".join(f"  - {anchor}" for anchor in missing)
        raise RuntimeError(
            "[anchor-check] FAIL — unresolved anchors referenced in READMEs but not defined in source:\n"
            f"{rendered}"
        )
    print(f"[anchor-check] OK — all {len(referenced)} referenced anchors resolve to source comments.")


def _resolve_max_clock_skew(*, doctor: DoctorConfig | None = None, max_skew_seconds: int | None = None) -> int:
    if max_skew_seconds is not None:
        return max_skew_seconds
    if doctor is not None:
        return doctor.max_clock_skew_seconds
    return int(os.environ.get("MAX_CLOCK_SKEW_SECONDS", "5"))


def fix_wsl_clock(*, doctor: DoctorConfig | None = None, max_skew_seconds: int | None = None) -> None:
    limit = _resolve_max_clock_skew(doctor=doctor, max_skew_seconds=max_skew_seconds)
    if not shutil.which("powershell.exe"):
        raise RuntimeError("powershell.exe not found. This fixer only applies inside WSL2.")

    win_ts = _read_windows_utc_timestamp()
    if not win_ts:
        raise RuntimeError("Could not read the Windows UTC clock.")

    win_epoch = int(win_ts)
    linux_epoch = int(subprocess.check_output(["date", "-u", "+%s"], text=True).strip())
    skew_before = _absolute_delta(linux_epoch, win_epoch)

    print(f"[fix-wsl-clock] Windows UTC epoch: {win_epoch}")
    print(f"[fix-wsl-clock] Linux UTC epoch:   {linux_epoch}")

    if skew_before < limit:
        print(f"[fix-wsl-clock] Clock skew already healthy ({skew_before}s < {limit}s).")
        return

    print(f"[fix-wsl-clock] Clock skew is {skew_before}s; syncing Linux time from Windows UTC...")
    command = ["date", "-u", "-s", f"@{win_epoch}"]
    if os.geteuid() != 0:
        if not shutil.which("sudo"):
            raise RuntimeError(
                f"Automatic sync failed. Try: sudo date -u -s '@{win_epoch}' or, "
                "from Windows PowerShell, run wsl.exe --shutdown and reopen WSL."
            )
        command = ["sudo", *command]
    run_command(command)

    linux_epoch_after = int(subprocess.check_output(["date", "-u", "+%s"], text=True).strip())
    skew_after = _absolute_delta(linux_epoch_after, win_epoch)
    if skew_after >= limit:
        raise RuntimeError(
            f"Clock skew is still {skew_after}s after sync. "
            "From Windows PowerShell, run wsl.exe --shutdown and reopen WSL."
        )
    print(f"[fix-wsl-clock] Clock skew fixed ({skew_before}s -> {skew_after}s).")


def run_preflight(
    *,
    remote: bool = False,
    doctor: DoctorConfig | None = None,
    remote_config: RemoteConfig | None = None,
) -> None:
    should_skip = doctor.skip_preflight if doctor is not None else os.environ.get("SKIP_PREFLIGHT") == "1"
    if should_skip:
        print(
            "WARNING: preflight skipped — operator-only; CI must not set this for validation",
            file=sys.stderr,
        )
        return

    reporter = PreflightReporter()
    repo = repo_root()
    _run_local_preflight(reporter, repo, doctor=doctor)
    if reporter.failed:
        raise RuntimeError("Preflight FAILED — fix errors above before continuing")
    print("Preflight OK")

    if not remote:
        return

    _run_remote_preflight(reporter, repo, remote_config=remote_config)
    if reporter.failed:
        raise RuntimeError("Remote preflight FAILED — fix errors above before continuing")

    settings = load_remote_settings(remote_config)
    print(f"Remote preflight OK (VCR_IMAGE_BASE={settings.get('VCR_IMAGE_BASE', 'unknown')})")


def _run_local_preflight(reporter: PreflightReporter, repo: Path, *, doctor: DoctorConfig | None = None) -> None:
    if not Path("/usr/lib/wsl/lib/libcuda.so.1").is_file():
        reporter.fail("/usr/lib/wsl/lib/libcuda.so.1 not found — install Windows NVIDIA driver and enable WSL2 CUDA")

    if str(repo).startswith("/mnt/c"):
        reporter.fail(
            "project is on /mnt/c — move to a Linux path (e.g. ~/workspace) "
            "for ext4 performance and correct file permissions"
        )

    if not shutil.which("nvidia-smi"):
        reporter.fail("nvidia-smi not found — install nvidia-container-toolkit")
    else:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            output = ""
        if not output.strip():
            reporter.fail("nvidia-smi found but reports no GPUs")

    compose_file = repo_path("infrastructure", "local-emulator", "compose", "docker-compose.yml")
    try:
        compose_config = subprocess.check_output(
            ["docker", "compose", "-f", str(compose_file), "config"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        compose_config = ""
    if "driver: nvidia" not in compose_config:
        reporter.fail(
            "docker compose config does not show nvidia GPU reservation — check "
            "infrastructure/local-emulator/compose/docker-compose.yml and nvidia-container-toolkit"
        )

    for relative in (".env.inference", ".env.mgmt"):
        path = repo / relative
        if not path.is_file():
            reporter.fail(f"{relative} not found — run: gpupoor parse-secrets")
            continue
        mode = _mode_octal(path)
        if mode != "600":
            reporter.fail(f"{relative} has mode {mode} — must be 600 (run: chmod 600 {relative})")

    if shutil.which("powershell.exe"):
        try:
            win_ts = _read_windows_utc_timestamp()
        except subprocess.CalledProcessError:
            win_ts = ""
        if win_ts:
            linux_ts = int(subprocess.check_output(["date", "-u", "+%s"], text=True).strip())
            skew = _absolute_delta(linux_ts, int(win_ts))
            max_skew_seconds = _resolve_max_clock_skew(doctor=doctor)
            if skew >= max_skew_seconds:
                reporter.fail(
                    f"clock skew {skew}s between WSL2 and Windows (must be < {max_skew_seconds}s) "
                    "— run: ./run.sh fix-clock"
                )
        else:
            reporter.warn("could not read Windows clock for skew check")
    else:
        reporter.warn("powershell.exe not found — skipping clock skew check (non-WSL2 host?)")

    wsl_conf = Path("/etc/wsl.conf")
    if wsl_conf.is_file():
        text = wsl_conf.read_text(encoding="utf-8", errors="ignore")
        if "systemd=true" not in text:
            reporter.warn("/etc/wsl.conf does not contain systemd=true — some features may not work")
    else:
        reporter.warn("/etc/wsl.conf not found — consider adding [boot] systemd=true")


def _run_remote_preflight(
    reporter: PreflightReporter,
    repo: Path,
    *,
    remote_config: RemoteConfig | None = None,
) -> None:
    for binary in ("docker", "cloudflared", "rsync", "curl"):
        if shutil.which(binary):
            continue
        if binary == "cloudflared":
            reporter.fail(
                "cloudflared not found — install: curl -L "
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
                "-o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"
            )
        else:
            reporter.fail(f"{binary} not found — install via apt or your package manager")

    try:
        find_dstack_bin()
    except RuntimeError:
        reporter.fail("no working dstack CLI found — install the isolated uv venv described in dstack/docs/README.md")

    if not shutil.which("jq"):
        reporter.warn("jq not found — python3 fallback will be used for JSON parsing (safe, but slower)")

    for relative in ("hf_token", "secrets"):
        path = repo / relative
        if not path.is_file():
            if relative == "hf_token":
                reporter.fail(f"{relative} not found — save your Hugging Face token to {path} (chmod 600)")
            else:
                reporter.fail(f"{relative} not found — save your Verda credentials to {path} (chmod 600)")
            continue
        mode = _mode_octal(path)
        if mode != "600":
            reporter.fail(f"{path} has mode {mode} — must be 600: run: chmod 600 {path}")

    remote_env_name = remote_config.env_file if remote_config is not None else ".env.remote"
    remote_env_path = repo / remote_env_name
    if remote_env_path.is_file():
        mode = _mode_octal(remote_env_path)
        if mode != "600":
            reporter.fail(f"{remote_env_path} has mode {mode} — must be 600: run: chmod 600 {remote_env_path}")

    settings = load_remote_settings(remote_config)
    if not settings.get("VCR_USERNAME") or not settings.get("VCR_PASSWORD"):
        reporter.fail(f"VCR_USERNAME/VCR_PASSWORD missing — export them or create {remote_env_path} with mode 600")

    if settings.get("PUSH_GHCR") == "1":
        gh_token = repo / "gh_token"
        if not gh_token.is_file():
            reporter.fail(
                f"PUSH_GHCR=1 but {gh_token} is missing — GHCR fallback needs a write:packages token"
            )
    elif (repo / "gh_token").is_file():
        reporter.warn("gh_token is not used by the default remote path anymore; GHCR is optional fallback only")


def read_env_value(path: Path, key: str) -> str:
    if not path.is_file():
        return ""
    prefix = f"{key}="
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith(prefix):
            return raw_line[len(prefix) :].strip()
    return ""


def collect_leak_scan_secrets(*, root: str | Path | None = None) -> list[str]:
    base = Path(root or repo_root()).resolve()
    secrets: list[str] = []
    inference_token = read_env_value(base / ".env.inference", "VERDA_INFERENCE_TOKEN")
    management_secret = read_env_value(base / ".env.mgmt", "VERDA_CLIENT_SECRET")
    if inference_token:
        secrets.append(inference_token)
    if management_secret:
        secrets.append(management_secret)
    return secrets


def detect_secret_leaks(output: str, secret_values: list[str]) -> list[str]:
    findings: list[str] = []

    def add_finding(message: str) -> None:
        if message not in findings:
            findings.append(message)

    for secret in secret_values:
        if not secret:
            continue
        b64_value = base64.b64encode(secret.encode("utf-8")).decode("ascii")
        if secret in output:
            add_finding("literal secret value found in image layers")
        if b64_value in output:
            add_finding("base64-encoded secret value found in image layers")
    if "VERDA_CLIENT_SECRET=" in output:
        add_finding("VERDA_CLIENT_SECRET= found in image layers")
    return findings


def resolve_image_name(image: str) -> str:
    output = subprocess.check_output(
        ["docker", "images", image, "--format", "{{.Repository}}:{{.Tag}}"],
        text=True,
    )
    names = [line.strip() for line in output.splitlines() if line.strip()]
    if not names:
        raise RuntimeError(
            f"no image matching '{image}' found — build the local emulator image first or run gpupoor smoke"
        )
    return names[0]


def _scan_output(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return (completed.stdout or "") + (completed.stderr or "")


def leak_scan(image: str = "verda-local", *, canary: bool = False) -> None:
    full_image = resolve_image_name(image)
    print(f"Scanning image: {full_image}")

    if shutil.which("dive"):
        print("Using: dive")
        output = _scan_output(["dive", full_image, "--ci"])
    elif shutil.which("syft"):
        print("Using: syft")
        output = _scan_output(["syft", full_image])
    else:
        print("Using: docker history (fallback)")
        output = _scan_output(["docker", "history", "--no-trunc", "--format", "{{.CreatedBy}}", full_image])

    findings = detect_secret_leaks(output, collect_leak_scan_secrets())
    for finding in findings:
        print(f"LEAK DETECTED: {finding}", file=sys.stderr)
    if canary:
        _run_canary_self_test()
    if findings:
        raise RuntimeError("Leak scan FAILED")
    print("Leak scan PASSED — no secrets found in image layers")


def _run_canary_self_test() -> None:
    print("--- CANARY self-test ---")
    canary_tag = "verda-local:canary-EPHEMERAL"
    with tempfile.TemporaryDirectory(prefix="verda-canary-build.") as temp_dir:
        dockerfile = Path(temp_dir) / "Dockerfile"
        dockerfile.write_text(
            "\n".join(
                [
                    "FROM busybox:1.36",
                    "ARG VERDA_FAKE_CANARY=unset",
                    'RUN echo "VERDA_FAKE_CANARY=${VERDA_FAKE_CANARY}"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        try:
            run_command(
                [
                    "docker",
                    "build",
                    "--no-cache",
                    "--build-arg",
                    "VERDA_FAKE_CANARY=PLANTED123",
                    "-t",
                    canary_tag,
                    temp_dir,
                ]
            )
            history = subprocess.check_output(
                ["docker", "history", "--no-trunc", "--format", "{{.CreatedBy}}", canary_tag],
                text=True,
            )
            if "PLANTED123" not in history:
                raise RuntimeError("CANARY self-test FAIL: canary value NOT detected — scanner may be unreliable")
            print("CANARY self-test PASS: canary value detected in layers")
        finally:
            run_command(["docker", "rmi", canary_tag], check=False)


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
        [*compose, "exec", "-u", "verda", "verda-local", "sh", "-c", "touch /data/.probe && rm /data/.probe"],
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
            _wait_for_health(settings.degraded_port, expected=200, timeout_seconds=settings.health_timeout_seconds)
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
