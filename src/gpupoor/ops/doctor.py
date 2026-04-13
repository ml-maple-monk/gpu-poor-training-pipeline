"""Preflight and documentation validation helpers."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from gpupoor.config import DoctorConfig, RemoteConfig, find_dstack_bin, load_remote_settings
from gpupoor.subprocess_utils import run_command
from gpupoor.utils import repo_path, repo_root

ANCHOR_PATTERN = re.compile(r"doc-anchor:\s*([\w-]+)")


class PreflightReporter:
    """Collect warnings and failures while preserving shell-like output."""

    def __init__(self) -> None:
        self.failed = False

    def fail(self, message: str) -> None:
        print(f"PREFLIGHT FAIL: {message}", file=sys.stderr)
        self.failed = True

    def warn(self, message: str) -> None:
        print(f"PREFLIGHT WARN: {message}", file=sys.stderr)


def _mode_octal(path: Path) -> str:
    return oct(path.stat().st_mode & 0o777)[2:]


def _read_windows_utc_timestamp() -> str:
    return subprocess.check_output(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _absolute_delta(lhs: int, rhs: int) -> int:
    return abs(lhs - rhs)


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
            f"[anchor-check] FAIL — unresolved anchors referenced in READMEs but not defined in source:\n{rendered}"
        )
    print(f"[anchor-check] OK — all {len(referenced)} referenced anchors resolve to source comments.")


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
            reporter.fail(f"PUSH_GHCR=1 but {gh_token} is missing — GHCR fallback needs a write:packages token")
    elif (repo / "gh_token").is_file():
        reporter.warn("gh_token is not used by the default remote path anymore; GHCR is optional fallback only")
