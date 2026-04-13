"""Secret parsing and leak scanning helpers."""

from __future__ import annotations

import base64
from pathlib import Path
import re
import secrets as py_secrets
import shutil
import subprocess
import sys
import tempfile

from gpupoor.subprocess_utils import run_command
from gpupoor.utils import repo_path, repo_root


CLIENT_ID_PATTERN = re.compile(r"^[Cc]lien[dt]\s*ID\s*:\s*(.+)$", re.MULTILINE)
SECRET_PATTERN = re.compile(r"^[Ss]ecret:\s*(.+)$", re.MULTILINE)


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


def parse_secrets(
    secrets_file: str | Path | None = None, *, output_dir: str | Path | None = None
) -> None:
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
        output = _scan_output(
            ["docker", "history", "--no-trunc", "--format", "{{.CreatedBy}}", full_image]
        )

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
                raise RuntimeError(
                    "CANARY self-test FAIL: canary value NOT detected — scanner may be unreliable"
                )
            print("CANARY self-test PASS: canary value detected in layers")
        finally:
            run_command(["docker", "rmi", canary_tag], check=False)
