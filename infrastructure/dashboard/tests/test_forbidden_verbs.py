"""test_forbidden_verbs.py — CI lint: no mutating docker/dstack/REST calls in src/.

Greps infrastructure/dashboard/src/ (excluding tests/) for:
- Mutating docker CLI verbs: stop, kill, rm, delete, apply (as subprocess args)
- Mutating dstack REST paths: /api/runs/stop, /api/runs/delete, /api/runs/apply,
  /api/users/
- Mutating HTTP method patterns outside the whitelist constants
"""

from __future__ import annotations

import re
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / "src"

# ── Patterns that must NOT appear outside whitelist constants ─────────────────

# Docker mutating verbs in subprocess args (string literals containing these words
# as standalone tokens)
FORBIDDEN_DOCKER_VERBS = re.compile(r"\b(stop|kill|rm|delete|apply|run|up|down|push)\b")

# dstack REST mutation paths
FORBIDDEN_REST_PATHS = re.compile(
    r"/api/runs/(stop|delete|apply|push|kill)"
    r"|/api/users/"
    r"|/api/projects/.*(delete|create|update)"
)

# Mutating HTTP methods used directly (not inside safe_dstack_rest or safe_docker)
FORBIDDEN_HTTP_CALLS = re.compile(
    r"requests\.(post|put|delete|patch)\s*\("
    r"|httpx\.(post|put|delete|patch)\s*\("
)

# subprocess patterns with mutating docker verbs
FORBIDDEN_SUBPROCESS = re.compile(
    r"subprocess\.(run|Popen|call|check_call|check_output).*"
    r'["\']docker["\'].*["\'](?:stop|kill|rm|delete|apply|run|push)["\']'
)


def _iter_py_files():
    for path in SRC_DIR.rglob("*.py"):
        if "test" in path.parts:
            continue
        yield path


def _file_lines(path: Path) -> list[tuple[int, str]]:
    with open(path) as f:
        return list(enumerate(f.readlines(), start=1))


def test_no_forbidden_rest_paths():
    """No mutating dstack REST paths in source."""
    violations = []
    for path in _iter_py_files():
        for lineno, line in _file_lines(path):
            if FORBIDDEN_REST_PATHS.search(line):
                # Allow inside ALLOWED_ENDPOINTS definition
                if "ALLOWED_ENDPOINTS" in line:
                    continue
                violations.append(f"{path.relative_to(SRC_DIR.parent)}:{lineno}: {line.rstrip()}")
    assert not violations, "Forbidden REST mutation paths found:\n" + "\n".join(violations)


def test_no_forbidden_http_methods():
    """No direct mutating HTTP calls (requests.post, httpx.post etc.) except inside whitelisted files.

    safe_exec.py: uses httpx for dstack REST (whitelisted endpoints only).
    mlflow_client.py: uses requests.post for MLflow search API (POST is the MLflow API convention,
      endpoints are read-only searches, not mutations).
    """
    _ALLOWED_FILES = {"safe_exec.py", "mlflow_client.py"}
    violations = []
    for path in _iter_py_files():
        if path.name in _ALLOWED_FILES:
            continue
        for lineno, line in _file_lines(path):
            # Allow requests.get and httpx.get
            if re.search(r"requests\.(get|head)\s*\(", line):
                continue
            if FORBIDDEN_HTTP_CALLS.search(line):
                violations.append(f"{path.relative_to(SRC_DIR.parent)}:{lineno}: {line.rstrip()}")
    assert not violations, "Forbidden HTTP mutation calls found:\n" + "\n".join(violations)


def test_no_write_calls():
    """No file write operations outside tmpfs paths."""
    forbidden = re.compile(
        r'\bopen\s*\(.*["\']w["\']'
        r'|\bopen\s*\(.*["\']a["\']'
        r"|\bos\.remove\s*\("
        r"|\bshutil\.rmtree\s*\("
        r"|\bshutil\.move\s*\("
    )
    violations = []
    for path in _iter_py_files():
        for lineno, line in _file_lines(path):
            if forbidden.search(line):
                violations.append(f"{path.relative_to(SRC_DIR.parent)}:{lineno}: {line.rstrip()}")
    assert not violations, "Forbidden file write operations found:\n" + "\n".join(violations)


def test_no_docker_mutating_subprocess():
    """No subprocess calls with mutating docker verbs as args."""
    # More targeted: look for subprocess with 'docker' AND a mutating verb
    forbidden = re.compile(r'["\'](?:stop|kill|rm|delete|apply|push|run|up|down)["\']')
    violations = []
    for path in _iter_py_files():
        # safe_exec.py contains ALLOWED_VERBS — skip it
        if path.name == "safe_exec.py":
            continue
        in_subprocess_context = False
        for lineno, line in _file_lines(path):
            if "subprocess" in line or "Popen" in line:
                in_subprocess_context = True
            if in_subprocess_context and "docker" in line.lower():
                if forbidden.search(line):
                    violations.append(f"{path.relative_to(SRC_DIR.parent)}:{lineno}: {line.rstrip()}")
                in_subprocess_context = False
    assert not violations, "Forbidden docker subprocess calls:\n" + "\n".join(violations)
