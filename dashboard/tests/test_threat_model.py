"""test_threat_model.py — Verify threat model documentation and safe_exec structure."""

from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

README = Path(__file__).parent.parent / "README.md"
SRC_SAFE_EXEC = Path(__file__).parent.parent / "src" / "safe_exec.py"


def test_readme_has_threat_model_section():
    """README must have ## Threat Model section."""
    text = README.read_text()
    assert "## Threat Model" in text, "README missing '## Threat Model' section"


def test_readme_names_docker_sock():
    """Threat Model section must mention docker.sock."""
    text = README.read_text()
    tm_start = text.index("## Threat Model")
    # Find next ## heading or end
    next_h2 = text.find("\n## ", tm_start + 1)
    section = text[tm_start:next_h2] if next_h2 != -1 else text[tm_start:]
    assert "docker.sock" in section, "Threat Model must mention 'docker.sock'"


def test_readme_names_argv_whitelist():
    """Threat Model section must mention argv-whitelist."""
    text = README.read_text()
    assert "argv-whitelist" in text, "README must mention 'argv-whitelist'"


def test_readme_names_rest_whitelist():
    """Threat Model section must mention REST-whitelist."""
    text = README.read_text()
    assert "REST-whitelist" in text, "README must mention 'REST-whitelist'"


def test_safe_exec_exposes_allowed_verbs():
    """safe_exec.py must export ALLOWED_VERBS as a frozenset."""
    from src.safe_exec import ALLOWED_VERBS
    assert isinstance(ALLOWED_VERBS, frozenset), "ALLOWED_VERBS must be a frozenset"
    assert "logs" in ALLOWED_VERBS
    assert "ps" in ALLOWED_VERBS
    assert "inspect" in ALLOWED_VERBS
    # Must NOT contain mutating verbs
    assert "stop" not in ALLOWED_VERBS
    assert "kill" not in ALLOWED_VERBS
    assert "rm" not in ALLOWED_VERBS


def test_safe_exec_exposes_allowed_endpoints():
    """safe_exec.py must export ALLOWED_ENDPOINTS as a frozenset."""
    from src.safe_exec import ALLOWED_ENDPOINTS
    assert isinstance(ALLOWED_ENDPOINTS, frozenset), "ALLOWED_ENDPOINTS must be a frozenset"
    assert "runs/get_plan" in ALLOWED_ENDPOINTS
    assert "runs/list" in ALLOWED_ENDPOINTS
    assert "runs/get_logs" in ALLOWED_ENDPOINTS
    # Must NOT contain mutating endpoints
    assert "runs/stop" not in ALLOWED_ENDPOINTS
    assert "runs/delete" not in ALLOWED_ENDPOINTS
    assert "runs/apply" not in ALLOWED_ENDPOINTS


def test_safe_docker_blocks_bad_verb():
    """safe_docker must raise ValueError on disallowed verb."""
    from src.safe_exec import safe_docker
    with pytest.raises(ValueError, match="not in whitelist"):
        safe_docker(["stop", "my-container"])


def test_safe_docker_blocks_kill():
    from src.safe_exec import safe_docker
    with pytest.raises(ValueError):
        safe_docker(["kill", "my-container"])


def test_safe_docker_blocks_rm():
    from src.safe_exec import safe_docker
    with pytest.raises(ValueError):
        safe_docker(["rm", "-f", "my-container"])


def test_safe_dstack_rest_blocks_bad_endpoint():
    """safe_dstack_rest must raise ValueError on disallowed endpoint."""
    from src.safe_exec import safe_dstack_rest
    with pytest.raises(ValueError, match="not in whitelist"):
        safe_dstack_rest("runs/stop")


def test_safe_dstack_rest_blocks_users():
    from src.safe_exec import safe_dstack_rest
    with pytest.raises(ValueError):
        safe_dstack_rest("users/list")
