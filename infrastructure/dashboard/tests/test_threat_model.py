"""test_threat_model.py — Verify threat model documentation and safe_exec structure."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.safe_exec import ALLOWED_ENDPOINTS, ALLOWED_VERBS

README = Path(__file__).parent.parent / "docs" / "README.md"
SRC_SAFE_EXEC = Path(__file__).parent.parent / "src" / "safe_exec.py"


def _threat_model_section() -> str:
    text = README.read_text()
    assert "## Threat Model" in text, "README missing '## Threat Model' section"
    tm_start = text.index("## Threat Model")
    next_h2 = text.find("\n## ", tm_start + 1)
    return text[tm_start:next_h2] if next_h2 != -1 else text[tm_start:]


def test_dashboard_docs_and_allowlists_explain_the_read_only_contract():
    """Proves the dashboard still documents the threat boundary and that the
    exported allowlists match that documented read-only model."""
    section = _threat_model_section()

    assert "docker.sock" in section, "Threat Model must mention 'docker.sock'"
    assert "argv-whitelist" in section, "README must mention 'argv-whitelist'"
    assert "REST-whitelist" in section, "README must mention 'REST-whitelist'"
    assert isinstance(ALLOWED_VERBS, frozenset), "ALLOWED_VERBS must be a frozenset"
    assert {"logs", "ps", "inspect"}.issubset(ALLOWED_VERBS)
    assert {"stop", "kill", "rm"}.isdisjoint(ALLOWED_VERBS)
    assert isinstance(ALLOWED_ENDPOINTS, frozenset), "ALLOWED_ENDPOINTS must be a frozenset"
    assert {"runs/get_plan", "runs/list", "runs/get_logs"}.issubset(ALLOWED_ENDPOINTS)
    assert {"runs/stop", "runs/delete", "runs/apply"}.isdisjoint(ALLOWED_ENDPOINTS)


@pytest.mark.parametrize("argv", [["stop", "my-container"], ["kill", "my-container"], ["rm", "-f", "my-container"]])
def test_safe_docker_rejects_mutating_verbs(argv):
    """Proves the docker wrapper blocks mutating verbs, so the dashboard cannot
    escalate from observability into a container control plane."""
    from src.safe_exec import safe_docker

    with pytest.raises(ValueError, match="not in whitelist"):
        safe_docker(argv)


@pytest.mark.parametrize("endpoint", ["runs/stop", "users/list"])
def test_safe_dstack_rest_rejects_non_allowlisted_endpoints(endpoint):
    """Proves the dstack REST wrapper blocks endpoints outside the read-only
    contract, so the dashboard cannot mutate remote runs."""
    from src.safe_exec import safe_dstack_rest

    with pytest.raises(ValueError, match="not in whitelist"):
        safe_dstack_rest(endpoint)
