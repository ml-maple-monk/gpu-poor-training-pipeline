"""Regression checks for contributor guardrails."""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_pyproject_registers_guardrail_extras_and_markers() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    extras = data["project"]["optional-dependencies"]
    assert {"quality", "test", "dev"} <= set(extras)
    assert any(dep.startswith("ruff") for dep in extras["quality"])
    assert any(dep.startswith("pytest") for dep in extras["test"])

    markers = data["tool"]["pytest"]["ini_options"]["markers"]
    for marker in ("slow", "docker", "live_dashboard", "remote"):
        assert any(entry.startswith(f"{marker}:") for entry in markers)

    extend_select = set(data["tool"]["ruff"]["lint"]["extend-select"])
    assert {"B007", "B905", "PIE810", "RUF046", "RUF059", "SIM102", "SIM115", "SIM117"} <= extend_select


def test_makefile_exposes_required_guardrail_commands() -> None:
    text = _read("Makefile")

    for target in (
        "install-dev",
        "format-check",
        "lint",
        "lint-fix",
        "style-check",
        "test-fast",
        "test-live",
        "ci-local",
    ):
        assert f"{target}:" in text

    for removed_target in (
        "format:",
        "test:",
        "test-integration:",
        "mutants:",
        "mutants-report:",
        "mutants-baseline:",
    ):
        assert removed_target not in text

    assert "pre_commit install --install-hooks" in text
    assert "pre_commit run --all-files --show-diff-on-failure" in text
    assert "--cov=src/gpupoor" in text
    assert "--cov=training/src/minimind" in text
    assert "--cov=infrastructure/dashboard/src" in text
    assert "not live_dashboard and not docker and not remote and not slow" in text


def test_precommit_and_workflows_pin_required_guardrails() -> None:
    precommit = _read(".pre-commit-config.yaml")
    assert "ruff-pre-commit" in precommit
    assert "training/src/minimind" in precommit
    assert "infrastructure/dashboard/src" in precommit
    assert "verda_remote_dry_run\\.yaml" in precommit
    assert "tokenizer(?:_config)?\\.json" in precommit

    quality = _read(".github/workflows/quality.yml")
    tests = _read(".github/workflows/tests.yml")
    live = _read(".github/workflows/live-checks.yml")

    assert "name: quality" in quality
    assert "name: tests" in tests
    assert "branches:" in quality and "- master" in quality
    assert "branches:" in tests and "- master" in tests
    assert "workflow_dispatch:" in quality
    assert "workflow_dispatch:" in tests
    assert "make style-check" in quality
    assert "torch==2.10.0+cpu" in tests
    assert 'cron: "0 14 * * 1"' in live


def test_contributing_documents_required_checks() -> None:
    text = _read("CONTRIBUTING.md")

    assert "quality" in text
    assert "tests" in text
    assert "make style-check" in text
    assert "pre-commit" in text
    assert "make test-fast" in text
    assert "make test-live" in text
