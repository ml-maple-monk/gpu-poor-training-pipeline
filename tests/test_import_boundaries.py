"""Dependency-free import boundary guardrails."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_files(*parts: str) -> list[Path]:
    root = REPO_ROOT.joinpath(*parts)
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.py"))


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _assert_forbidden_imports(paths: list[Path], forbidden_prefixes: tuple[str, ...], rule: str) -> None:
    failures: list[str] = []
    for path in paths:
        for imported in sorted(_imports(path)):
            if any(
                imported == prefix
                or imported.startswith(f"{prefix}.")
                or (prefix.endswith("_") and imported.startswith(prefix))
                for prefix in forbidden_prefixes
            ):
                failures.append(f"{path.relative_to(REPO_ROOT)}: forbidden import {imported} violates {rule}")
    assert not failures, "\n".join(failures)


def test_orchestration_modules_do_not_import_concrete_minimind_trainer() -> None:
    paths = [
        *_python_files("src", "gpupoor", "backends"),
        *_python_files("src", "gpupoor", "services"),
        REPO_ROOT / "src" / "gpupoor" / "deployer.py",
    ]

    _assert_forbidden_imports(
        paths,
        ("minimind.trainer",),
        "src/gpupoor orchestration cannot import concrete minimind trainer modules",
    )


def test_trainer_core_does_not_import_orchestration_or_experiment_harness() -> None:
    _assert_forbidden_imports(
        _python_files("training", "src", "minimind", "trainer", "core"),
        ("gpupoor", "dstack", "trainer.experiment_", "minimind.trainer.experiment_"),
        "trainer core cannot import orchestration, dstack, or experiment harness modules",
    )


def test_backends_do_not_import_training_heavy_dependencies() -> None:
    _assert_forbidden_imports(
        _python_files("src", "gpupoor", "backends"),
        ("torch", "transformers", "datasets"),
        "src/gpupoor/backends cannot import training-heavy dependencies",
    )
