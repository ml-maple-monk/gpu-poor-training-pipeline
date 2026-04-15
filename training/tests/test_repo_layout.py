"""Layout regression tests for the streamlined repo structure."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "parts",
    [
        ("training", "start.sh"),
        ("dstack", "start.sh"),
        ("infrastructure", "dashboard", "start.sh"),
        ("infrastructure", "mlflow", "start.sh"),
        ("infrastructure", "local-emulator", "start.sh"),
    ],
)
def test_each_subsystem_has_one_top_level_start_script(repo_path, parts):
    assert repo_path(*parts).is_file()


def test_infrastructure_shape_is_explicit(repo_path):
    infra_dirs = {path.name for path in repo_path("infrastructure").iterdir() if path.is_dir()}
    assert infra_dirs == {"dashboard", "local-emulator", "mlflow"}


@pytest.mark.parametrize(
    "parts",
    [
        ("dashboard",),
        ("emulator",),
    ],
)
def test_old_top_level_local_infra_dirs_are_gone(repo_path, parts):
    assert not repo_path(*parts).exists()


def test_training_source_is_repo_owned(minimind_root):
    assert minimind_root.joinpath("model", "model_minimind.py").is_file()
    assert minimind_root.joinpath("trainer", "train_pretrain.py").is_file()


@pytest.mark.parametrize(
    "parts",
    [
        ("training", "setup-minimind.sh"),
        ("training", "build-and-push.sh"),
        ("training", "run-train.sh"),
    ],
)
def test_old_clone_based_training_entrypoints_are_gone(repo_path, parts):
    assert not repo_path(*parts).exists()


@pytest.mark.parametrize(
    "parts",
    [
        ("training", "docker", "Dockerfile.base"),
        ("training", "config", "requirements.train.base.txt"),
        ("training", "scripts", "build-base-image.sh"),
    ],
)
def test_training_base_image_artifacts_exist(repo_path, parts):
    assert repo_path(*parts).is_file()


def test_training_start_script_exposes_base_build(repo_text, repo_relpath):
    start_script = repo_text("training", "start.sh")

    assert f"{repo_relpath('training', 'start.sh')} build-base" in start_script
    assert "build-base)" in start_script


def test_remote_training_image_uses_the_shared_base(repo_text, repo_relpath):
    dockerfile = repo_text("training", "docker", "Dockerfile.remote")
    build_script = repo_text("training", "scripts", "build-and-push.sh")

    assert "ARG BASE_IMAGE=" in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert repo_relpath("training", "docker", "Dockerfile.base") in build_script
    assert "--build-arg BASE_IMAGE=" in build_script


@pytest.mark.parametrize(
    ("parts", "should_exist"),
    [
        (("Makefile",), True),
        (("PARITY.md",), False),
        (("REPO_REVIEW.md",), False),
        (("PR_DRAFT.md",), False),
    ],
)
def test_top_level_artifacts_match_current_guardrail_contract(repo_path, parts, should_exist):
    assert repo_path(*parts).exists() is should_exist


def test_dashboard_no_longer_owns_mlflow_assets(repo_path, repo_text):
    dashboard_root = ("infrastructure", "dashboard")
    start_script = repo_text(*dashboard_root, "start.sh")

    assert not repo_path(*dashboard_root, "compose", "docker-compose.mlflow.yml").exists()
    assert "tunnel)" not in start_script
    assert "mlflow)" not in start_script
    assert "mlflow-up" not in repo_text(*dashboard_root, "docs", "README.md")


@pytest.mark.parametrize(
    ("parts", "expected_prefix"),
    [
        (("infrastructure", "dashboard", "compose", "docker-compose.yml"), "name: verda-dashboard\n"),
        (("infrastructure", "local-emulator", "compose", "docker-compose.yml"), "name: verda-local-emulator\n"),
        (("training", "compose", "docker-compose.train.yml"), "name: minimind-training\n"),
    ],
)
def test_each_compose_stack_has_a_distinct_project_name(repo_text, parts, expected_prefix):
    assert repo_text(*parts).startswith(expected_prefix)


def test_training_python_uses_simple_parent_path_hierarchy(repo_path):
    banned_fragments = (
        "resolve" + "()",
        ".parents" + "[",
        "from conftest import " + "REPO_ROOT",
        "from conftest import " + "MINIMIND_ROOT",
        "from conftest import " + "MINIMIND_AVAILABLE",
        "TESTS_" + "ROOT =",
        "TRAINING_" + "ROOT =",
    )

    for path in repo_path("training").rglob("*.py"):
        if any(part in {".venv", "__pycache__"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        for fragment in banned_fragments:
            assert fragment not in text, (
                f"{path} uses banned path-resolution style {fragment!r}; "
                "use explicit chained .parent hierarchy instead"
            )


def test_training_tests_use_shared_repo_helpers(repo_path):
    banned_fragments = (
        "Path(" + "__file__" + ")",
        "sys.path." + "insert(",
        "spec_from_file_" + "location(",
        "REPO_" + "ROOT =",
        "MINIMIND_" + "ROOT =",
        "MINIMIND_" + "AVAILABLE =",
        "METRICS_" + "PATH =",
        "HELPER_" + "PATH =",
        "ARGS_" + "HELPER =",
    )

    for path in repo_path("training", "tests").glob("*.py"):
        if path.name == "conftest.py":
            continue
        text = path.read_text(encoding="utf-8")
        for fragment in banned_fragments:
            assert fragment not in text, (
                f"{path} should use shared pytest fixtures instead of local path bootstrapping {fragment!r}"
            )
