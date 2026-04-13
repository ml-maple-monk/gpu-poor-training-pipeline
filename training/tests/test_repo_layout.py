"""Layout regression tests for the streamlined repo structure."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_each_subsystem_has_one_top_level_start_script():
    assert (REPO_ROOT / "training" / "start.sh").is_file()
    assert (REPO_ROOT / "dstack" / "start.sh").is_file()
    assert (REPO_ROOT / "infrastructure" / "dashboard" / "start.sh").is_file()
    assert (REPO_ROOT / "infrastructure" / "mlflow" / "start.sh").is_file()
    assert (REPO_ROOT / "infrastructure" / "local-emulator" / "start.sh").is_file()


def test_infrastructure_shape_is_explicit():
    infra_dirs = {path.name for path in (REPO_ROOT / "infrastructure").iterdir() if path.is_dir()}
    assert infra_dirs == {"dashboard", "local-emulator", "mlflow"}


def test_old_top_level_local_infra_dirs_are_gone():
    assert not (REPO_ROOT / "dashboard").exists()
    assert not (REPO_ROOT / "emulator").exists()


def test_training_source_is_repo_owned():
    assert (REPO_ROOT / "training" / "src" / "minimind" / "model" / "model_minimind.py").is_file()
    assert (REPO_ROOT / "training" / "src" / "minimind" / "trainer" / "train_pretrain.py").is_file()


def test_old_clone_based_training_entrypoints_are_gone():
    assert not (REPO_ROOT / "training" / "setup-minimind.sh").exists()
    assert not (REPO_ROOT / "training" / "build-and-push.sh").exists()
    assert not (REPO_ROOT / "training" / "run-train.sh").exists()


def test_top_level_artifacts_match_current_guardrail_contract():
    assert (REPO_ROOT / "Makefile").exists()
    assert not (REPO_ROOT / "PARITY.md").exists()
    assert not (REPO_ROOT / "REPO_REVIEW.md").exists()
    assert not (REPO_ROOT / "PR_DRAFT.md").exists()


def test_dashboard_no_longer_owns_mlflow_assets():
    dashboard_root = REPO_ROOT / "infrastructure" / "dashboard"
    start_script = (dashboard_root / "start.sh").read_text()
    assert not (dashboard_root / "compose" / "docker-compose.mlflow.yml").exists()
    assert "tunnel)" not in start_script
    assert "mlflow)" not in start_script
    assert "mlflow-up" not in (dashboard_root / "docs" / "README.md").read_text()


def test_each_compose_stack_has_a_distinct_project_name():
    assert (
        (REPO_ROOT / "infrastructure" / "dashboard" / "compose" / "docker-compose.yml")
        .read_text()
        .startswith("name: verda-dashboard\n")
    )
    assert (
        (REPO_ROOT / "infrastructure" / "local-emulator" / "compose" / "docker-compose.yml")
        .read_text()
        .startswith("name: verda-local-emulator\n")
    )
    assert (
        (REPO_ROOT / "training" / "compose" / "docker-compose.train.yml")
        .read_text()
        .startswith("name: minimind-training\n")
    )
