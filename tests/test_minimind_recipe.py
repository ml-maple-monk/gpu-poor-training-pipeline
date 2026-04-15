from __future__ import annotations

from pathlib import Path

from gpupoor.config import load_run_config
from gpupoor.recipes import minimind


def test_ensure_local_dataset_reuses_existing_pretokenized_artifact(tmp_path: Path, monkeypatch) -> None:
    config = load_run_config("examples/tiny_cpu.toml")
    dataset_dir = tmp_path / "data" / "datasets" / "pretrain_t2t_mini"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "metadata.json").write_text('{"version": 1}', encoding="utf-8")

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    bash_calls: list[tuple[Path, tuple[str, ...]]] = []

    def fake_bash_script(script: Path, *args: str, **kwargs: object) -> None:
        bash_calls.append((script, args))

    monkeypatch.setattr(minimind, "repo_path", fake_repo_path)
    monkeypatch.setattr(minimind, "bash_script", fake_bash_script)

    resolved = minimind.ensure_local_dataset(config)

    assert resolved == dataset_dir
    assert bash_calls == []
