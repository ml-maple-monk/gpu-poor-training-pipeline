"""Characterization test locking in load_run_config's unknown-key behavior.

This contract lets us delete TOML keys without breaking operator
configs that still set them. The fixture intentionally includes
`mystery_top_level`, `mystery_backend_key`, plus the now-removed
`keep_tunnel` and `pull_artifacts` keys. If a future change tightens
the loader to reject unknown keys, this test fails — that is the
signal to add a one-cycle deprecation step before removing any
public TOML keys.
"""

from __future__ import annotations

from pathlib import Path

from gpupoor.config import load_run_config

REPO_ROOT = Path(__file__).resolve().parents[1]
UNKNOWN_KEY_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "toml_unknown_keys" / "unknown_key_config.toml"


def test_load_run_config_silently_ignores_unknown_keys() -> None:
    config = load_run_config(UNKNOWN_KEY_FIXTURE)

    assert config.name == "unknown_key_probe"
    assert config.backend.kind == "local"
    assert not hasattr(config, "mystery_top_level")
    assert not hasattr(config.backend, "mystery_backend_key")
    assert not hasattr(config.backend, "keep_tunnel")
    assert not hasattr(config.backend, "pull_artifacts")
