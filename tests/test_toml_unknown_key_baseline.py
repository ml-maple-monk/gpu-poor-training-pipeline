"""Characterization test locking in load_run_config's unknown-key behavior.

This exists as a precondition for the refactor plan in
.claude/plans/zazzy-sniffing-prism.md (Task 0): PR1 deletes the
BackendConfig.keep_tunnel and .pull_artifacts fields, and we need a
known contract for operator TOML files that still set those keys.

As of this baseline, load_run_config silently ignores unknown keys at
both the top level and inside sections. If a future change tightens
the loader to reject unknown keys, this test will fail — that is the
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
    assert config.backend.keep_tunnel is False
    assert config.backend.pull_artifacts is False
    assert not hasattr(config, "mystery_top_level")
    assert not hasattr(config.backend, "mystery_backend_key")
