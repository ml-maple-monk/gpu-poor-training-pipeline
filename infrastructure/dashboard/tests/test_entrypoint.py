"""Tests for infrastructure/dashboard/scripts/entrypoint.sh.

The entrypoint interpolates DSTACK_TOKEN into a YAML config. If the token
contains colons, newlines, quotes, or backslashes, naive shell interpolation
corrupts the YAML. These tests run the script with pathological tokens and
verify the generated YAML round-trips the token through yaml.safe_load.

Because entrypoint.sh ends with `exec "$@"`, the test invokes it with `true`
as the command so the config is written before the exec completes successfully.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = REPO_ROOT / "infrastructure" / "dashboard" / "scripts" / "entrypoint.sh"


PATHOLOGICAL_TOKENS = [
    pytest.param("a:b", id="colons"),
    pytest.param("line1\nline2", id="embedded-newline"),
    pytest.param('has"quotes', id="double-quote"),
    pytest.param("ends with \\", id="trailing-backslash"),
]


@pytest.mark.parametrize("token", PATHOLOGICAL_TOKENS)
def test_entrypoint_writes_valid_yaml_with_pathological_token(token: str, tmp_path: Path) -> None:
    """entrypoint.sh must emit YAML that round-trips the token losslessly."""
    # The script writes to /tmp/.dstack/config.yml (hard-coded tmpfs path).
    # We cannot easily redirect that without refactoring, so isolate the run
    # in a sandbox with a private /tmp via unshare, OR run the script and
    # just inspect /tmp/.dstack/config.yml after.
    # Simpler: run it and read /tmp/.dstack/config.yml, then clean up.
    config_path = Path("/tmp/.dstack/config.yml")
    if config_path.exists():
        config_path.unlink()

    env = {
        **os.environ,
        "DSTACK_TOKEN": token,
        "DSTACK_SERVER": "http://host.docker.internal:3000",
        "DSTACK_PROJECT": "dashboard",
    }

    # Pass `true` as argv so `exec "$@"` at the end of the script succeeds
    # after the config has been written.
    result = subprocess.run(
        ["bash", str(ENTRYPOINT), "true"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"entrypoint.sh failed: stdout={result.stdout!r} stderr={result.stderr!r}"

    assert config_path.exists(), "entrypoint.sh did not write the config"
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert isinstance(loaded, dict)
    assert "projects" in loaded
    assert len(loaded["projects"]) == 1
    project = loaded["projects"][0]
    assert project["name"] == "dashboard"
    assert project["url"] == "http://host.docker.internal:3000"
    # The critical assertion: token must round-trip unchanged.
    assert project["token"] == token, f"token corruption: wrote {token!r}, read back {project['token']!r}"
