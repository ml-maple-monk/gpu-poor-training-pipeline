"""Golden-fixture test locking in the dstack --dry-run output signature.

Part of Task 0 in .claude/plans/zazzy-sniffing-prism.md. PR2 (cli/legacy
dispatch rewrite) and PR3 (launch_remote split) must preserve the
exact sequence of env-dict writes and "Would …" lines emitted by
`gpupoor launch dstack --dry-run examples/verda_remote.toml`. The
captured fixture lives at tests/fixtures/golden/verda_remote_dry_run.yaml
with <HOME> and <REPO_ROOT> placeholders so the test is portable
across checkouts for the same operator.

This fixture also encodes operator-specific values (e.g.,
VCR_IMAGE_BASE UUID from .env.remote, jq-install-state preflight
warnings). If you run this on a different machine or after rotating
secrets, regenerate the fixture via:

    python -m gpupoor launch dstack --dry-run examples/verda_remote.toml \\
      | sed -e "s|$(git rev-parse --show-toplevel)|<REPO_ROOT>|g" \\
            -e "s|$HOME|<HOME>|g" \\
      > tests/fixtures/golden/verda_remote_dry_run.yaml
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "golden" / "verda_remote_dry_run.yaml"
DRY_RUN_CONFIG = REPO_ROOT / "examples" / "verda_remote.toml"


def normalize(text: str) -> str:
    home = str(Path.home())
    return text.replace(str(REPO_ROOT), "<REPO_ROOT>").replace(home, "<HOME>")


def test_dstack_dry_run_matches_golden_fixture() -> None:
    if not GOLDEN_FIXTURE.exists():
        pytest.skip(f"Golden fixture missing at {GOLDEN_FIXTURE}; regenerate per module docstring.")

    expected = GOLDEN_FIXTURE.read_text()
    proc = subprocess.run(
        [sys.executable, "-m", "gpupoor", "launch", "dstack", "--dry-run", str(DRY_RUN_CONFIG)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONHASHSEED": "0", "TZ": "UTC"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    actual = normalize(proc.stdout)

    assert proc.returncode == 0, f"dry-run exited {proc.returncode}:\n{actual}"
    assert actual == expected, (
        "dstack --dry-run output diverged from golden fixture. "
        "If this is an intentional behavior change, regenerate the fixture "
        "per the module docstring."
    )
