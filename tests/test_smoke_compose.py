"""Tests for gpupoor.ops.smoke compose-down invocation.

The smoke harness must not wipe named volumes by default — `docker compose
down -v` removes volumes and would destroy any user data in shared volumes.
Volume pruning is an explicit opt-in via the `--prune-volumes` flag.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gpupoor.config import SmokeConfig
from gpupoor.ops import smoke as smoke_module


def _collect_down_argvs(call_args_list) -> list[list[str]]:
    """Extract argv lists from run_command calls that invoked `... down ...`."""
    down_calls: list[list[str]] = []
    for call in call_args_list:
        # run_command is invoked positionally: run_command(command, ...)
        if not call.args:
            continue
        argv = list(call.args[0])
        if "down" in argv:
            down_calls.append(argv)
    return down_calls


@pytest.fixture
def smoke_stubs():
    """Stub the heavy pieces of run_smoke so we can inspect the down invocation.

    We patch:
      - run_preflight (no-op)
      - _build_local_image (returns fake tag, no docker build)
      - run_command (records argvs, returns a fake CompletedProcess)
      - _wait_for_health, probe functions (no-ops)
      - leak_scan (no-op)
    """
    with (
        patch.object(smoke_module, "run_preflight"),
        patch.object(smoke_module, "_build_local_image", return_value="fake:tag"),
        patch.object(smoke_module, "run_command") as mock_run,
        patch.object(smoke_module, "_wait_for_health"),
        patch.object(smoke_module, "_probe_uid_gid"),
        patch.object(smoke_module, "_probe_non_root_write"),
        patch.object(smoke_module, "_probe_sigterm_latency"),
        patch.object(smoke_module, "_probe_env_leak"),
        patch.object(smoke_module, "_probe_degraded_gating"),
        patch.object(smoke_module, "_probe_data_wait_timeout"),
        patch.object(smoke_module, "leak_scan"),
    ):
        yield mock_run


def test_smoke_down_does_not_remove_volumes_by_default(smoke_stubs) -> None:
    """The default `down` teardown must NOT pass `-v` (which removes volumes)."""
    smoke_module.run_smoke(SmokeConfig())

    down_argvs = _collect_down_argvs(smoke_stubs.call_args_list)
    assert down_argvs, "expected at least one `down` invocation"
    # None of the default-path down calls should contain '-v'.
    for argv in down_argvs:
        assert "-v" not in argv, (
            f"default `down` invocation unexpectedly includes '-v': {argv!r}"
        )


def test_smoke_prune_volumes_flag_passes_v(smoke_stubs) -> None:
    """When `prune_volumes=True`, the teardown `down` must include `-v`."""
    config = SmokeConfig(prune_volumes=True)
    smoke_module.run_smoke(config)

    down_argvs = _collect_down_argvs(smoke_stubs.call_args_list)
    assert down_argvs, "expected at least one `down` invocation"
    # At least one of the down calls (the final teardown) must include '-v'.
    assert any("-v" in argv for argv in down_argvs), (
        f"prune_volumes=True did not cause `-v`: {down_argvs!r}"
    )
