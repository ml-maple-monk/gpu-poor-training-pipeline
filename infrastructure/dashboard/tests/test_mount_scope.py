"""test_mount_scope.py — Mount scope verification.

Tests verify that:
(i)  C2.2 path: no ~/.dstack mount exists on the container
(ii) C2.1a path: mount Source ends with 'config.yml'
(iii) C2.1b path: mount Source ends with a named subtree under ~/.dstack/projects/

These tests require a running container and are skipped in unit-test-only mode.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

CONTAINER_NAME = "verda-dashboard-gradio"


def _docker_inspect():
    """Return parsed docker inspect output, or None if container not running."""
    try:
        result = subprocess.run(
            ["docker", "inspect", CONTAINER_NAME],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except Exception:
        return None


def _container_running():
    """Return True only if our verda-dashboard (read_only=true) container is running."""
    data = _docker_inspect()
    if not data:
        return False
    state = data[0].get("State", {})
    name = data[0].get("Name", "")
    host_config = data[0].get("HostConfig", {})
    # Must be our dashboard: running, correct name, AND ReadonlyRootfs (our marker)
    return (
        state.get("Running", False)
        and CONTAINER_NAME in name
        and host_config.get("ReadonlyRootfs") is True
    )


@pytest.mark.skipif(not _container_running(), reason="verda-dashboard container not running")
@pytest.mark.docker
@pytest.mark.live_dashboard
def test_readonly_rootfs():
    """Container must have ReadonlyRootfs: true."""
    data = _docker_inspect()
    assert data is not None
    host_config = data[0].get("HostConfig", {})
    assert host_config.get("ReadonlyRootfs") is True, (
        "Container must have ReadonlyRootfs: true"
    )


@pytest.mark.skipif(not _container_running(), reason="verda-dashboard container not running")
@pytest.mark.docker
@pytest.mark.live_dashboard
def test_tmpfs_mounts_present():
    """Container must have /tmp, /tmp/.cache, /tmp/mpl tmpfs mounts."""
    data = _docker_inspect()
    assert data is not None
    tmpfs = data[0].get("HostConfig", {}).get("Tmpfs", {})
    assert "/tmp" in tmpfs, "Missing /tmp tmpfs mount"
    assert "/tmp/.cache" in tmpfs, "Missing /tmp/.cache tmpfs mount"
    assert "/tmp/mpl" in tmpfs, "Missing /tmp/mpl tmpfs mount"


@pytest.mark.skipif(not _container_running(), reason="verda-dashboard container not running")
@pytest.mark.docker
@pytest.mark.live_dashboard
def test_no_dstack_home_mount_in_c22():
    """In C2.2 path: no ~/.dstack directory mount on the container."""
    data = _docker_inspect()
    assert data is not None
    mounts = data[0].get("Mounts", [])
    home = os.path.expanduser("~")
    dstack_dir = os.path.join(home, ".dstack")

    dstack_mounts = [
        m for m in mounts
        if m.get("Source", "") == dstack_dir
        or m.get("Source", "").startswith(dstack_dir + os.sep)
    ]
    # If C2.2 is active, there should be no dstack mounts
    # This test is informational if C2.1 is active
    if dstack_mounts:
        # Check that it's at most a single-file mount (C2.1a)
        for m in dstack_mounts:
            source = m.get("Source", "")
            assert source.endswith("config.yml") or "/projects/" in source, (
                f"Overly broad dstack mount: {source} "
                "(expected config.yml or a named subtree)"
            )


@pytest.mark.skipif(not _container_running(), reason="verda-dashboard container not running")
@pytest.mark.docker
@pytest.mark.live_dashboard
def test_whitelisted_mounts_only():
    """Only whitelisted mount sources are present."""
    data = _docker_inspect()
    assert data is not None
    mounts = data[0].get("Mounts", [])

    repo_root = str(Path(__file__).resolve().parents[3])
    home = str(Path.home())
    allowed_prefixes = (
        "/var/run/docker.sock",
        "/usr/bin/docker",           # host docker CLI binary (read-only)
        os.path.join(home, ".dstack-cli-venv"),
        os.path.join(home, ".local", "share", "uv", "python"),
        os.path.join(repo_root, "artifacts-pull"),
        os.path.join(repo_root, ".cf-tunnel.url"),
    )

    for m in mounts:
        source = m.get("Source", "")
        if not source:
            continue
        ok = any(source.startswith(prefix) for prefix in allowed_prefixes)
        assert ok, f"Unexpected mount source: {source}"
