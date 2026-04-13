"""test_collectors.py — Unit tests for collectors with mocked dependencies."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.errors import SourceStatus


def test_collect_tunnel_url_missing_file(tmp_path, monkeypatch):
    """collect_tunnel_url returns STALE when file doesn't exist."""
    monkeypatch.setattr("src.collectors.tunnel.CF_TUNNEL_URL_FILE", str(tmp_path / "missing.url"))
    from src.collectors.tunnel import collect_tunnel_url

    url, status = collect_tunnel_url()
    assert url == ""
    assert status == SourceStatus.STALE


def test_collect_tunnel_url_present(tmp_path, monkeypatch):
    """collect_tunnel_url returns URL when file exists."""
    url_file = tmp_path / ".cf-tunnel.url"
    url_file.write_text("https://example.trycloudflare.com\n")
    monkeypatch.setattr("src.collectors.tunnel.CF_TUNNEL_URL_FILE", str(url_file))
    from src.collectors.tunnel import collect_tunnel_url

    url, status = collect_tunnel_url()
    assert url == "https://example.trycloudflare.com"
    assert status == SourceStatus.OK


def test_collect_tunnel_url_empty_file(tmp_path, monkeypatch):
    """collect_tunnel_url returns STALE when file is empty."""
    url_file = tmp_path / ".cf-tunnel.url"
    url_file.write_text("")
    monkeypatch.setattr("src.collectors.tunnel.CF_TUNNEL_URL_FILE", str(url_file))
    from src.collectors.tunnel import collect_tunnel_url

    url, status = collect_tunnel_url()
    assert url == ""
    assert status == SourceStatus.STALE


def test_collect_training_snapshot_not_found():
    """collect_training_snapshot returns STALE when container not found."""
    from src.collectors.docker_logs import collect_training_snapshot

    with patch("src.collectors.docker_logs.safe_docker") as mock_docker:
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("", "")
        mock_proc.returncode = 1
        mock_docker.return_value = mock_proc

        snap, status = collect_training_snapshot("nonexistent-container")
        assert status == SourceStatus.STALE
        assert snap.status == "not_found"


def test_collect_training_snapshot_running():
    """collect_training_snapshot parses running container."""
    import json

    from src.collectors.docker_logs import collect_training_snapshot

    mock_data = json.dumps(
        [
            {
                "Id": "abc123def456",
                "State": {"Status": "running", "ExitCode": 0},
                "Config": {"Image": "my-image:latest"},
            }
        ]
    )

    with patch("src.collectors.docker_logs.safe_docker") as mock_docker:
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (mock_data, "")
        mock_proc.returncode = 0
        mock_docker.return_value = mock_proc

        snap, status = collect_training_snapshot("minimind-trainer")
        assert status == SourceStatus.OK
        assert snap.status == "running"
        assert snap.image == "my-image:latest"
        assert snap.container_id == "abc123def456"[:12]


def test_collect_dstack_runs_rest_error():
    """collect_dstack_runs returns ERROR on REST failure."""
    from src.collectors.dstack_rest import collect_dstack_runs

    with patch("src.collectors.dstack_rest.safe_dstack_rest") as mock_rest:
        mock_rest.side_effect = Exception("connection refused")
        runs, status = collect_dstack_runs()
        assert runs == []
        assert status == SourceStatus.ERROR


def test_collect_dstack_runs_success():
    """collect_dstack_runs parses runs list."""
    from src.collectors.dstack_rest import collect_dstack_runs

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "runs": [
            {"run_name": "run-1", "status": "RUNNING", "backend": "vastai", "gpu_count": 1},
        ]
    }
    with patch("src.collectors.dstack_rest.safe_dstack_rest", return_value=mock_resp):
        runs, status = collect_dstack_runs()
        assert status == SourceStatus.OK
        assert len(runs) == 1
        assert runs[0].run_name == "run-1"
        assert runs[0].status == "RUNNING"


def test_collect_artifacts_no_mount(monkeypatch):
    """collect_artifacts returns STALE when mount dir doesn't exist."""
    monkeypatch.setattr("src.collectors.artifacts.ARTIFACTS_MOUNT", "/nonexistent-path-xyz")
    from src.collectors.artifacts import collect_artifacts

    items, status = collect_artifacts()
    assert items == []
    assert status == SourceStatus.STALE
