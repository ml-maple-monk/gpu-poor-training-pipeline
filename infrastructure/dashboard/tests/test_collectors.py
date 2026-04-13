"""test_collectors.py — Unit tests for collectors with mocked dependencies."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.errors import SourceStatus


@pytest.mark.parametrize(
    ("contents", "expected_url", "expected_status"),
    [
        (None, "", SourceStatus.STALE),
        ("", "", SourceStatus.STALE),
        ("https://example.trycloudflare.com\n", "https://example.trycloudflare.com", SourceStatus.OK),
    ],
    ids=["missing-file", "empty-file", "present-url"],
)
def test_collect_tunnel_url_reports_source_health(tmp_path, monkeypatch, contents, expected_url, expected_status):
    """Proves the tunnel collector distinguishes missing, stale, and healthy
    sources so the dashboard reports tunnel state accurately."""
    url_file = tmp_path / ".cf-tunnel.url"
    if contents is not None:
        url_file.write_text(contents, encoding="utf-8")
    monkeypatch.setattr("src.collectors.tunnel.CF_TUNNEL_URL_FILE", str(url_file))
    from src.collectors.tunnel import collect_tunnel_url

    url, status = collect_tunnel_url()
    assert url == expected_url
    assert status == expected_status


@pytest.mark.parametrize(
    ("payload", "returncode", "expected_status", "expected_snapshot_status", "expected_image"),
    [
        ("", 1, SourceStatus.STALE, "not_found", ""),
        (
            json.dumps(
                [
                    {
                        "Id": "abc123def456",
                        "State": {"Status": "running", "ExitCode": 0},
                        "Config": {"Image": "my-image:latest"},
                    }
                ]
            ),
            0,
            SourceStatus.OK,
            "running",
            "my-image:latest",
        ),
    ],
    ids=["container-missing", "container-running"],
)
def test_collect_training_snapshot_reports_container_state(
    payload,
    returncode,
    expected_status,
    expected_snapshot_status,
    expected_image,
):
    """Proves the training collector maps docker inspect results into the
    dashboard's source-health and container-state model."""
    from src.collectors.docker_logs import collect_training_snapshot

    with patch("src.collectors.docker_logs.safe_docker") as mock_docker:
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (payload, "")
        mock_proc.returncode = returncode
        mock_docker.return_value = mock_proc

        snap, status = collect_training_snapshot("minimind-trainer")
        assert status == expected_status
        assert snap.status == expected_snapshot_status
        assert snap.image == expected_image
        if expected_status is SourceStatus.OK:
            assert snap.container_id == "abc123def456"[:12]


@pytest.mark.parametrize(
    ("side_effect", "response_payload", "expected_status", "expected_run_name"),
    [
        (Exception("connection refused"), None, SourceStatus.ERROR, None),
        (
            None,
            {"runs": [{"run_name": "run-1", "status": "RUNNING", "backend": "vastai", "gpu_count": 1}]},
            SourceStatus.OK,
            "run-1",
        ),
    ],
    ids=["rest-error", "rest-success"],
)
def test_collect_dstack_runs_maps_rest_results_into_dashboard_state(
    side_effect,
    response_payload,
    expected_status,
    expected_run_name,
):
    """Proves the dstack collector converts REST success/failure into the
    dashboard's run list plus source-health status."""
    from src.collectors.dstack_rest import collect_dstack_runs

    with patch("src.collectors.dstack_rest.safe_dstack_rest") as mock_rest:
        if side_effect is not None:
            mock_rest.side_effect = side_effect
        else:
            mock_resp = MagicMock()
            mock_resp.json.return_value = response_payload
            mock_rest.return_value = mock_resp
        runs, status = collect_dstack_runs()
        assert status == expected_status
        if expected_run_name is None:
            assert runs == []
        else:
            assert len(runs) == 1
            assert runs[0].run_name == expected_run_name
            assert runs[0].status == "RUNNING"


def test_collect_artifacts_no_mount(monkeypatch):
    """Proves the artifact collector reports a missing mount as stale state
    instead of pretending artifacts are present or healthy."""
    monkeypatch.setattr("src.collectors.artifacts.ARTIFACTS_MOUNT", "/nonexistent-path-xyz")
    from src.collectors.artifacts import collect_artifacts

    items, status = collect_artifacts()
    assert items == []
    assert status == SourceStatus.STALE
