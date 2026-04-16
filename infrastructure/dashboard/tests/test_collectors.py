"""test_collectors.py — Unit tests for collectors with mocked dependencies."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.errors import SourceStatus


def _plan_response(offers):
    resp = MagicMock()
    resp.json.return_value = {"job_plans": [{"offers": offers}]}
    return resp


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


def test_collect_seeker_state_reads_queue_offer_and_attempt_files(tmp_path, monkeypatch):
    seeker_dir = tmp_path / "seeker"
    seeker_dir.mkdir()
    (seeker_dir / "latest_offers.json").write_text(
        json.dumps(
            {
                "offers": [
                    {
                        "backend": "runpod",
                        "region": "US-KS-1",
                        "gpu": "H100",
                        "count": 1,
                        "mode": "spot",
                        "price_per_hour": 1.25,
                        "instance_type": "h100x1",
                        "availability": "available",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (seeker_dir / "queue.json").write_text(
        json.dumps(
            {
                "active": {
                    "job_id": "job-1",
                    "config_name": "debug-run",
                    "submitted_run_name": "debug-run",
                    "submit_retries": 1,
                    "last_status": "submitted",
                    "last_reason": "",
                    "enqueued_at": "2026-04-16T00:00:00+00:00",
                },
                "pending": [],
            }
        ),
        encoding="utf-8",
    )
    (seeker_dir / "attempts.jsonl").write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "attempt_id": "attempt-1",
                "backend": "runpod",
                "region": "US-KS-1",
                "gpu": "H100",
                "count": 1,
                "mode": "spot",
                "price_per_hour": 1.25,
                "status": "submitted",
                "reason": "launch ok",
                "started_at": "2026-04-16T00:00:00+00:00",
                "ended_at": "2026-04-16T00:00:05+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.collectors.seeker_state.SEEKER_DATA_DIR", str(seeker_dir))
    from src.collectors.seeker_state import collect_seeker_state

    offers, active, pending, attempts, status = collect_seeker_state()

    assert status == SourceStatus.OK
    assert len(offers) == 1
    assert offers[0].backend == "runpod"
    assert active is not None and active.job_id == "job-1"
    assert pending == []
    assert len(attempts) == 1
    assert attempts[0].status == "submitted"


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
        (httpx.ConnectError("connection refused"), None, SourceStatus.ERROR, None),
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


def test_collect_verda_offers_targets_five_gpu_specs_and_prefers_a100_80g(monkeypatch):
    """Proves the offer probe only targets the intended five GPUs and rewrites
    A100 results to the stable A100-80G display name when 80G capacity exists."""
    from src.collectors import verda_offers

    calls = []

    def fake_rest(endpoint, *, method="GET", json=None, timeout=10.0):
        calls.append((endpoint, method, timeout, json["run_spec"]["configuration"]["resources"]["gpu"]["name"]))
        probe_name = json["run_spec"]["configuration"]["resources"]["gpu"]["name"]
        if probe_name == "A100":
            return _plan_response(
                [
                    {
                        "price": 2.8,
                        "backend": "runpod",
                        "region": "US-KS-1",
                        "instance": {
                            "name": "a100-40g",
                            "resources": {"spot": True, "gpus": [{"memory_mib": 40_960}]},
                        },
                    },
                    {
                        "price": 3.2,
                        "backend": "runpod",
                        "region": "US-KS-1",
                        "instance": {
                            "name": "a100-80g",
                            "resources": {"spot": True, "gpus": [{"memory_mib": 81_920}]},
                        },
                    },
                ]
            )
        return _plan_response([])

    monkeypatch.setattr(verda_offers, "safe_dstack_rest", fake_rest)

    offers, status = verda_offers.collect_verda_offers()

    assert status == SourceStatus.OK
    assert [spec[0] for spec in verda_offers.GPU_SPECS] == ["A100", "H100", "H200", "B200", "B300"]
    assert [call[3] for call in calls] == ["A100", "H100", "H200", "B200", "B300"]
    assert all(call[2] == 15.0 for call in calls)
    assert len(offers) == 1
    assert offers[0].gpu_name == "A100-80G"
    assert offers[0].instance_type == "a100-80g"


def test_collect_verda_offers_falls_back_when_no_a100_80g_exists(monkeypatch):
    """Proves the A100 memory preference is non-destructive when only smaller
    A100 variants are returned by the API."""
    from src.collectors import verda_offers

    def fake_rest(endpoint, *, method="GET", json=None, timeout=10.0):
        probe_name = json["run_spec"]["configuration"]["resources"]["gpu"]["name"]
        if probe_name == "A100":
            return _plan_response(
                [
                    {
                        "price": 2.5,
                        "backend": "vastai",
                        "region": "CA-ON",
                        "instance": {
                            "name": "a100-40g",
                            "resources": {"spot": False, "gpus": [{"memory_mib": 40_960}]},
                        },
                    }
                ]
            )
        return _plan_response([])

    monkeypatch.setattr(verda_offers, "safe_dstack_rest", fake_rest)

    offers, status = verda_offers.collect_verda_offers()

    assert status == SourceStatus.OK
    assert len(offers) == 1
    assert offers[0].gpu_name == "A100-80G"
    assert offers[0].instance_type == "a100-40g"


def test_archive_offer_snapshots_uses_cheapest_offer_and_tracks_unavailable_pairs():
    """Proves snapshot archival is bounded, keeps the cheapest current offer
    per GPU/backend pair, and still records unavailable GPU/backend pairs."""
    from src.collector_workers import _archive_offer_snapshots
    from src.state import SeekerOffer, reset_state

    state = reset_state()
    now = datetime.now(UTC)
    offers = [
        SeekerOffer(
            gpu_name="H100",
            backend="runpod",
            region="US-KS-1",
            price_per_hour=2.5,
            count=1,
            mode="spot",
            availability="available",
            instance_type="h100-a",
        ),
        SeekerOffer(
            gpu_name="H100",
            backend="runpod",
            region="US-KS-2",
            price_per_hour=1.9,
            count=1,
            mode="spot",
            availability="available",
            instance_type="h100-b",
        ),
        SeekerOffer(
            gpu_name="H200",
            backend="vastai",
            region="CA-ON",
            price_per_hour=3.1,
            count=2,
            mode="on-demand",
            availability="available",
            instance_type="h200-x2",
        ),
    ]

    for idx in range(61):
        _archive_offer_snapshots(state, offers, now + timedelta(minutes=idx))

    runpod_h100 = state.offer_history[("H100", "runpod")]
    assert len(runpod_h100) == 60
    assert runpod_h100[-1].available is True
    assert runpod_h100[-1].price_per_hour == 1.9
    assert runpod_h100[-1].count == 1

    unavailable = state.offer_history[("B300", "runpod")]
    assert len(unavailable) == 60
    assert unavailable[-1].available is False
    assert unavailable[-1].price_per_hour == 0.0
    assert unavailable[-1].count == 0


def test_start_all_collectors_exposes_dstack_offer_worker_at_sixty_seconds(monkeypatch):
    """Proves the dstack offer worker cadence was widened to avoid overlapping
    the slower multi-GPU probe sequence."""
    from src import collector_workers
    from src.state import reset_state

    state = reset_state()
    monkeypatch.setattr(collector_workers, "collect_training_snapshot", lambda target: (MagicMock(), SourceStatus.OK))
    monkeypatch.setattr(collector_workers, "collect_live_metrics", lambda: ({}, SourceStatus.OK))
    monkeypatch.setattr(collector_workers, "collect_dstack_runs", lambda: ([], SourceStatus.OK))
    monkeypatch.setattr(collector_workers, "collect_system", lambda: (MagicMock(), SourceStatus.OK))
    monkeypatch.setattr(collector_workers, "collect_mlflow_recent", lambda: ([], SourceStatus.OK))
    monkeypatch.setattr(collector_workers, "collect_tunnel_url", lambda: ("", SourceStatus.STALE))
    monkeypatch.setattr(collector_workers, "collect_seeker_state", lambda: ([], None, [], [], SourceStatus.OK))
    monkeypatch.setattr(collector_workers, "collect_verda_offers", lambda: ([], SourceStatus.OK))

    workers = collector_workers.start_all_collectors(state)
    state.shutdown_event.set()
    for worker in workers:
        worker.join(timeout=1.0)

    cadence_by_name = {worker.name: worker.cadence for worker in workers}
    assert cadence_by_name["dstack-offers-60s"] == 60.0


# ── F6 Test 5: collector timestamps must be timezone-aware UTC ────────────────


def test_collector_uses_timezone_aware_datetime(monkeypatch):
    """Proves the training collector stamps `last_refreshed_at` with a
    tz-aware datetime (datetime.now(timezone.utc)), not the deprecated naive
    datetime.utcnow() which loses zone information and emits DeprecationWarning
    on Python 3.12+.
    """
    from src import collector_workers
    from src.state import reset_state

    state = reset_state()

    # Stub the underlying collect_fn so we exercise only the state-write path.
    fake_snap = MagicMock()
    monkeypatch.setattr(
        collector_workers,
        "collect_training_snapshot",
        lambda target: (fake_snap, SourceStatus.OK),
    )
    monkeypatch.setattr(collector_workers, "collect_live_metrics", lambda: ({}, SourceStatus.OK))
    monkeypatch.setattr(collector_workers, "collect_dstack_runs", lambda: ([], SourceStatus.OK))
    monkeypatch.setattr(collector_workers, "collect_system", lambda: (MagicMock(), SourceStatus.OK))
    monkeypatch.setattr(collector_workers, "collect_mlflow_recent", lambda: ([], SourceStatus.OK))
    monkeypatch.setattr(collector_workers, "collect_tunnel_url", lambda: ("", SourceStatus.STALE))
    monkeypatch.setattr(collector_workers, "collect_seeker_state", lambda: ([], None, [], [], SourceStatus.OK))
    monkeypatch.setattr(collector_workers, "collect_verda_offers", lambda: ([], SourceStatus.OK))

    # Build the collector list but don't start threads — just invoke the 2s
    # training closure directly.
    workers = collector_workers.start_all_collectors(state)
    for _worker in workers:
        # Stop the daemon threads so they can't race with our assertions.
        state.shutdown_event.set()
    # Let the threads observe the shutdown and exit.
    for w in workers:
        w.join(timeout=1.0)

    stamp = state.last_refreshed_at.get("training")
    assert stamp is not None, "expected 'training' key to be populated"
    assert stamp.tzinfo is not None, (
        f"collector must record a tz-aware datetime (datetime.now(timezone.utc)), got naive: {stamp!r}"
    )
