"""Tests for the file-backed remote capacity seeker."""

from __future__ import annotations

import json
from pathlib import Path

from gpupoor.config import SeekerTarget, load_run_config
from gpupoor.services import seeker


def seeker_config_text(*, max_submit_retries: int = 3) -> str:
    return f"""
name = "seek-h100"
[recipe]
[backend]
kind = "dstack"
[mlflow]
[doctor]
[smoke]
[remote]
[seeker]
poll_seconds = 15
max_submit_retries = {max_submit_retries}

[[seeker.targets]]
backend = "runpod"
gpu = "H100"
count = 1
mode = "spot"
regions = ["US-KS-1"]
max_price = 2.0
"""


def make_repo_path(tmp_path: Path):
    return lambda *parts: tmp_path.joinpath(*parts)


def read_attempt_rows(tmp_path: Path) -> list[dict[str, object]]:
    attempts_file = tmp_path / "data" / "seeker" / "attempts.jsonl"
    return [json.loads(line) for line in attempts_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_enqueue_writes_pending_queue(tmp_path: Path, monkeypatch, capsys) -> None:
    config_file = tmp_path / "seek.toml"
    config_file.write_text(seeker_config_text(), encoding="utf-8")
    monkeypatch.setattr(seeker, "repo_path", make_repo_path(tmp_path))

    seeker.enqueue(str(config_file))

    queue_payload = json.loads((tmp_path / "data" / "seeker" / "queue.json").read_text(encoding="utf-8"))
    assert queue_payload["active"] is None
    assert len(queue_payload["pending"]) == 1
    assert "Enqueued seek-h100 as job" in capsys.readouterr().out


def test_run_daemon_cycle_records_no_match_without_submitting(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "seek.toml"
    config_file.write_text(seeker_config_text(), encoding="utf-8")
    monkeypatch.setattr(seeker, "repo_path", make_repo_path(tmp_path))
    seeker.enqueue(str(config_file))

    submitted: list[object] = []
    targeted_calls: list[dict[str, object]] = []

    def fake_targeted_offers(dstack_bin: str, **kwargs):
        targeted_calls.append(kwargs)
        return {"offers": []}

    monkeypatch.setattr(seeker.dstack_backend, "fetch_targeted_offers", fake_targeted_offers)
    monkeypatch.setattr(seeker, "deploy_remote_request", lambda *args, **kwargs: submitted.append(args))

    seeker.run_daemon_cycle("dstack")

    queue = seeker.load_queue()
    assert queue.active is not None
    assert queue.active.last_status == "no_match"
    assert submitted == []
    assert targeted_calls == [
        {
            "backend": "runpod",
            "gpu": "H100",
            "count": 1,
            "mode": "spot",
            "regions": ("US-KS-1",),
            "max_price": 2.0,
        }
    ]
    assert read_attempt_rows(tmp_path)[-1]["status"] == "no_match"


def test_run_daemon_cycle_submits_first_matching_offer(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "seek.toml"
    config_file.write_text(seeker_config_text(), encoding="utf-8")
    monkeypatch.setattr(seeker, "repo_path", make_repo_path(tmp_path))
    seeker.enqueue(str(config_file))

    requests = []

    def capture_request(request, *, skip_build=None, dry_run=False):
        requests.append((request, skip_build, dry_run))

    monkeypatch.setattr(
        seeker.dstack_backend,
        "fetch_targeted_offers",
        lambda dstack_bin, **kwargs: {
            "offers": [
                {
                    "backend": "runpod",
                    "region": "US-KS-1",
                    "price": 1.25,
                    "availability": "available",
                    "instance": {
                        "name": "h100x1",
                        "resources": {
                            "spot": True,
                            "gpus": [{"name": "NVIDIA H100"}],
                        },
                    },
                }
            ]
        },
    )
    monkeypatch.setattr(seeker, "deploy_remote_request", capture_request)

    seeker.run_daemon_cycle("dstack")

    queue = seeker.load_queue()
    assert len(requests) == 1
    request, skip_build, dry_run = requests[0]
    assert request.deployment_target == "remote"
    assert request.backend == "runpod"
    assert request.region == "US-KS-1"
    assert request.gpu == "H100"
    assert request.mode == "spot"
    assert skip_build is None
    assert dry_run is False
    assert queue.active is not None
    assert queue.active.submitted_run_name == load_run_config(config_file).name
    assert read_attempt_rows(tmp_path)[-1]["status"] == "submitted"


def test_failed_start_cancels_once_retry_budget_is_exhausted(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "seek.toml"
    config_file.write_text(seeker_config_text(max_submit_retries=1), encoding="utf-8")
    config = load_run_config(config_file)
    monkeypatch.setattr(seeker, "repo_path", make_repo_path(tmp_path))
    queue = seeker.SeekerQueue(
        active=seeker.SeekerJob(
            job_id="job-1",
            config_name=config.name,
            config_path=str(config.source),
            enqueued_at="2026-04-16T00:00:00+00:00",
            submit_retries=0,
            submitted_run_name=config.name,
            last_status="submitted",
        ),
    )
    seeker.save_queue(queue)
    monkeypatch.setattr(seeker.dstack_backend, "dstack_has_run", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        seeker.dstack_backend,
        "dstack_run_status_triplet",
        lambda *args, **kwargs: ("pending", "failed", "failed_to_start_due_to_no_capacity"),
    )

    seeker.run_daemon_cycle("dstack")

    assert seeker.load_queue().active is None
    statuses = [row["status"] for row in read_attempt_rows(tmp_path)]
    assert statuses == ["failed_to_start", "cancelled"]


def test_match_offer_normalizes_vast_ai_backend_alias() -> None:
    target = SeekerTarget(
        backend="vast ai",
        gpu="H100",
        count=1,
        mode="on-demand",
    )
    offer = seeker.normalize_offer(
        {
            "backend": "vastai",
            "region": "US-CA",
            "price": 2.1,
            "availability": "available",
            "instance": {
                "name": "NVIDIA H100 80GB HBM3",
                "resources": {
                    "spot": False,
                    "gpus": [{"name": "NVIDIA H100"}],
                },
            },
        }
    )

    assert seeker.match_offer(target, [offer]) == offer


def test_fetch_target_offers_normalizes_vast_ai_backend_before_cli(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_targeted_offers(dstack_bin: str, **kwargs):
        captured.update(kwargs)
        return {"offers": []}

    monkeypatch.setattr(seeker.dstack_backend, "fetch_targeted_offers", fake_targeted_offers)

    seeker.fetch_target_offers(
        "dstack",
        SeekerTarget(backend="Vast AI", gpu="H100", count=1, mode="on-demand"),
    )

    assert captured["backend"] == "vastai"


def test_match_offer_does_not_broaden_h100nvl_to_h100() -> None:
    target = SeekerTarget(
        backend="vastai",
        gpu="H100",
        count=1,
        mode="on-demand",
    )
    offer = seeker.normalize_offer(
        {
            "backend": "vastai",
            "region": "US-CA",
            "price": 3.0,
            "availability": "available",
            "instance": {
                "name": "NVIDIA H100 NVL",
                "resources": {
                    "spot": False,
                    "gpus": [{"name": "NVIDIA H100NVL"}],
                },
            },
        }
    )

    assert seeker.match_offer(target, [offer]) is None
