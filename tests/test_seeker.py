"""Focused tests for the seeker state machine and queue projections."""

from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path

import psycopg
import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

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


@pytest.fixture
def postgres_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> seeker.SeekerOrchestrator:
    schema = f"seeker_test_{uuid.uuid4().hex[:12]}"
    dsn = seeker.queue_dsn()

    monkeypatch.setattr(seeker, "repo_path", make_repo_path(tmp_path))
    monkeypatch.setattr(seeker, "_QUEUE_SCHEMA", schema)

    store = seeker.PostgresQueueStore(dsn)
    store.ensure_schema()
    orchestrator = seeker.SeekerOrchestrator(
        store=store,
        projector=seeker.FileSnapshotProjector(),
        worker_id="test-worker",
        lease_seconds=60,
    )
    monkeypatch.setattr(seeker, "default_orchestrator", lambda: orchestrator)

    yield orchestrator

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def read_attempt_rows(tmp_path: Path) -> list[dict[str, object]]:
    attempts_file = tmp_path / "data" / "seeker" / "attempts.jsonl"
    if not attempts_file.is_file():
        return []
    return [json.loads(line) for line in attempts_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def seed_submitted_job(
    orchestrator: seeker.SeekerOrchestrator,
    config,
) -> seeker.SeekerJob:
    store = orchestrator.store
    store.enqueue_job(seeker.FrozenRunConfigSnapshot.from_config(config))
    now = seeker.datetime.now(seeker.UTC)
    claimed = store.claim_next_pending_job(orchestrator.worker_id, now, orchestrator.lease_seconds)
    assert claimed is not None
    claimed.submitted_run_name = config.name
    claimed.last_status = "submitted"
    submitted = store.update_submitted_job(claimed, now, orchestrator.lease_seconds)
    with store._connect() as conn:
        conn.execute(
            f"""
            UPDATE {seeker._QUEUE_SCHEMA}.jobs
            SET next_poll_at = NULL
            WHERE job_id = %(job_id)s
            """,
            {"job_id": submitted.job_id},
        )
    store._upsert_queue_file()
    queue = store.projection_queue()
    assert queue.active_jobs
    return queue.active_jobs[0]


def test_enqueue_writes_pending_queue(
    tmp_path: Path,
    postgres_orchestrator: seeker.SeekerOrchestrator,
    capsys,
) -> None:
    config_file = tmp_path / "seek.toml"
    config_file.write_text(seeker_config_text(), encoding="utf-8")

    seeker.enqueue(str(config_file))

    queue_payload = json.loads((tmp_path / "data" / "seeker" / "queue.json").read_text(encoding="utf-8"))
    assert queue_payload["active"] is None
    assert len(queue_payload["pending"]) == 1
    assert "Enqueued seek-h100 as job" in capsys.readouterr().out


def test_enqueue_freezes_config_snapshot_and_future_edits_do_not_change_job(
    tmp_path: Path,
    postgres_orchestrator: seeker.SeekerOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "seek.toml"
    config_file.write_text(seeker_config_text(), encoding="utf-8")

    seeker.enqueue(str(config_file))

    config_file.write_text(
        seeker_config_text().replace('gpu = "H100"', 'gpu = "H200"').replace('regions = ["US-KS-1"]', ""),
        encoding="utf-8",
    )

    requests = []
    targeted_calls: list[dict[str, object]] = []

    def fake_targeted_offers(dstack_bin: str, **kwargs):
        targeted_calls.append(kwargs)
        return {
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
        }

    monkeypatch.setattr(seeker.dstack_backend, "fetch_targeted_offers", fake_targeted_offers)
    monkeypatch.setattr(seeker, "deploy_remote_request", lambda request, **kwargs: requests.append(request))

    seeker.run_daemon_cycle("dstack")

    assert targeted_calls[0]["gpu"] == "H100"
    assert targeted_calls[0]["regions"] == ("US-KS-1",)
    frozen_toml = base64.b64decode(requests[0].frozen_config_b64).decode("utf-8")
    frozen_payload = tomllib.loads(frozen_toml)
    assert frozen_payload["seeker"]["targets"][0]["gpu"] == "H100"
    assert frozen_payload["seeker"]["targets"][0]["regions"] == ["US-KS-1"]


def test_claim_next_pending_job_is_explicit_transition(
    tmp_path: Path,
    postgres_orchestrator: seeker.SeekerOrchestrator,
) -> None:
    config_file = tmp_path / "seek.toml"
    config_file.write_text(seeker_config_text(), encoding="utf-8")

    seeker.enqueue(str(config_file))

    claimed = postgres_orchestrator.claim_next_pending_job(seeker.datetime.now(seeker.UTC))

    queue = seeker.load_queue()
    assert claimed is not None
    assert claimed.state == seeker.SeekerJobState.CLAIMED.value
    assert queue.active is not None
    assert queue.active.job_id == claimed.job_id
    assert queue.pending == []


def test_mark_launching_keeps_job_visible_in_queue_projection(
    tmp_path: Path,
    postgres_orchestrator: seeker.SeekerOrchestrator,
) -> None:
    config_file = tmp_path / "seek.toml"
    config_file.write_text(seeker_config_text(), encoding="utf-8")
    config = load_run_config(config_file)

    seeker.enqueue(str(config_file))

    now = seeker.datetime.now(seeker.UTC)
    claimed = postgres_orchestrator.store.claim_next_pending_job(
        postgres_orchestrator.worker_id,
        now,
        postgres_orchestrator.lease_seconds,
    )
    assert claimed is not None
    claimed.submitted_run_name = config.name

    postgres_orchestrator.store.mark_launching(
        claimed,
        now,
        postgres_orchestrator.submit_lease_seconds(claimed),
    )

    queue = seeker.load_queue()
    assert queue.active is not None
    assert queue.active.state == seeker.SeekerJobState.LAUNCHING.value
    assert queue.active.submitted_run_name == config.name


def test_run_daemon_cycle_records_no_match_without_submitting(
    tmp_path: Path,
    postgres_orchestrator: seeker.SeekerOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "seek.toml"
    config_file.write_text(seeker_config_text(), encoding="utf-8")
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
    assert queue.active is None
    assert len(queue.pending) == 1
    assert queue.pending[0].state == seeker.SeekerJobState.RETRY_WAIT.value
    assert queue.pending[0].last_status == "no_match"
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


def test_run_daemon_cycle_submits_first_matching_offer(
    tmp_path: Path,
    postgres_orchestrator: seeker.SeekerOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "seek.toml"
    config_file.write_text(seeker_config_text(), encoding="utf-8")
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
    assert request.frozen_config_b64
    assert read_attempt_rows(tmp_path)[-1]["status"] == "submitted"


def test_failed_start_cancels_once_retry_budget_is_exhausted(
    tmp_path: Path,
    postgres_orchestrator: seeker.SeekerOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "seek.toml"
    config_file.write_text(seeker_config_text(max_submit_retries=1), encoding="utf-8")
    config = load_run_config(config_file)
    seed_submitted_job(postgres_orchestrator, config)

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


def test_refresh_submitted_job_keeps_running_job_active(
    tmp_path: Path,
    postgres_orchestrator: seeker.SeekerOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "seek.toml"
    config_file.write_text(seeker_config_text(), encoding="utf-8")
    config = load_run_config(config_file)
    seed_submitted_job(postgres_orchestrator, config)

    monkeypatch.setattr(seeker.dstack_backend, "dstack_has_run", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        seeker.dstack_backend,
        "dstack_run_status_triplet",
        lambda *args, **kwargs: ("running", "running", ""),
    )

    seeker.run_daemon_cycle("dstack")

    queue = seeker.load_queue()
    assert queue.active is not None
    assert queue.active.state == seeker.SeekerJobState.SUBMITTED.value
    assert read_attempt_rows(tmp_path) == []


def test_probe_targets_preserve_declared_target_order(
    postgres_orchestrator: seeker.SeekerOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = (
        SeekerTarget(backend="runpod", gpu="H100", count=1, mode="spot"),
        SeekerTarget(backend="vastai", gpu="H100", count=1, mode="spot"),
    )

    def fake_fetch(_dstack_bin: str, target: SeekerTarget):
        if target.backend == "runpod":
            time.sleep(0.05)
            return [
                seeker.SeekerOffer(
                    backend="runpod",
                    region="US-KS-1",
                    gpu="NVIDIA H100",
                    count=1,
                    mode="spot",
                    price_per_hour=1.75,
                    instance_type="runpod-h100x1",
                    availability="available",
                    normalized_gpu="h100",
                    raw={},
                )
            ]
        return [
            seeker.SeekerOffer(
                backend="vastai",
                region="US-CA",
                gpu="NVIDIA H100",
                count=1,
                mode="spot",
                price_per_hour=0.95,
                instance_type="vast-h100x1",
                availability="available",
                normalized_gpu="h100",
                raw={},
            )
        ]

    monkeypatch.setattr(seeker, "fetch_target_offers", fake_fetch)

    probe_results, _snapshot = postgres_orchestrator.probe_targets("dstack", targets)
    selected_target, selected_offer = postgres_orchestrator.select_target_offer(probe_results)

    assert [result.target.backend for result in probe_results] == ["runpod", "vastai"]
    assert selected_target is not None and selected_target.backend == "runpod"
    assert selected_offer is not None and selected_offer.backend == "runpod"


def test_fetch_target_offers_uses_all_regions_when_regions_are_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_targeted_offers(dstack_bin: str, **kwargs):
        captured.update(kwargs)
        return {"offers": []}

    monkeypatch.setattr(seeker.dstack_backend, "fetch_targeted_offers", fake_targeted_offers)

    seeker.fetch_target_offers(
        "dstack",
        SeekerTarget(backend="runpod", gpu="H100", count=1, mode="spot"),
    )

    assert captured["regions"] == ()


def test_target_probe_timeout_does_not_block_other_targets_and_records_diagnostics(
    tmp_path: Path,
    postgres_orchestrator: seeker.SeekerOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "seek.toml"
    config_file.write_text(
        """
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

[[seeker.targets]]
backend = "runpod"
gpu = "H100"
count = 1
mode = "spot"

[[seeker.targets]]
backend = "vastai"
gpu = "H100"
count = 1
mode = "spot"
""",
        encoding="utf-8",
    )
    seeker.enqueue(str(config_file))

    def fake_fetch(_dstack_bin: str, target: SeekerTarget):
        if target.backend == "runpod":
            raise TimeoutError("runpod probe timed out")
        return [
            seeker.SeekerOffer(
                backend="vastai",
                region="US-CA",
                gpu="NVIDIA H100",
                count=1,
                mode="spot",
                price_per_hour=1.05,
                instance_type="vast-h100x1",
                availability="available",
                normalized_gpu="h100",
                raw={},
            )
        ]

    requests = []
    monkeypatch.setattr(seeker, "fetch_target_offers", fake_fetch)
    monkeypatch.setattr(seeker, "deploy_remote_request", lambda request, **kwargs: requests.append(request))

    seeker.run_daemon_cycle("dstack")

    queue = seeker.load_queue()
    attempts = read_attempt_rows(tmp_path)
    offers_payload = json.loads((tmp_path / "data" / "seeker" / "latest_offers.json").read_text(encoding="utf-8"))
    assert len(requests) == 1
    assert requests[0].backend == "vastai"
    assert queue.active is not None
    assert queue.active.submitted_run_name == "seek-h100"
    assert attempts[-1]["status"] == "submitted"
    assert "runpod:H100:runpod probe timed out" in attempts[-1]["probe_error"]
    assert offers_payload["offers"][0]["backend"] == "vastai"


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


def test_fetch_target_offers_normalizes_vast_ai_backend_before_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_default_queue_store_fails_fast_when_postgres_startup_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        seeker.PostgresQueueStore,
        "ensure_schema",
        lambda self: (_ for _ in ()).throw(OSError("connection refused")),
    )

    with pytest.raises(
        RuntimeError,
        match="Seeker requires a reachable Postgres queue via SEEKER_QUEUE_DSN; startup failed: connection refused",
    ):
        seeker.default_queue_store()
