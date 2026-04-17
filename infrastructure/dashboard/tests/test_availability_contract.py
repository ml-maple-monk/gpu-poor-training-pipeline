"""Tiny contract checks for the Postgres-backed availability dashboard."""

from __future__ import annotations

import sys
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path

import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

utils = import_module("src.utils")
NormalizedOffer = import_module("src.models").NormalizedOffer


TEST_QUEUE_DSN = "postgresql://mlflow:mlflow@127.0.0.1:55432/mlflow"


@pytest.fixture
def dashboard_config(monkeypatch: pytest.MonkeyPatch):
    schema = f"dashboard_test_{uuid.uuid4().hex[:12]}"
    monkeypatch.setenv("SEEKER_QUEUE_DSN", TEST_QUEUE_DSN)
    monkeypatch.setattr(utils, "dashboard_schema", schema)
    utils.invalidate_snapshot_cache()

    config = utils.load_dashboard_config()
    utils.ensure_dashboard_schema(config)

    yield config

    with psycopg.connect(TEST_QUEUE_DSN, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    utils.invalidate_snapshot_cache()


def count_rows(config, table: str) -> int:
    with utils.db_connect(config) as conn:
        row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"])


def seed_success_sweep(config, sampled_at: datetime, updates: dict[tuple[str, str, str], ProviderUpdate] | None = None) -> None:
    updates = updates or {}
    rows = utils.blank_provider_rows(config)
    patched_rows = []
    for row in rows:
        patched_rows.append(replace(row, **updates.get((row.backend, row.gpu, row.mode), {})))

    with utils.db_connect(config) as conn, conn.transaction():
        sweep_id = utils.write_sweep_start(conn, sampled_at)
        utils.write_provider_samples(conn, sweep_id, sampled_at, patched_rows)
        utils.finalize_sweep_success(conn, sweep_id, sampled_at, len(patched_rows), 0, ())
    utils.invalidate_snapshot_cache()


ProviderUpdate = dict[str, object]


def test_schema_bootstrap_and_sweep_contract(dashboard_config, monkeypatch: pytest.MonkeyPatch) -> None:
    config = dashboard_config
    utils.ensure_dashboard_schema(config)

    offers = [
        NormalizedOffer(
            source="dstack",
            backend="runpod",
            provider_label="RunPod",
            provider_color="#a371f7",
            gpu="H100",
            mode=utils.normalize_mode("spot"),
            region="US-KS-1",
            instance_type="h100x1",
            price_per_hour=1.75,
            count=1,
            available=True,
        ),
        NormalizedOffer(
            source="dstack",
            backend="runpod",
            provider_label="RunPod",
            provider_color="#a371f7",
            gpu="H100",
            mode=utils.normalize_mode("on-demand"),
            region="US-KS-1",
            instance_type="h100x1",
            price_per_hour=2.99,
            count=1,
            available=True,
        ),
        NormalizedOffer(
            source="dstack",
            backend="vastai",
            provider_label="Vast.ai",
            provider_color="#58a6ff",
            gpu="A100-80G",
            mode=utils.normalize_mode("spot"),
            region="EU-NL-1",
            instance_type="a100x1",
            price_per_hour=1.05,
            count=2,
            available=True,
        ),
        NormalizedOffer(
            source="dstack",
            backend="runpod",
            provider_label="RunPod",
            provider_color="#a371f7",
            gpu="H100",
            mode=utils.normalize_mode("auction"),
            region="US-KS-1",
            instance_type="h100x1",
            price_per_hour=1.0,
            count=1,
            available=True,
        ),
    ]

    monkeypatch.setattr(utils, "fetch_dstack_offers", lambda _: (offers, "dstack ok (3 offers)"))

    assert utils.run_sweep_cycle(config) is True
    assert count_rows(config, utils.sweep_runs_table()) == 1
    assert count_rows(config, utils.provider_samples_table()) == len(config.platform_colors) * len(config.gpu_specs) * 2

    snapshot = utils.build_dashboard_snapshot(config)
    h100_card = next(card for card in snapshot.preemptible.cards if card.gpu == "H100")
    runpod_row = next(row for row in h100_card.rows if row.backend == "runpod")
    verda_row = next(row for row in h100_card.rows if row.backend == "verda")

    assert utils.assert_plan_endpoint("runs/get_plan") == "runs/get_plan"
    with pytest.raises(ValueError):
        utils.assert_plan_endpoint("runs/list")
    assert runpod_row.available is True
    assert runpod_row.current_count == 1
    assert runpod_row.cheapest_price == pytest.approx(1.75)
    assert verda_row.available is False
    assert snapshot.hidden_unknown_count == 1
    assert all(card.mode in {"preemptible", "on-demand"} for card in snapshot.preemptible.cards + snapshot.on_demand.cards)


def test_sql_history_last_available_and_error_fallback(dashboard_config, monkeypatch: pytest.MonkeyPatch) -> None:
    config = dashboard_config
    older = datetime.now(UTC) - timedelta(minutes=20)
    newer = datetime.now(UTC) - timedelta(minutes=5)

    seed_success_sweep(
        config,
        older,
        {
            ("runpod", "H100", "preemptible"): {
                "available": True,
                "current_count": 3,
                "cheapest_price": 1.5,
                "regions_label": "US-KS-1",
                "instance_label": "h100x1",
            }
        },
    )
    seed_success_sweep(config, newer)

    snapshot = utils.build_dashboard_snapshot(config)
    h100_card = next(card for card in snapshot.preemptible.cards if card.gpu == "H100")
    runpod_row = next(row for row in h100_card.rows if row.backend == "runpod")

    assert runpod_row.available is False
    assert runpod_row.availability_percent == pytest.approx(50.0)
    assert runpod_row.last_available_at == older

    monkeypatch.setattr(utils, "fetch_dstack_offers", lambda _: (_ for _ in ()).throw(RuntimeError("probe crashed")))
    assert utils.run_sweep_cycle(config) is True

    errored = utils.build_dashboard_snapshot(config)
    errored_h100 = next(card for card in errored.preemptible.cards if card.gpu == "H100")
    errored_runpod = next(row for row in errored_h100.rows if row.backend == "runpod")

    assert errored.sweep.state == "error"
    assert "probe crashed" in errored.sweep.last_error_text
    assert errored_runpod.availability_percent == pytest.approx(50.0)
    assert errored_runpod.last_available_at == older


def test_advisory_lock_and_render_without_seeker_mount(dashboard_config, monkeypatch: pytest.MonkeyPatch) -> None:
    config = dashboard_config
    with utils.db_connect(config) as conn:
        assert utils.acquire_sweep_lock(conn) is True
        monkeypatch.setattr(utils, "fetch_dstack_offers", lambda _: ([], "dstack ok (0 offers)"))
        assert utils.run_sweep_cycle(config) is False
        utils.release_sweep_lock(conn)

    monkeypatch.delenv("SEEKER_DATA_DIR", raising=False)
    snapshot = utils.build_dashboard_snapshot(config)

    from src.app import render_dashboard

    shell = render_dashboard(snapshot, config)
    assert shell is not None


def test_dstack_token_falls_back_to_config_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        "projects:\n- default: true\n  name: main\n  token: file-token\n  url: http://127.0.0.1:3000\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("DSTACK_SERVER_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("DSTACK_CONFIG_PATH", str(config_file))
    monkeypatch.setattr(utils, "dstack_config_path", config_file)

    config = utils.load_dashboard_config()

    assert config.dstack_token == "file-token"

    monkeypatch.setenv("DSTACK_SERVER_ADMIN_TOKEN", "env-token")
    assert utils.load_dashboard_config().dstack_token == "env-token"
