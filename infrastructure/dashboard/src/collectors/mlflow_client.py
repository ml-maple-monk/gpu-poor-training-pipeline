"""collectors/mlflow_client.py — Collect MLflow run data and live metrics."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import requests

from ..config import MLFLOW_URL
from ..errors import SourceStatus
from ..state import MLflowRun, TrainingSnapshot

log = logging.getLogger(__name__)

_MLFLOW_API = f"{MLFLOW_URL}/api/2.0/mlflow"


def _ts_ms_to_dt(ts_ms: int | None) -> datetime | None:
    if ts_ms is None:
        return None
    return datetime.utcfromtimestamp(ts_ms / 1000)


def collect_mlflow_recent(max_results: int = 20) -> tuple[list[MLflowRun], SourceStatus]:
    """Fetch recent MLflow runs across all experiments."""
    try:
        resp = requests.post(
            f"{_MLFLOW_API}/runs/search",
            json={"max_results": max_results, "order_by": ["start_time DESC"]},
            timeout=5,
        )
        resp.raise_for_status()
        runs_raw = resp.json().get("runs", [])
        runs: list[MLflowRun] = []
        for r in runs_raw:
            info = r.get("info", {})
            metrics: dict[str, float] = {}
            for m in r.get("data", {}).get("metrics", []):
                metrics[m["key"]] = m["value"]
            params: dict[str, str] = {}
            for p in r.get("data", {}).get("params", []):
                params[p["key"]] = p["value"]
            tags: dict[str, str] = {}
            for t in r.get("data", {}).get("tags", []):
                tags[t["key"]] = t["value"]
            runs.append(
                MLflowRun(
                    run_id=info.get("run_id", ""),
                    run_name=info.get("run_name", ""),
                    experiment_id=info.get("experiment_id", ""),
                    status=info.get("status", ""),
                    start_time=_ts_ms_to_dt(info.get("start_time")),
                    end_time=_ts_ms_to_dt(info.get("end_time")),
                    metrics=metrics,
                    params=params,
                    tags=tags,
                )
            )
        return runs, SourceStatus.OK
    except Exception as exc:
        log.warning("mlflow_recent collect failed: %s", exc)
        return [], SourceStatus.ERROR


def collect_live_metrics(
    run_id: str | None = None,
) -> tuple[dict[str, list[tuple[int, float]]], SourceStatus]:
    """Fetch metric history for an active run. Returns {metric_name: [(step, value), ...]}."""
    if not run_id:
        # Try to find the most recent RUNNING run
        try:
            resp = requests.post(
                f"{_MLFLOW_API}/runs/search",
                json={"filter": "attributes.status = 'RUNNING'", "max_results": 1},
                timeout=5,
            )
            resp.raise_for_status()
            runs_raw = resp.json().get("runs", [])
            if not runs_raw:
                return {}, SourceStatus.OK
            run_id = runs_raw[0]["info"]["run_id"]
        except Exception as exc:
            log.warning("mlflow live_metrics: could not find running run: %s", exc)
            return {}, SourceStatus.ERROR

    try:
        # Get the run to find metric keys
        resp = requests.get(
            f"{_MLFLOW_API}/runs/get",
            params={"run_id": run_id},
            timeout=5,
        )
        resp.raise_for_status()
        run_data = resp.json().get("run", {})
        metric_keys = [m["key"] for m in run_data.get("data", {}).get("metrics", [])]

        result: dict[str, list[tuple[int, float]]] = {}
        for key in metric_keys[:10]:  # cap at 10 metrics
            try:
                hist_resp = requests.get(
                    f"{_MLFLOW_API}/metrics/get-history",
                    params={"run_id": run_id, "metric_key": key, "max_results": 200},
                    timeout=5,
                )
                hist_resp.raise_for_status()
                history = hist_resp.json().get("metrics", [])
                result[key] = [(m.get("step", i), m["value"]) for i, m in enumerate(history)]
            except Exception:
                pass
        return result, SourceStatus.OK
    except Exception as exc:
        log.warning("mlflow live_metrics collect failed: %s", exc)
        return {}, SourceStatus.ERROR
