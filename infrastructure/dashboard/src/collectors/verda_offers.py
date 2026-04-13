"""collectors/verda_offers.py — Collect Verda GPU offer pricing via dstack REST."""

from __future__ import annotations

import logging
from typing import Any

from ..errors import SourceStatus
from ..safe_exec import safe_dstack_rest
from ..state import VerdaOffer

log = logging.getLogger(__name__)

# GPU names to query pricing for
GPU_NAMES = [
    "A100",
    "A10",
    "H100",
    "RTX 4090",
    "RTX 3090",
    "T4",
    "L4",
    "V100",
]


# doc-anchor: verda-offers-busybox
def collect_verda_offers() -> tuple[list[VerdaOffer], SourceStatus]:
    """Collect GPU offers from dstack via get_plan endpoint."""
    all_offers: list[VerdaOffer] = []
    last_status = SourceStatus.OK

    for gpu_name in GPU_NAMES:
        try:
            resp = safe_dstack_rest(
                "runs/get_plan",
                method="POST",
                json={
                    "run_spec": {
                        "configuration_path": "offers-probe",
                        "configuration": {
                            "type": "task",
                            "image": "busybox",
                            "commands": ["true"],
                            "resources": {"gpu": {"name": gpu_name, "count": 1}},
                            "spot_policy": "auto",
                        },
                        "repo_id": "offers-probe",
                        "repo_data": {"repo_type": "virtual"},
                    }
                },
                timeout=10.0,
            )
            data = resp.json()
            offers_raw: list[dict[str, Any]] = data if isinstance(data, list) else data.get("job_plans", [])
            for o in offers_raw:
                instance = o.get("instance", {}) or {}
                all_offers.append(
                    VerdaOffer(
                        gpu_name=gpu_name,
                        price_per_hour=float(o.get("price", 0.0) or instance.get("price", 0.0) or 0.0),
                        region=instance.get("region", ""),
                        backend=instance.get("backend", ""),
                        instance_type=instance.get("name", ""),
                        availability="available",
                        raw=o,
                    )
                )
        except Exception as exc:
            log.debug("verda offers for %s failed: %s", gpu_name, exc)
            last_status = SourceStatus.STALE

    if not all_offers and last_status == SourceStatus.STALE:
        return [], SourceStatus.ERROR
    return all_offers, last_status
