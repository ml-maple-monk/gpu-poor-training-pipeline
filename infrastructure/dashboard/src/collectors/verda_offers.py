"""collectors/verda_offers.py — Collect Verda GPU offer pricing via dstack REST."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..config import GPU_SPECS, MAX_OFFERS_PER_GPU, TIMEOUT_DSTACK_OFFERS
from ..errors import SourceStatus
from ..safe_exec import safe_dstack_rest
from ..state import SeekerOffer

log = logging.getLogger(__name__)

# GPU_SPECS imported from config


# doc-anchor: verda-offers-busybox
def collect_verda_offers() -> tuple[list[SeekerOffer], SourceStatus]:
    """Collect GPU offers from dstack via get_plan endpoint."""
    all_offers: list[SeekerOffer] = []
    last_status = SourceStatus.OK

    for probe_name, display_name, min_memory_mib in GPU_SPECS:
        try:
            resp = safe_dstack_rest(
                "runs/get_plan",
                method="POST",
                json={
                    "run_spec": {
                        "configuration_path": "offers-probe",
                        "configuration": {
                            "type": "task",
                            # scratch + root: prevents dstack from pulling image
                            # config and defaulting gpu.vendor to nvidia, which
                            # would filter out non-Nvidia backends (matches
                            # `dstack offer` CLI behavior).
                            "image": "scratch",
                            "user": "root",
                            "commands": [":"],
                            "resources": {"gpu": {"name": probe_name, "count": 1}},
                            "spot_policy": "auto",
                        },
                        "repo_id": "offers-probe",
                        "repo_data": {"repo_type": "virtual"},
                    },
                    "max_offers": MAX_OFFERS_PER_GPU,
                },
                timeout=TIMEOUT_DSTACK_OFFERS,
            )
            data = resp.json()
            # dstack 0.20+: offers are nested in job_plans[*].offers[*]
            job_plans = data.get("job_plans", []) if isinstance(data, dict) else []
            offers_raw: list[dict[str, Any]] = []
            for plan in job_plans:
                if not isinstance(plan, dict):
                    continue
                for o in plan.get("offers", []):
                    if not isinstance(o, dict):
                        continue
                    offers_raw.append(o)

            selected_offers = _filter_offers_by_memory(offers_raw, min_memory_mib)
            for offer in selected_offers:
                instance = offer.get("instance", {}) or {}
                all_offers.append(
                    SeekerOffer(
                        gpu_name=display_name,
                        price_per_hour=float(offer.get("price", 0.0) or 0.0),
                        region=offer.get("region", "") or instance.get("region", ""),
                        backend=offer.get("backend", ""),
                        instance_type=instance.get("name", ""),
                        availability="available",
                        count=1,
                        mode="spot" if instance.get("resources", {}).get("spot") else "on-demand",
                        raw=offer,
                    )
                )
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            log.debug("verda offers for %s failed: %s", probe_name, exc)
            last_status = SourceStatus.STALE

    if not all_offers and last_status == SourceStatus.STALE:
        return [], SourceStatus.ERROR
    return all_offers, last_status


def _filter_offers_by_memory(offers: list[dict[str, Any]], min_memory_mib: int | None) -> list[dict[str, Any]]:
    """Prefer memory-qualified offers, but fall back to the full list if none match."""
    if min_memory_mib is None:
        return offers

    filtered = [offer for offer in offers if _first_gpu_memory_mib(offer) >= min_memory_mib]
    return filtered or offers


def _first_gpu_memory_mib(offer: dict[str, Any]) -> int:
    instance = offer.get("instance", {}) or {}
    resources = instance.get("resources", {}) or {}
    gpus = resources.get("gpus", []) or [{}]
    first_gpu = gpus[0] if gpus else {}
    try:
        return int(first_gpu.get("memory_mib", 0) or 0)
    except (TypeError, ValueError):
        return 0
