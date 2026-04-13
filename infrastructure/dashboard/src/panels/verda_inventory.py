"""panels/verda_inventory.py — Verda GPU offer inventory table."""

from __future__ import annotations

from ..state import AppState


def format_verda_table(state: AppState) -> list[list[str]]:
    """Return rows for the Verda GPU offer table."""
    with state.lock:
        offers = list(state.verda_offers)

    if not offers:
        return [["(no offers)", "", "", "", ""]]

    rows = []
    for o in sorted(offers, key=lambda x: x.price_per_hour):
        rows.append(
            [
                o.gpu_name,
                f"${o.price_per_hour:.3f}/hr",
                o.backend,
                o.region,
                o.instance_type,
            ]
        )
    return rows
