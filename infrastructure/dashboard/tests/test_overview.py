"""Tests for the glance-first overview surface."""

from __future__ import annotations

import sys
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.panels.overview import format_hero_html, format_market_grid_html
from src.state import OfferSnapshot, SeekerOffer, reset_state


def test_market_grid_hides_cpu_only_rows_from_primary_board() -> None:
    state = reset_state()
    with state.lock:
        state.seeker_offers = [
            SeekerOffer(
                backend="verda",
                region="FIN-01",
                gpu_name="",
                count=0,
                mode="spot",
                price_per_hour=0.01,
                instance_type="CPU.4V.16G",
                availability="available",
            ),
            SeekerOffer(
                backend="runpod",
                region="US-KS-1",
                gpu_name="H100",
                count=1,
                mode="spot",
                price_per_hour=1.25,
                instance_type="h100x1",
                availability="available",
            ),
        ]

    html = format_market_grid_html(state)

    assert "H100" in html
    assert "CPU.4V.16G" not in html
    assert "Hidden CPU-only rows: 1" in html
    assert "A100-80G" in html
    assert "RTX 5090" in html
    assert html.count("No availability") >= 1


def test_market_grid_renders_fixed_gpu_cards_with_svg_and_repeated_backend_rows() -> None:
    state = reset_state()
    now = datetime.now(UTC)
    with state.lock:
        state.seeker_offers = [
            SeekerOffer(
                backend="vastai",
                region="US-CA",
                gpu_name="H100",
                count=1,
                mode="on-demand",
                price_per_hour=2.10,
                instance_type="h100-on-demand",
                availability="available",
            ),
            SeekerOffer(
                backend="runpod",
                region="US-KS-1",
                gpu_name="H100",
                count=1,
                mode="spot",
                price_per_hour=1.25,
                instance_type="h100-spot-a",
                availability="available",
            ),
            SeekerOffer(
                backend="runpod",
                region="US-KS-2",
                gpu_name="H100",
                count=1,
                mode="spot",
                price_per_hour=1.55,
                instance_type="h100-spot-b",
                availability="available",
            ),
            SeekerOffer(
                backend="verda",
                region="FIN-01",
                gpu_name="H200",
                count=1,
                mode="spot",
                price_per_hour=1.75,
                instance_type="h200-spot",
                availability="available",
            ),
        ]
        state.offer_history[("H100", "vastai")] = deque(
            [
                OfferSnapshot(timestamp=now, available=False),
                OfferSnapshot(timestamp=now, available=True),
            ],
            maxlen=60,
        )
        state.offer_history[("H100", "runpod")] = deque(
            [
                OfferSnapshot(timestamp=now, available=True),
                OfferSnapshot(timestamp=now, available=True),
            ],
            maxlen=60,
        )

    html = format_market_grid_html(state)

    expected_order = ["A100-80G", "H100", "H200", "B200", "RTX 5090"]
    positions = [html.index(gpu) for gpu in expected_order]
    assert positions == sorted(positions)
    assert "vd-heatmap-svg" in html
    assert "vd-offer-card-best" not in html
    assert "h100-spot-a" in html
    assert "h100-spot-b" in html
    assert html.index("h100-on-demand") < html.index("h100-spot-a")
    assert "No availability" in html


def test_hero_best_offer_prefers_gpu_capacity_over_cpu_inventory() -> None:
    state = reset_state()
    with state.lock:
        state.seeker_offers = [
            SeekerOffer(
                backend="verda",
                region="FIN-01",
                gpu_name="",
                count=0,
                mode="spot",
                price_per_hour=0.01,
                instance_type="CPU.4V.16G",
                availability="available",
            ),
            SeekerOffer(
                backend="vastai",
                region="US-CA",
                gpu_name="H100",
                count=1,
                mode="on-demand",
                price_per_hour=2.10,
                instance_type="NVIDIA H100 80GB",
                availability="available",
            ),
        ]

    html = format_hero_html(state)

    assert "Best Offer Now" in html
    assert "H100" in html
    assert "CPU.4V.16G" not in html
