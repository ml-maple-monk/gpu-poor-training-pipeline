"""Seeker market and attempt formatters."""

from __future__ import annotations

from ..state import AppState
from .ui import badge, compact_time, esc, meta, money, tone_for_status, truncate


def _merge_offers(state: AppState) -> list:
    """Merge seeker + dstack probe offers, dedup by key."""
    combined = list(state.seeker_offers) + list(state.dstack_probe_offers)
    seen: set[tuple[str, str, str, str]] = set()
    deduped = []
    for o in combined:
        key = (o.backend, o.region, o.gpu_name, o.instance_type)
        if key not in seen:
            seen.add(key)
            deduped.append(o)
    return deduped


def _sorted_offers(state: AppState):
    with state.lock:
        offers = _merge_offers(state)
    return sorted(
        offers,
        key=lambda item: (
            item.availability != "available",
            item.price_per_hour,
            item.backend,
            item.region,
        ),
    )


def format_seeker_summary_md(state: AppState) -> str:
    """Return a compact seeker mission summary card."""
    with state.lock:
        active_jobs = list(state.seeker_active_jobs)
        active = state.seeker_active
        pending_count = len(state.seeker_pending)
        attempts = list(state.seeker_attempts)
        offers = _merge_offers(state)

    gpu_offers = [offer for offer in offers if offer.gpu_name]
    available_gpu = [offer for offer in gpu_offers if offer.availability == "available"]
    best_live = min(available_gpu, key=lambda offer: offer.price_per_hour) if available_gpu else None

    body = [
        '<section class="section-card section-card--seeker-summary">',
        '<div class="section-kicker">Mission control</div>',
        '<div class="section-title">Seeker status</div>',
        '<div class="chip-strip compact">',
        badge(f"{len(available_gpu)} GPU offers live", tone="good" if available_gpu else "neutral"),
        badge(f"{len(active_jobs)} active", tone="good" if active_jobs else "neutral"),
        badge(f"{pending_count} queued", tone="warn" if pending_count else "neutral"),
        badge(f"{len(attempts)} recent attempts", tone="neutral"),
        "</div>",
        '<div class="meta-stack">',
        meta("Active job", active.config_name or active.job_id if active else "none"),
        meta("Status", active.last_status if active and active.last_status else "idle"),
        meta("Best live price", money(best_live.price_per_hour) if best_live else "n/a"),
        meta("Best location", f"{best_live.backend} · {best_live.region}" if best_live else "n/a"),
        "</div>",
        "</section>",
    ]
    return "".join(body)


def format_seeker_offer_table(state: AppState) -> list[list[str]]:
    """Return detailed seeker offers for diagnostics."""
    offers = _sorted_offers(state)

    if not offers:
        return [["(no offers)", "", "", "", "", "", ""]]

    rows = []
    for offer in offers:
        rows.append(
            [
                offer.backend,
                offer.region,
                offer.gpu_name,
                str(offer.count),
                offer.mode,
                money(offer.price_per_hour),
                offer.instance_type,
            ]
        )
    return rows


def format_seeker_offer_glance(state: AppState, limit: int = 8) -> str:
    """Return the GPU market board without overwhelming tables."""
    offers = _sorted_offers(state)
    gpu_offers = [offer for offer in offers if offer.gpu_name]
    cpu_only_count = len(offers) - len(gpu_offers)

    available_gpu = [offer for offer in gpu_offers if offer.availability == "available"]
    backend_counts = {}
    for offer in available_gpu:
        backend_counts[offer.backend] = backend_counts.get(offer.backend, 0) + 1

    body = [
        '<section class="section-card section-card--market">',
        '<div class="section-kicker">Capacity board</div>',
        '<div class="section-title">Live GPU market</div>',
        '<div class="chip-strip compact">',
        badge(f"{len(gpu_offers)} GPU listings", tone="good" if gpu_offers else "neutral"),
        badge(f"{len(available_gpu)} available now", tone="good" if available_gpu else "warn"),
        badge(f"{len(backend_counts)} backends", tone="neutral"),
        *[
            badge(f"{backend} {count}", tone="neutral", subtle=True)
            for backend, count in sorted(backend_counts.items())
        ],
        "</div>",
    ]

    if not gpu_offers:
        body.append(
            '<div class="section-empty">No GPU offers are visible yet. CPU-only inventory is intentionally hidden from the main board.</div>'
        )
    else:
        body.append('<div class="offer-grid">')
        for offer in gpu_offers[:limit]:
            tone = "good" if offer.availability == "available" else "warn"
            body.append(
                '<article class="offer-card">'
                f'<div class="offer-card-top">{badge(offer.availability or "unknown", tone=tone)}{badge(offer.mode or "n/a", tone="neutral", subtle=True)}</div>'
                f'<div class="offer-card-price">{esc(money(offer.price_per_hour))}</div>'
                f'<div class="offer-card-title">{esc(offer.gpu_name)} · {esc(str(offer.count or 0))}x</div>'
                f'<div class="offer-card-meta">{esc(offer.backend or "-")} · {esc(offer.region or "-")}</div>'
                f'<div class="offer-card-instance">{esc(offer.instance_type or "instance n/a")}</div>'
                "</article>"
            )
        body.append("</div>")

    if cpu_only_count:
        body.append(
            f'<div class="section-note">{cpu_only_count} CPU-only listings are hidden here so the board stays GPU-first.</div>'
        )

    body.append("</section>")
    return "".join(body)


def format_seeker_attempt_table(state: AppState) -> list[list[str]]:
    """Return detailed attempt rows for diagnostics."""
    with state.lock:
        attempts = list(state.seeker_attempts)

    if not attempts:
        return [["(no attempts)", "", "", "", "", "", ""]]

    rows = []
    for attempt in reversed(attempts):
        rows.append(
            [
                attempt.status,
                attempt.job_id,
                attempt.backend or "-",
                attempt.region or "-",
                attempt.gpu or "-",
                money(attempt.price_per_hour, "-"),
                attempt.reason or "-",
            ]
        )
    return rows


def format_seeker_attempt_glance(state: AppState, limit: int = 6) -> str:
    """Return a dense timeline of recent seeker decisions."""
    with state.lock:
        attempts = list(state.seeker_attempts)

    body = [
        '<section class="section-card section-card--attempts">',
        '<div class="section-kicker">Decision trail</div>',
        '<div class="section-title">Recent attempts</div>',
    ]

    if not attempts:
        body.append('<div class="section-empty">No attempt history has been recorded yet.</div>')
    else:
        body.append('<div class="timeline-list">')
        for attempt in reversed(attempts[-limit:]):
            price = money(attempt.price_per_hour, "price n/a")
            body.append(
                '<article class="timeline-card">'
                f'<div class="timeline-top">{badge(attempt.status or "unknown", tone=tone_for_status(attempt.status))}<span class="timeline-time">{esc(compact_time(attempt.ended_at or attempt.started_at))}</span></div>'
                f'<div class="timeline-title">{esc(attempt.gpu or "unknown GPU")} · {esc(attempt.backend or "-")} · {esc(attempt.region or "-")}</div>'
                f'<div class="timeline-meta">{esc(price)} · {esc(str(attempt.count or 0))}x · {esc(attempt.mode or "n/a")}</div>'
                f'<div class="timeline-reason">{esc(truncate(attempt.reason or "No reason recorded."))}</div>'
                "</article>"
            )
        body.append("</div>")

    body.append("</section>")
    return "".join(body)
