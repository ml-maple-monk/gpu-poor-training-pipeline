"""overview.py -- Glance-first HTML sections for the dashboard shell."""

from __future__ import annotations

from datetime import datetime
from html import escape
from urllib.parse import urlparse

from ..collectors.verda_offers import GPU_SPECS
from ..config import DEFAULT_PLATFORM_COLOR, GPU_COLORS, MLFLOW_URL, PLATFORM_COLORS
from ..state import AppState

TARGET_GPUS = [display_name for _, display_name, _ in GPU_SPECS]


def _badge_class(status: str) -> str:
    normalized = (status or "").strip().lower()
    if normalized in {"running", "submitted", "ok", "healthy", "success", "finished"}:
        return "vd-badge-success"
    if normalized in {"provisioning", "starting", "pending", "stale"}:
        return "vd-badge-pending"
    if normalized in {"failed", "error", "cancelled", "not_found"}:
        return "vd-badge-error"
    return "vd-badge-idle"


def _dot_class(status: str) -> str:
    normalized = (status or "").strip().lower()
    if normalized in {"ok", "running", "success", "submitted"}:
        return "vd-dot-ok"
    if normalized in {"stale", "pending", "provisioning", "starting"}:
        return "vd-dot-warn"
    if normalized in {"error", "failed", "cancelled", "not_found"}:
        return "vd-dot-error"
    return "vd-dot-idle"


def _progress_class(percent: float) -> str:
    if percent >= 85:
        return "vd-progress-red"
    if percent >= 60:
        return "vd-progress-yellow"
    return "vd-progress-green"


def _clamp_percent(value: float, total: float | None = None) -> float:
    if total and total > 0:
        value = (value / total) * 100.0
    return max(0.0, min(float(value), 100.0))


def _fmt_money(value: float) -> str:
    if value <= 0:
        return "n/a"
    return f"${value:.3f}/hr"


def _fmt_url(url: str) -> str:
    if not url:
        return "offline"
    try:
        parsed = urlparse(url)
        host = parsed.netloc or parsed.path
    except ValueError:
        host = url
    return host or url


def _fmt_timestamp(ts: datetime | None) -> str:
    if ts is None:
        return "never"
    return ts.strftime("%H:%M:%S")


def _safe(text: str | int | float | None) -> str:
    return escape("" if text is None else str(text))


# Platform-specific brand colors for visual differentiation
_PLATFORM_COLORS = PLATFORM_COLORS
_DEFAULT_PLATFORM_COLOR = DEFAULT_PLATFORM_COLOR


def _platform_color(backend: str) -> tuple[str, str, str]:
    return _PLATFORM_COLORS.get(backend.lower().strip(), _DEFAULT_PLATFORM_COLOR)


def _render_availability_heatmap(history: list, *, points: int = 30, color: str = "#3fb950") -> str:
    """Render a horizontal heatmap bar: colored blocks = available, dark = unavailable."""
    samples = history[-points:] if history else []
    available_flags = [s.available for s in samples] if samples else []
    # Pad left with False if not enough data
    if len(available_flags) < points:
        available_flags = ([False] * (points - len(available_flags))) + available_flags

    bar_w = 4
    bar_h = 14
    gap = 1
    total_w = points * (bar_w + gap)

    rects = []
    for idx, avail in enumerate(available_flags):
        x = idx * (bar_w + gap)
        fill = color if avail else "#21262d"
        opacity = "1" if avail else "0.4"
        rects.append(
            f"<rect x='{x}' y='0' width='{bar_w}' height='{bar_h}' rx='1' fill='{fill}' opacity='{opacity}'/>"
        )

    avail_count = sum(available_flags)
    pct = int(avail_count / points * 100) if points else 0

    return (
        f"<div class='vd-heatmap-row'>"
        f"<svg class='vd-heatmap-svg' viewBox='0 0 {total_w} {bar_h}' preserveAspectRatio='none'>"
        f"{''.join(rects)}</svg>"
        f"<span class='vd-heatmap-pct' style='color:{color}'>{pct}%</span>"
        f"</div>"
    )


def _render_backend_row(offer, history: list) -> str:
    status = offer.availability or "unknown"
    primary, _, label = _platform_color(offer.backend)
    dot = "vd-dot-ok" if status == "available" else "vd-dot-idle"

    return (
        f"<div class='vd-gpu-backend-row' style='border-left: 3px solid {primary}'>"
        f"<div class='vd-gpu-backend-meta'>"
        f"<div class='vd-gpu-backend-name' style='color:{primary}'>"
        f"<span class='vd-dot {dot}' style='background:{primary};box-shadow:0 0 6px {primary}40'></span>"
        f"{_safe(label)}</div>"
        f"<div class='vd-gpu-backend-context'>"
        f"{_safe(offer.region or 'n/a')} · {_safe(offer.mode or 'n/a')} · {_safe(offer.instance_type or 'n/a')}"
        f"</div>"
        f"</div>"
        f"<div class='vd-gpu-backend-price' style='color:{primary}'>{_safe(_fmt_money(offer.price_per_hour))}</div>"
        f"{_render_availability_heatmap(history, color=primary)}"
        f"</div>"
    )


def _render_gpu_card(gpu_name: str, offers: list, history_by_backend: dict[str, list]) -> str:
    available_count = sum(offer.availability == "available" for offer in offers)
    header = f"{available_count}/{len(offers)}" if offers else "0/0"
    card_class = "vd-gpu-card" if offers else "vd-gpu-card vd-gpu-card-empty"

    if not offers:
        return (
            f"<div class='{card_class}'>"
            f"<div class='vd-gpu-card-header'>"
            f"<div class='vd-gpu-card-name'>{_safe(gpu_name)}</div>"
            f"<div class='vd-gpu-card-count'>{_safe(header)}</div>"
            f"</div>"
            f"<div class='vd-gpu-card-empty-copy'>No availability</div>"
            f"{_render_availability_heatmap([])}"
            f"</div>"
        )

    rows = [_render_backend_row(offer, history_by_backend.get(offer.backend, [])) for offer in offers]

    # Build stacked availability chart
    chart_rows = []
    all_backends = sorted(set(o.backend for o in offers))
    for backend in all_backends:
        primary, _, label = _platform_color(backend)
        hist = history_by_backend.get(backend, [])
        chart_rows.append(
            f"<div class='vd-stacked-row'>"
            f"<span class='vd-stacked-label' style='color:{primary}'>{_safe(label)}</span>"
            f"{_render_availability_heatmap(hist, color=primary)}"
            f"</div>"
        )

    return (
        f"<div class='{card_class}'>"
        f"<div class='vd-gpu-card-header'>"
        f"<div class='vd-gpu-card-name'>{_safe(gpu_name)}</div>"
        f"<div class='vd-gpu-card-count'>{_safe(header)}</div>"
        f"</div>"
        f"<div class='vd-gpu-backend-list'>{''.join(rows)}</div>"
        f"<div class='vd-stacked-chart'>"
        f"<div class='vd-stacked-title'>Availability (last 30 min)</div>"
        f"{''.join(chart_rows)}"
        f"</div>"
        f"</div>"
    )


def format_availability_matrix_html(state: AppState, *, slots: int = 30) -> str:
    """Render two sections:
    1. Five large status blocks — lit up if at least 1 GPU is deployable, dimmed otherwise
    2. Five SVG line charts — one per GPU showing per-platform availability over time
    """
    with state.lock:
        history = {k: list(v) for k, v in state.offer_history.items()}
        all_offers = list(state.seeker_offers) + list(state.dstack_probe_offers)

    # ── Section 1: Status blocks ─────────────────────────────────────
    # GPU-specific colors for the blocks
    _GPU_COLORS = GPU_COLORS

    blocks = []
    for gpu_name in TARGET_GPUS:
        color = _GPU_COLORS.get(gpu_name, "#8b949e")

        # Check current availability: any offer with this gpu_name that is available?
        available_offers = [o for o in all_offers if o.gpu_name == gpu_name and o.availability == "available"]
        is_available = len(available_offers) > 0
        offer_count = len(available_offers)

        # Find available backends
        avail_backends = sorted(set(o.backend for o in available_offers))
        backend_chips = ""
        for b in avail_backends:
            b_color, _, b_label = _platform_color(b)
            backend_chips += (
                f"<span class='vd-block-chip' style='color:{b_color};border-color:{b_color}40'>{_safe(b_label)}</span>"
            )

        # Find last-available timestamp if currently unavailable
        last_available_text = ""
        if not is_available:
            # Search history for the most recent available snapshot
            backend_keys = [k for k in history if k[0] == gpu_name]
            latest_ts = None
            for k in backend_keys:
                for snap in reversed(history[k]):
                    if snap.available:
                        if latest_ts is None or snap.timestamp > latest_ts:
                            latest_ts = snap.timestamp
                        break
            if latest_ts is not None:
                last_available_text = (
                    f"<div class='vd-block-last'>Last seen: {_safe(latest_ts.strftime('%H:%M:%S UTC'))}</div>"
                )
            else:
                last_available_text = "<div class='vd-block-last'>No recent availability</div>"

        if is_available:
            blocks.append(
                f"<div class='vd-gpu-block vd-gpu-block-lit' style='border-color:{color};box-shadow:0 0 20px {color}30'>"
                f"<div class='vd-block-name' style='color:{color}'>{_safe(gpu_name)}</div>"
                f"<div class='vd-block-status'>AVAILABLE</div>"
                f"<div class='vd-block-count'>{offer_count} instance{'s' if offer_count != 1 else ''}</div>"
                f"<div class='vd-block-backends'>{backend_chips}</div>"
                f"</div>"
            )
        else:
            blocks.append(
                f"<div class='vd-gpu-block vd-gpu-block-dim'>"
                f"<div class='vd-block-name'>{_safe(gpu_name)}</div>"
                f"<div class='vd-block-status vd-block-status-off'>UNAVAILABLE</div>"
                f"{last_available_text}"
                f"</div>"
            )

    status_section = f"<div class='vd-gpu-blocks'>{''.join(blocks)}</div>"

    # ── Section 2: Historical line charts ────────────────────────────
    # One chart per GPU, showing availability count per platform over time
    charts = []
    all_backends = sorted({k[1] for k in history})

    for gpu_name in TARGET_GPUS:
        color = _GPU_COLORS.get(gpu_name, "#8b949e")

        # Build per-backend time series
        chart_w = 300
        chart_h = 60
        pad_left = 4
        pad_bottom = 12

        plot_w = chart_w - pad_left
        plot_h = chart_h - pad_bottom

        lines_svg = []
        legend_items = []

        for backend in all_backends:
            key = (gpu_name, backend)
            snaps = history.get(key, [])
            b_color, _, b_label = _platform_color(backend)

            if not snaps:
                continue

            # Build points: y=1 if available, y=0 if not
            recent = snaps[-slots:]
            if len(recent) < 2:
                continue

            coords = []
            for i, snap in enumerate(recent):
                x = pad_left + (i / max(len(recent) - 1, 1)) * plot_w
                y = 4 if snap.available else plot_h - 2
                coords.append(f"{x:.1f},{y:.1f}")

            # Area fill under the line
            area_coords = list(coords)
            area_coords.append(f"{pad_left + plot_w:.1f},{plot_h:.1f}")
            area_coords.append(f"{pad_left:.1f},{plot_h:.1f}")

            lines_svg.append(f"<polygon points='{' '.join(area_coords)}' fill='{b_color}' opacity='0.08'/>")
            lines_svg.append(
                f"<polyline points='{' '.join(coords)}' "
                f"fill='none' stroke='{b_color}' stroke-width='2' "
                f"stroke-linecap='round' stroke-linejoin='round'/>"
            )

            legend_items.append(
                f"<span class='vd-chart-legend-item'>"
                f"<span class='vd-chart-legend-dot' style='background:{b_color}'></span>"
                f"{_safe(b_label)}</span>"
            )

        # Time axis
        lines_svg.append(
            f"<line x1='{pad_left}' y1='{plot_h}' x2='{pad_left + plot_w}' y2='{plot_h}' "
            f"stroke='#30363d' stroke-width='0.5'/>"
        )
        # "available" / "unavailable" labels
        lines_svg.append(f"<text x='{pad_left + plot_w + 2}' y='8' class='vd-chart-axis-label'>avail</text>")

        legend_html = f"<div class='vd-chart-legend'>{''.join(legend_items)}</div>" if legend_items else ""

        charts.append(
            f"<div class='vd-history-card'>"
            f"<div class='vd-history-title' style='color:{color}'>{_safe(gpu_name)}</div>"
            f"<svg class='vd-history-svg' viewBox='0 0 {chart_w} {chart_h}' preserveAspectRatio='none'>"
            f"{''.join(lines_svg)}"
            f"</svg>"
            f"{legend_html}"
            f"</div>"
        )

    history_section = (
        f"<div class='vd-history-grid'>{''.join(charts)}</div>"
        if charts
        else "<div class='vd-empty'>Collecting availability history...</div>"
    )

    return f"{status_section}{history_section}"


def format_statusbar_html(state: AppState) -> str:
    """Render the top status strip with live access links and collector health."""
    with state.lock:
        tunnel_url = state.tunnel_url
        health = dict(state.collector_health)
        refreshed = dict(state.last_refreshed_at)

    ok = sum(1 for status in health.values() if status == "ok")
    stale = sum(1 for status in health.values() if status == "stale")
    error = sum(1 for status in health.values() if status == "error")

    health_bits = []
    for name, status in sorted(health.items()):
        dot = _dot_class(status)
        stamp = _fmt_timestamp(refreshed.get(name))
        health_bits.append(
            "<span class='vd-badge vd-badge-idle'>"
            f"<span class='vd-dot {dot}'></span>{_safe(name)} {_safe(status)} {_safe(stamp)}"
            "</span>"
        )

    tunnel_link = (
        f"<a href='{escape(tunnel_url, quote=True)}' target='_blank' rel='noreferrer'>Tunnel {_safe(_fmt_url(tunnel_url))}</a>"
        if tunnel_url
        else "<span class='vd-badge vd-badge-idle'>Tunnel offline</span>"
    )
    mlflow_link = f"<a href='{escape(MLFLOW_URL, quote=True)}' target='_blank' rel='noreferrer'>MLflow {_safe(_fmt_url(MLFLOW_URL))}</a>"

    return f"""
    <div class="vd-statusbar">
      <div>
        <div class="vd-statusbar-title">Verda Dashboard</div>
        <div class="vd-statusbar-time">One-glance capacity and run health</div>
      </div>
      <div class="vd-statusbar-links">
        {tunnel_link}
        {mlflow_link}
      </div>
      <div class="vd-statusbar-health">
        <span class="vd-badge vd-badge-success">ok {ok}</span>
        <span class="vd-badge vd-badge-pending">stale {stale}</span>
        <span class="vd-badge vd-badge-error">error {error}</span>
        {"".join(health_bits[:4])}
      </div>
    </div>
    """


def _merge_offers(state: AppState) -> list:
    """Merge seeker + dstack probe offers, deduplicating by key."""
    combined = list(state.seeker_offers) + list(state.dstack_probe_offers)
    seen: set[tuple[str, str, str, str]] = set()
    deduped = []
    for o in combined:
        key = (o.backend, o.region, o.gpu_name, o.instance_type)
        if key not in seen:
            seen.add(key)
            deduped.append(o)
    return deduped


def format_hero_html(state: AppState) -> str:
    """Render four top hero cards for immediate operational decisions."""
    with state.lock:
        offers = sorted(
            _merge_offers(state),
            key=lambda item: (
                item.gpu_name == "",
                item.availability != "available",
                item.price_per_hour,
                item.backend,
                item.region,
            ),
        )
        active_jobs = list(state.seeker_active_jobs)
        active_job = state.seeker_active
        pending_count = len(state.seeker_pending)
        attempts = list(state.seeker_attempts)
        runs = list(state.dstack_runs)
        active_run_name = state.active_dstack_run or "(none)"
        training = state.training

    gpu_offers = [offer for offer in offers if offer.gpu_name]
    best_offer = next((offer for offer in gpu_offers if offer.availability == "available"), None)
    if best_offer is None:
        best_offer = gpu_offers[0] if gpu_offers else None
    recent_attempt = attempts[-1] if attempts else None
    running_run = next((run for run in runs if run.status.lower() == "running"), runs[0] if runs else None)

    best_offer_card = (
        f"""
        <div class="vd-card">
          <div class="vd-card-accent vd-progress-green"></div>
          <div class="vd-card-title">Best Offer Now</div>
          <div class="vd-card-value">{_safe(best_offer.gpu_name)} · {_safe(best_offer.count)}x</div>
          <div class="vd-card-sub">{_safe(best_offer.backend)} · {_safe(best_offer.region)} · {_safe(best_offer.mode or "mode n/a")}</div>
          <div class="vd-card-sub">{_safe(best_offer.instance_type or "instance n/a")} · <span class="vd-green">{_safe(_fmt_money(best_offer.price_per_hour))}</span></div>
        </div>
        """
        if best_offer
        else """
        <div class="vd-card">
          <div class="vd-card-accent vd-progress-yellow"></div>
          <div class="vd-card-title">Best Offer Now</div>
          <div class="vd-card-value">No capacity</div>
          <div class="vd-card-sub">The seeker has not found any offers yet.</div>
        </div>
        """
    )

    seeker_status = active_job.last_status if active_job and active_job.last_status else "idle"
    seeker_title = active_job.config_name or active_job.job_id if active_job else "No active job"
    seeker_retries = active_job.submit_retries if active_job else 0
    seeker_progress = min(100, pending_count * 18 + seeker_retries * 22)
    seeker_card = f"""
    <div class="vd-card">
      <div class="vd-card-accent vd-progress-blue"></div>
      <div class="vd-card-title">Seeker Queue</div>
      <div class="vd-card-value">{_safe(pending_count)}</div>
      <div class="vd-card-sub">{_safe(seeker_title)}</div>
      <div class="vd-card-sub">{_safe(len(active_jobs))} active jobs</div>
      <div class="vd-card-sub"><span class="vd-badge {_badge_class(seeker_status)}">{_safe(seeker_status or "idle")}</span> · retries {_safe(seeker_retries)}</div>
      <div class="vd-progress vd-progress-lg"><div class="vd-progress-fill vd-progress-blue" style="width: {seeker_progress}%"></div></div>
    </div>
    """

    run_status = running_run.status if running_run else "idle"
    run_cost = _fmt_money(running_run.cost_per_hour) if running_run else "n/a"
    run_card = f"""
    <div class="vd-card">
      <div class="vd-card-accent vd-progress-green"></div>
      <div class="vd-card-title">Active Run</div>
      <div class="vd-card-value">{_safe(active_run_name)}</div>
      <div class="vd-card-sub">{_safe(running_run.backend if running_run else "backend n/a")} · {_safe(running_run.region if running_run else "region n/a")}</div>
      <div class="vd-card-sub"><span class="vd-badge {_badge_class(run_status)}">{_safe(run_status)}</span> · {_safe(run_cost)}</div>
    </div>
    """

    training_status = training.status or "unknown"
    gpu_percent = _clamp_percent(training.gpu_util_percent)
    training_card = f"""
    <div class="vd-card">
      <div class="vd-card-accent vd-progress-yellow"></div>
      <div class="vd-card-title">Training Container</div>
      <div class="vd-card-value">{_safe(training.container_name or "trainer")}</div>
      <div class="vd-card-sub"><span class="vd-badge {_badge_class(training_status)}">{_safe(training_status)}</span> · uptime {_safe(f"{training.uptime_seconds:.0f}s" if training.uptime_seconds > 0 else "n/a")}</div>
      <div class="vd-card-sub">GPU {_safe(f"{training.gpu_util_percent:.0f}%") if training.gpu_mem_total_mb else "n/a"} · VRAM {_safe(f"{training.gpu_mem_used_mb:.0f}/{training.gpu_mem_total_mb:.0f} MB" if training.gpu_mem_total_mb else "n/a")}</div>
      <div class="vd-progress vd-progress-lg"><div class="vd-progress-fill {_progress_class(gpu_percent)}" style="width: {gpu_percent}%"></div></div>
    </div>
    """

    if recent_attempt is not None:
        attempt_chip = (
            f"<div class='vd-card-sub'>Latest decision: "
            f"<span class='vd-badge {_badge_class(recent_attempt.status)}'>{_safe(recent_attempt.status)}</span> "
            f"{_safe(recent_attempt.backend)} {_safe(recent_attempt.gpu)} {_safe(_fmt_money(recent_attempt.price_per_hour))}</div>"
        )
    else:
        attempt_chip = "<div class='vd-card-sub'>Latest decision: none yet</div>"

    return f"<div class='vd-hero'>{best_offer_card}{seeker_card}{run_card}{training_card}</div>{attempt_chip}"


def format_resource_gauges_html(state: AppState) -> str:
    """Render compact CPU/RAM/GPU gauges below the hero strip."""
    with state.lock:
        sys = state.system

    gauges = [
        (
            "CPU",
            f"{sys.cpu_percent:.1f}%",
            _clamp_percent(sys.cpu_percent),
            f"{sys.cpu_count} cores",
        ),
        (
            "Memory",
            f"{sys.mem_percent:.1f}%",
            _clamp_percent(sys.mem_percent),
            f"{sys.mem_used_gb:.1f}/{sys.mem_total_gb:.1f} GB",
        ),
        (
            "GPU Util",
            f"{sys.gpu_util_percent:.0f}%" if sys.nvidia_smi_available else "n/a",
            _clamp_percent(sys.gpu_util_percent if sys.nvidia_smi_available else 0.0),
            sys.gpu_name or "GPU unavailable",
        ),
        (
            "GPU Memory",
            f"{_clamp_percent(sys.gpu_mem_used_mb, sys.gpu_mem_total_mb):.0f}%" if sys.gpu_mem_total_mb else "n/a",
            _clamp_percent(sys.gpu_mem_used_mb, sys.gpu_mem_total_mb),
            f"{sys.gpu_mem_used_mb:.0f}/{sys.gpu_mem_total_mb:.0f} MB" if sys.gpu_mem_total_mb else "n/a",
        ),
    ]

    parts = []
    for label, value, percent, detail in gauges:
        parts.append(
            f"""
            <div class="vd-gauge">
              <div class="vd-gauge-header">
                <div class="vd-gauge-label">{_safe(label)}</div>
                <div class="vd-gauge-value">{_safe(value)}</div>
              </div>
              <div class="vd-progress"><div class="vd-progress-fill {_progress_class(percent)}" style="width: {percent}%"></div></div>
              <div class="vd-gauge-detail">{_safe(detail)}</div>
            </div>
            """
        )
    return f"<div class='vd-gauges'>{''.join(parts)}</div>"


def format_market_grid_html(state: AppState, limit: int = 6) -> str:
    """Render a fixed GPU-first market board with backend availability history."""
    with state.lock:
        offers = list(_merge_offers(state))
        offer_history = {key: list(history) for key, history in state.offer_history.items()}

    gpu_offers = [offer for offer in offers if offer.gpu_name]
    cpu_only_count = len(offers) - len(gpu_offers)

    available_count = sum(offer.availability == "available" for offer in gpu_offers)
    backend_counts: dict[str, int] = {}
    for offer in gpu_offers:
        backend_counts[offer.backend] = backend_counts.get(offer.backend, 0) + 1

    chips = [
        f"<span class='vd-badge vd-badge-success'>gpu listings {len(gpu_offers)}</span>",
        f"<span class='vd-badge vd-badge-pending'>available now {available_count}</span>",
        *[
            f"<span class='vd-badge vd-badge-idle'>{_safe(backend)} {count}</span>"
            for backend, count in sorted(backend_counts.items())
        ],
    ]

    offers_by_gpu: dict[str, list] = {gpu_name: [] for gpu_name in TARGET_GPUS}
    for offer in gpu_offers:
        if offer.gpu_name in offers_by_gpu:
            offers_by_gpu[offer.gpu_name].append(offer)

    cards = []
    for gpu_name in TARGET_GPUS:
        offers_for_gpu = offers_by_gpu[gpu_name]
        history_by_backend = {
            offer.backend: offer_history.get((gpu_name, offer.backend), []) for offer in offers_for_gpu
        }
        cards.append(_render_gpu_card(gpu_name, offers_for_gpu, history_by_backend))

    cpu_note = (
        f"<div class='vd-card-sub' style='margin-top: 10px;'>Hidden CPU-only rows: {_safe(cpu_only_count)}</div>"
        if cpu_only_count
        else ""
    )
    return (
        f"<div class='vd-market-chips'>{''.join(chips)}</div>"
        f"<div class='vd-market-row'>{''.join(cards)}</div>"
        f"{cpu_note}"
    )


def format_alert_feed_html(state: AppState, limit: int = 6) -> str:
    """Render recent failures, pending work, and stale collector signals."""
    with state.lock:
        attempts = list(state.seeker_attempts)
        health = dict(state.collector_health)
        refreshed = dict(state.last_refreshed_at)
        active_jobs = list(state.seeker_active_jobs)
        active_job = state.seeker_active
        pending_count = len(state.seeker_pending)

    items: list[str] = []
    for name, status in sorted(health.items()):
        if status == "ok":
            continue
        items.append(
            f"""
            <div class="vd-feed-item">
              <span class="vd-dot {_dot_class(status)}"></span>
              <div class="vd-feed-name">{_safe(name)}</div>
              <div class="vd-feed-detail">collector {_safe(status)} · last refresh {_safe(_fmt_timestamp(refreshed.get(name)))}</div>
            </div>
            """
        )

    if active_job and active_job.last_reason:
        items.append(
            f"""
            <div class="vd-feed-item">
              <span class="vd-dot {_dot_class(active_job.last_status)}"></span>
              <div class="vd-feed-name">{_safe(active_job.config_name or active_job.job_id)}</div>
              <div class="vd-feed-detail">{_safe(active_job.last_reason)} · retries {_safe(active_job.submit_retries)}</div>
            </div>
            """
        )

    if pending_count:
        items.append(
            f"""
            <div class="vd-feed-item">
              <span class="vd-dot vd-dot-warn"></span>
              <div class="vd-feed-name">queue</div>
              <div class="vd-feed-detail">{_safe(pending_count)} pending · {_safe(len(active_jobs))} active</div>
            </div>
            """
        )

    for attempt in reversed(attempts[-limit:]):
        reason = attempt.reason or "no reason recorded"
        items.append(
            f"""
            <div class="vd-feed-item">
              <span class="vd-dot {_dot_class(attempt.status)}"></span>
              <div class="vd-feed-name">{_safe(attempt.backend or "backend")}</div>
              <div class="vd-feed-detail">{_safe(attempt.status)} · {_safe(attempt.gpu)} · {_safe(reason)}</div>
              <div class="vd-feed-metric">{_safe(_fmt_money(attempt.price_per_hour))}</div>
            </div>
            """
        )

    if not items:
        return "<div class='vd-feed'><div class='vd-feed-empty'>No active alerts. Collectors are healthy and no seeker failures are recorded.</div></div>"

    return f"<div class='vd-feed'>{''.join(items[:limit])}</div>"


def format_metrics_html(state: AppState, limit: int = 6) -> str:
    """Render compact metric tiles for the active MLflow stream."""
    with state.lock:
        metrics = dict(state.live_metrics)

    if not metrics:
        return "<div class='vd-empty'>No live metrics yet. Waiting for an active MLflow run.</div>"

    blocks = []
    for name, points in sorted(metrics.items())[:limit]:
        if not points:
            continue
        step, value = points[-1]
        blocks.append(
            f"""
            <div class="vd-metric">
              <div class="vd-metric-name">{_safe(name)}</div>
              <div class="vd-metric-value">{_safe(f"{value:.5g}")}</div>
              <div class="vd-metric-step">step {_safe(step)}</div>
            </div>
            """
        )
    content = "".join(blocks) if blocks else "<div class='vd-empty'>No metric points yet.</div>"
    return f"<div class='vd-metrics'>{content}</div>"


def format_mlflow_feed_html(state: AppState, limit: int = 5) -> str:
    """Render recent MLflow runs as a compact activity feed."""
    with state.lock:
        runs = list(state.mlflow_runs)

    if not runs:
        return "<div class='vd-feed'><div class='vd-feed-empty'>No recent MLflow runs.</div></div>"

    items = []
    for run in runs[:limit]:
        metrics = " · ".join(f"{key}={value:.4g}" for key, value in list(run.metrics.items())[:2])
        run_name = run.run_name or (run.run_id[:8] if run.run_id else "run")
        started = run.start_time.strftime("%H:%M") if run.start_time else "--:--"
        items.append(
            f"""
            <div class="vd-feed-item">
              <span class="vd-dot {_dot_class(run.status)}"></span>
              <div class="vd-feed-name">{_safe(run_name)}</div>
              <div class="vd-feed-detail"><span class="vd-badge {_badge_class(run.status)}">{_safe(run.status)}</span> · {_safe(started)} · {_safe(metrics or "no metrics yet")}</div>
            </div>
            """
        )
    return f"<div class='vd-feed'>{''.join(items)}</div>"


def format_runs_feed_html(state: AppState, limit: int = 5) -> str:
    """Render dstack runs as a compact feed below the fold."""
    with state.lock:
        runs = list(state.dstack_runs)
        active = state.active_dstack_run or "(none)"

    if not runs:
        return "<div class='vd-feed'><div class='vd-feed-empty'>No dstack runs discovered.</div></div>"

    items = []
    for run in runs[:limit]:
        cost = _fmt_money(run.cost_per_hour)
        gpu_count = f"{run.gpu_count}x GPU" if run.gpu_count else "GPU n/a"
        items.append(
            f"""
            <div class="vd-feed-item">
              <span class="vd-dot {_dot_class(run.status)}"></span>
              <div class="vd-feed-name">{_safe(run.run_name)}</div>
              <div class="vd-feed-detail">{_safe(run.backend)} · {_safe(run.region)} · {_safe(run.instance_type or "instance n/a")} · {_safe(gpu_count)}</div>
              <div class="vd-feed-metric">{_safe(cost)}</div>
            </div>
            """
        )

    return (
        f"<div class='vd-card'><div class='vd-card-title'>Tracked dstack runs</div>"
        f"<div class='vd-card-sub'>Active run: {_safe(active)}</div></div>"
        f"<div class='vd-feed'>{''.join(items)}</div>"
    )


def get_active_run_name(state: AppState) -> str:
    """Return the current run name tracked for remote log streaming."""
    with state.lock:
        return state.active_dstack_run or "(none)"


def format_seeker_offer_table(state: AppState) -> list[list[str]]:
    """Return raw seeker offers for the hidden diagnostics table."""
    with state.lock:
        offers = sorted(_merge_offers(state), key=lambda item: (item.price_per_hour, item.backend, item.region))

    if not offers:
        return [["(no offers)", "", "", "", "", "", ""]]

    return [
        [
            offer.backend,
            offer.region,
            offer.gpu_name,
            str(offer.count),
            offer.mode,
            f"${offer.price_per_hour:.3f}/hr",
            offer.instance_type,
        ]
        for offer in offers
    ]


def format_seeker_attempt_table(state: AppState) -> list[list[str]]:
    """Return raw seeker attempts for the hidden diagnostics table."""
    with state.lock:
        attempts = list(state.seeker_attempts)

    if not attempts:
        return [["(no attempts)", "", "", "", "", "", ""]]

    return [
        [
            attempt.status,
            attempt.job_id,
            attempt.backend or "-",
            attempt.region or "-",
            attempt.gpu or "-",
            f"${attempt.price_per_hour:.3f}/hr" if attempt.price_per_hour else "-",
            attempt.reason or "-",
        ]
        for attempt in reversed(attempts)
    ]
