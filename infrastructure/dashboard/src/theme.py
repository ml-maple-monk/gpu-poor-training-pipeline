"""theme.py -- Dark theme CSS and Gradio theme for the Verda Dashboard.

Provides DASHBOARD_CSS (injected via gr.Blocks(css=...)) and build_theme()
for the Gradio theme object.  All custom classes use the ``vd-`` prefix to
avoid collisions with Gradio internals.
"""

from __future__ import annotations


def build_theme():
    """Return a dark Gradio theme tuned for the dashboard."""
    import gradio as gr

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[
            gr.themes.GoogleFont("Sora"),
            gr.themes.GoogleFont("IBM Plex Sans"),
            "ui-sans-serif",
            "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("JetBrains Mono"),
            "SFMono-Regular",
            "Consolas",
            "monospace",
        ],
    )
    theme.set(
        body_background_fill="#0d1117",
        body_background_fill_dark="#0d1117",
        body_text_color="#e6edf3",
        body_text_color_dark="#e6edf3",
        block_background_fill="transparent",
        block_background_fill_dark="transparent",
        block_border_color="transparent",
        block_border_color_dark="transparent",
        block_label_background_fill="#161b22",
        block_label_background_fill_dark="#161b22",
        block_label_text_color="#8b949e",
        block_label_text_color_dark="#8b949e",
        block_title_text_color="#e6edf3",
        block_title_text_color_dark="#e6edf3",
        border_color_primary="#30363d",
        border_color_primary_dark="#30363d",
        input_background_fill="#0d1117",
        input_background_fill_dark="#0d1117",
        input_border_color="#30363d",
        input_border_color_dark="#30363d",
        button_primary_background_fill="#238636",
        button_primary_background_fill_dark="#238636",
    )
    return theme


# ---------------------------------------------------------------------------
# CSS Design System
# ---------------------------------------------------------------------------
#
# Color palette (GitHub-dark inspired):
#   bg-page:      #0d1117     bg-card:     #161b22     bg-elevated: #1c2128
#   border:       #30363d     text-primary: #e6edf3    text-secondary: #8b949e
#   green:        #3fb950     green-bg:    #238636
#   yellow:       #d29922     yellow-bg:   #9e6a03
#   red:          #f85149     red-bg:      #da3633
#   blue:         #58a6ff     blue-bg:     #1f6feb
#   purple:       #bc8cff
#
# Layout classes:
#   .vd-row           flex row with gap
#   .vd-hero          4-column grid for hero cards
#   .vd-card          standard card container
#   .vd-market-grid   auto-fill grid for GPU offer cards
#
# Component classes:
#   .vd-card-title    uppercase label
#   .vd-card-value    big number
#   .vd-card-sub      subtitle / detail text
#   .vd-progress      progress bar track
#   .vd-progress-fill progress bar fill (set width via inline style)
#   .vd-badge-*       status badges (running, finished, failed, pending, idle)
#   .vd-dot-*         status dots (ok, warn, error, idle)
#   .vd-feed-item     activity feed row
#   .vd-section-hdr   section heading
#   .vd-statusbar     top status bar
#   .vd-gauge         single gauge widget
# ---------------------------------------------------------------------------

DASHBOARD_CSS = r"""
/* ── Gradio overrides ───────────────────────────────────────────────── */
.gradio-container {
    background: #0d1117 !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 16px 24px !important;
}
.gradio-container .contain {
    gap: 8px !important;
}
/* Remove default block borders/shadows from HTML components */
.gradio-container .gr-block,
.gradio-container .gr-box,
.gradio-container .gr-panel {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
/* Make gr.Row gaps tighter */
.gradio-container .gr-row {
    gap: 8px !important;
}
.gradio-container details {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 4px 12px 12px;
    margin-top: 6px;
}
.gradio-container summary {
    color: #e6edf3;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
}

/* ── Status bar ─────────────────────────────────────────────────────── */
.vd-statusbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #161b22;
    padding: 10px 20px;
    border-radius: 10px;
    border: 1px solid #30363d;
    margin-bottom: 4px;
}
.vd-statusbar-title {
    font-size: 16px;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.3px;
}
.vd-statusbar-links {
    display: flex;
    gap: 12px;
    align-items: center;
}
.vd-statusbar-links a {
    color: #58a6ff;
    text-decoration: none;
    font-size: 12px;
    padding: 3px 10px;
    border-radius: 6px;
    background: #0d1117;
    border: 1px solid #30363d;
    transition: background 0.15s;
}
.vd-statusbar-links a:hover {
    background: #1c2128;
}
.vd-statusbar-health {
    display: flex;
    gap: 6px;
    align-items: center;
}
.vd-statusbar-time {
    font-size: 12px;
    color: #8b949e;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Cards ──────────────────────────────────────────────────────────── */
.vd-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 20px;
    transition: border-color 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.vd-card:hover {
    border-color: #484f58;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
}
.vd-card-accent {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    border-radius: 10px 10px 0 0;
}
.vd-card-title {
    font-size: 11px;
    text-transform: uppercase;
    color: #8b949e;
    letter-spacing: 1.2px;
    font-weight: 600;
    margin-bottom: 8px;
}
.vd-card-value {
    font-size: 30px;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1.1;
    font-family: 'JetBrains Mono', monospace;
}
.vd-card-sub {
    font-size: 12px;
    color: #8b949e;
    margin-top: 6px;
    line-height: 1.4;
}
.vd-card-icon {
    font-size: 20px;
    margin-bottom: 4px;
}

/* ── Hero grid ──────────────────────────────────────────────────────── */
.vd-hero {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 4px;
}
@media (max-width: 900px) {
    .vd-hero { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 520px) {
    .vd-hero { grid-template-columns: 1fr; }
}

/* ── Progress bars ──────────────────────────────────────────────────── */
.vd-progress {
    height: 8px;
    background: #21262d;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 10px;
}
.vd-progress-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}
.vd-progress-green  { background: linear-gradient(90deg, #238636, #3fb950); }
.vd-progress-yellow { background: linear-gradient(90deg, #9e6a03, #d29922); }
.vd-progress-red    { background: linear-gradient(90deg, #da3633, #f85149); }
.vd-progress-blue   { background: linear-gradient(90deg, #1f6feb, #58a6ff); }

/* Large progress variant for hero cards */
.vd-progress-lg {
    height: 10px;
    margin-top: 12px;
}

/* ── Gauge row ──────────────────────────────────────────────────────── */
.vd-gauges {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 4px;
}
.vd-gauge {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 16px;
}
.vd-gauge-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.vd-gauge-label {
    font-size: 11px;
    text-transform: uppercase;
    color: #8b949e;
    letter-spacing: 1px;
    font-weight: 600;
}
.vd-gauge-value {
    font-size: 16px;
    font-weight: 700;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
}
.vd-gauge-detail {
    font-size: 11px;
    color: #484f58;
    margin-top: 4px;
}
.vd-gauge .vd-progress { height: 6px; margin-top: 8px; }

/* ── Status dots ────────────────────────────────────────────────────── */
.vd-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 4px;
    vertical-align: middle;
}
.vd-dot-ok    { background: #3fb950; box-shadow: 0 0 6px rgba(63,185,80,0.4); }
.vd-dot-warn  { background: #d29922; box-shadow: 0 0 6px rgba(210,153,34,0.4); }
.vd-dot-error { background: #f85149; box-shadow: 0 0 6px rgba(248,81,73,0.4); }
.vd-dot-idle  { background: #484f58; }

/* Pulse animation for active/running dots */
@keyframes vd-pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.5; }
}
.vd-dot-pulse { animation: vd-pulse 2s ease-in-out infinite; }

/* ── Badges ─────────────────────────────────────────────────────────── */
.vd-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
    vertical-align: middle;
}
.vd-badge-running    { background: #238636; color: #fff; }
.vd-badge-finished   { background: #1f6feb; color: #fff; }
.vd-badge-failed     { background: #da3633; color: #fff; }
.vd-badge-pending    { background: #9e6a03; color: #fff; }
.vd-badge-idle       { background: #21262d; color: #8b949e; }
.vd-badge-success    { background: #238636; color: #fff; }
.vd-badge-error      { background: #da3633; color: #fff; }
.vd-badge-provisioning { background: #1f6feb; color: #fff; }

/* ── Section headers ────────────────────────────────────────────────── */
.vd-section-hdr {
    font-size: 12px;
    text-transform: uppercase;
    color: #8b949e;
    letter-spacing: 1.2px;
    font-weight: 600;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
}

/* ── Market board (GPU offers) ──────────────────────────────────────── */
.vd-market-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin-bottom: 10px;
}
.vd-market-row {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 12px;
    margin-bottom: 8px;
}
.vd-gpu-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-top: 3px solid #3fb950;
    border-radius: 10px;
    padding: 14px 14px 12px;
    min-height: 180px;
}
.vd-gpu-card.vd-gpu-card-empty {
    border-top-color: #484f58;
}
.vd-gpu-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 10px;
}
.vd-gpu-card-name {
    font-size: 18px;
    font-weight: 700;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
}
.vd-gpu-card-count {
    font-size: 11px;
    color: #8b949e;
    font-family: 'JetBrains Mono', monospace;
}
.vd-gpu-backend-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.vd-gpu-backend-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    border-radius: 6px;
    background: #0d1117;
    margin-bottom: 6px;
    transition: background 0.15s;
}
.vd-gpu-backend-row:hover {
    background: #1c2128;
}
.vd-gpu-backend-row:last-child { margin-bottom: 0; }
.vd-gpu-backend-meta {
    flex: 1;
    min-width: 0;
}
.vd-gpu-backend-name {
    font-size: 13px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 6px;
}
.vd-gpu-backend-context {
    font-size: 11px;
    color: #8b949e;
    margin-top: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.vd-gpu-backend-price {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 15px;
    white-space: nowrap;
    min-width: 80px;
    text-align: right;
}

/* ── Heatmap availability bars ────────────────────────────────────── */
.vd-heatmap-row {
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
}
.vd-heatmap-svg {
    flex: 1;
    height: 14px;
    min-width: 60px;
    border-radius: 2px;
}
.vd-heatmap-pct {
    font-size: 11px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    min-width: 32px;
    text-align: right;
}

/* ── Stacked availability chart (bottom of GPU card) ──────────────── */
.vd-stacked-chart {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #21262d;
}
.vd-stacked-title {
    font-size: 10px;
    text-transform: uppercase;
    color: #484f58;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 6px;
}
.vd-stacked-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}
.vd-stacked-row:last-child { margin-bottom: 0; }
.vd-stacked-label {
    font-size: 10px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    min-width: 48px;
    text-transform: lowercase;
}

.vd-gpu-card-empty-copy {
    color: #484f58;
    font-style: italic;
    font-size: 13px;
    padding: 16px 0;
    text-align: center;
}
@media (max-width: 700px) {
    .vd-market-row {
        grid-template-columns: 1fr;
    }
    .vd-gpu-backend-row {
        flex-wrap: wrap;
    }
}

/* ── Activity feed ──────────────────────────────────────────────────── */
.vd-feed {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 16px;
}
.vd-feed-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #21262d;
    font-size: 13px;
}
.vd-feed-item:last-child { border-bottom: none; }
.vd-feed-name {
    color: #e6edf3;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    min-width: 120px;
}
.vd-feed-detail {
    color: #8b949e;
    font-size: 12px;
    flex: 1;
}
.vd-feed-metric {
    color: #58a6ff;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
}
.vd-feed-empty {
    color: #484f58;
    font-style: italic;
    padding: 12px 0;
    font-size: 13px;
}
.vd-statusbar-health {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
}

/* ── Metric cards ───────────────────────────────────────────────────── */
.vd-metrics {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 4px;
}
.vd-metric {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    min-width: 140px;
    flex: 1;
}
.vd-metric-name {
    font-size: 11px;
    text-transform: uppercase;
    color: #8b949e;
    letter-spacing: 0.8px;
    font-weight: 600;
    margin-bottom: 4px;
}
.vd-metric-value {
    font-size: 20px;
    font-weight: 700;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
}
.vd-metric-step {
    font-size: 11px;
    color: #484f58;
    margin-top: 2px;
}

/* ── Seeker status card ─────────────────────────────────────────────── */
.vd-seeker-status {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 10px;
}
.vd-seeker-stat {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}
.vd-seeker-stat-label {
    font-size: 11px;
    text-transform: uppercase;
    color: #8b949e;
    letter-spacing: 1px;
    font-weight: 600;
}
.vd-seeker-stat-value {
    font-size: 24px;
    font-weight: 700;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
}

/* ── Attempt timeline ───────────────────────────────────────────────── */
.vd-attempts {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 8px;
}
.vd-attempt {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #21262d;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11px;
    color: #8b949e;
}
.vd-attempt-success { border-left: 3px solid #3fb950; }
.vd-attempt-failed  { border-left: 3px solid #f85149; }
.vd-attempt-pending { border-left: 3px solid #d29922; }

/* ── Log terminal ───────────────────────────────────────────────────── */
#vd-docker-log textarea,
#vd-dstack-log textarea {
    background: #0d1117 !important;
    color: #7ee787 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
#vd-docker-log label,
#vd-dstack-log label {
    color: #8b949e !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* ── Two-column layout ──────────────────────────────────────────────── */
.vd-split {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 4px;
}
@media (max-width: 768px) {
    .vd-split { grid-template-columns: 1fr; }
}

/* ── Footer ─────────────────────────────────────────────────────────── */
.vd-footer {
    text-align: center;
    font-size: 12px;
    color: #484f58;
    padding: 12px 0;
    border-top: 1px solid #21262d;
    margin-top: 8px;
}

/* ── Empty state ────────────────────────────────────────────────────── */
.vd-empty {
    color: #484f58;
    font-style: italic;
    font-size: 13px;
    padding: 20px;
    text-align: center;
}

/* ── Utility ────────────────────────────────────────────────────────── */
.vd-mono { font-family: 'JetBrains Mono', monospace; }
.vd-green  { color: #3fb950; }
.vd-yellow { color: #d29922; }
.vd-red    { color: #f85149; }
.vd-blue   { color: #58a6ff; }
.vd-muted  { color: #8b949e; }
.vd-truncate {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ═══════════════════════════════════════════════════════════════════════
   Section-card design system (used by ui.py helpers + _glance/_md fns)
   ═══════════════════════════════════════════════════════════════════════ */

/* ── Tone pills (badges) ───────────────────────────────────────────── */
.tone-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
    vertical-align: middle;
}
.tone-good    { background: #238636; color: #fff; }
.tone-bad     { background: #da3633; color: #fff; }
.tone-warn    { background: #9e6a03; color: #fff; }
.tone-neutral { background: #21262d; color: #8b949e; }
.tone-pill.is-subtle {
    background: transparent;
    border: 1px solid #30363d;
    font-size: 10px;
    padding: 1px 8px;
}
.tone-pill.is-subtle.tone-good    { color: #3fb950; border-color: #238636; }
.tone-pill.is-subtle.tone-bad     { color: #f85149; border-color: #da3633; }
.tone-pill.is-subtle.tone-warn    { color: #d29922; border-color: #9e6a03; }
.tone-pill.is-subtle.tone-neutral { color: #8b949e; border-color: #30363d; }

/* ── Meta rows (label/value pairs) ─────────────────────────────────── */
.meta-stack {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 16px;
    margin: 8px 0;
}
.meta-row {
    display: flex;
    gap: 6px;
    align-items: baseline;
    font-size: 12px;
}
.meta-label {
    color: #8b949e;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600;
    white-space: nowrap;
}
.meta-value {
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}

/* ── Progress rows ─────────────────────────────────────────────────── */
.progress-row {
    margin: 6px 0;
}
.progress-copy {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 4px;
}
.progress-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #8b949e;
    font-weight: 600;
}
.progress-value {
    font-size: 12px;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
}
.progress-track {
    height: 6px;
    background: #21262d;
    border-radius: 3px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
}
.progress-fill.tone-good    { background: linear-gradient(90deg, #238636, #3fb950); }
.progress-fill.tone-warn    { background: linear-gradient(90deg, #9e6a03, #d29922); }
.progress-fill.tone-bad     { background: linear-gradient(90deg, #da3633, #f85149); }
.progress-fill.tone-neutral { background: linear-gradient(90deg, #30363d, #484f58); }

/* ── Hero shell (topbar glance) ────────────────────────────────────── */
.hero-shell {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px 14px;
    margin-bottom: 4px;
}
.hero-grid {
    display: grid;
    grid-template-columns: 2fr repeat(3, 1fr);
    gap: 16px;
    align-items: start;
}
@media (max-width: 900px) {
    .hero-grid { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 520px) {
    .hero-grid { grid-template-columns: 1fr; }
}
.hero-copy {
    padding-right: 12px;
}
.hero-eyebrow {
    font-size: 11px;
    text-transform: uppercase;
    color: #8b949e;
    letter-spacing: 1.2px;
    font-weight: 600;
    margin-bottom: 4px;
}
.hero-copy h1 {
    font-size: 22px;
    font-weight: 700;
    color: #e6edf3;
    margin: 0 0 6px;
    line-height: 1.2;
}
.hero-copy p {
    font-size: 13px;
    color: #8b949e;
    margin: 0 0 10px;
    line-height: 1.5;
}
.hero-links {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 16px;
    font-size: 12px;
}
.hero-links a {
    color: #58a6ff;
    text-decoration: none;
    transition: color 0.15s;
}
.hero-links a:hover { color: #79c0ff; }

/* ── Hero stat cards ───────────────────────────────────────────────── */
.hero-stat {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
    transition: border-color 0.2s;
}
.hero-stat:hover { border-color: #484f58; }
.hero-stat.tone-good    { border-top: 3px solid #3fb950; }
.hero-stat.tone-warn    { border-top: 3px solid #d29922; }
.hero-stat.tone-bad     { border-top: 3px solid #f85149; }
.hero-stat.tone-neutral { border-top: 3px solid #484f58; }
.hero-stat-label {
    font-size: 11px;
    text-transform: uppercase;
    color: #8b949e;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 6px;
}
.hero-stat-value {
    font-size: 18px;
    font-weight: 700;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
}
.hero-stat-detail {
    font-size: 11px;
    color: #484f58;
    line-height: 1.3;
}

/* ── Chip strip ────────────────────────────────────────────────────── */
.chip-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #21262d;
}
.chip-strip.compact {
    margin-top: 6px;
    padding-top: 0;
    border-top: none;
}

/* ── Section cards (panel containers) ──────────────────────────────── */
.section-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 4px;
    transition: border-color 0.2s;
}
.section-card:hover { border-color: #484f58; }
.section-kicker {
    font-size: 11px;
    text-transform: uppercase;
    color: #8b949e;
    letter-spacing: 1.2px;
    font-weight: 600;
    margin-bottom: 4px;
}
.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 8px;
}
.section-subtitle {
    font-size: 13px;
    color: #8b949e;
    margin-bottom: 8px;
}
.section-empty {
    color: #484f58;
    font-style: italic;
    font-size: 13px;
    padding: 12px 0;
}

/* Accent borders per section type */
.section-card--training    { border-left: 3px solid #3fb950; }
.section-card--system      { border-left: 3px solid #58a6ff; }
.section-card--metrics     { border-left: 3px solid #bc8cff; }
.section-card--mlflow-summary { border-left: 3px solid #1f6feb; }
.section-card--mlflow-feed { border-left: 3px solid #1f6feb; }
.section-card--dstack      { border-left: 3px solid #d29922; }
.section-card--seeker      { border-left: 3px solid #3fb950; }

/* ── Feed grid (card-based activity) ───────────────────────────────── */
.feed-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 10px;
    margin-top: 10px;
}
.feed-grid.compact-grid {
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 8px;
}
.feed-card {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 14px;
    transition: border-color 0.15s;
}
.feed-card:hover { border-color: #484f58; }
.feed-card.dense { padding: 10px 12px; }
.feed-card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.feed-card-time {
    font-size: 11px;
    color: #484f58;
    font-family: 'JetBrains Mono', monospace;
}
.feed-card-title {
    font-size: 13px;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.feed-card-meta {
    font-size: 11px;
    color: #8b949e;
    margin-bottom: 2px;
}
.feed-card-metrics {
    font-size: 11px;
    color: #58a6ff;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
}
.feed-card-metrics.muted-copy {
    color: #484f58;
    font-style: italic;
    font-family: inherit;
}

/* ── Metric grid ───────────────────────────────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 10px;
    margin-top: 10px;
}
.metric-card {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 14px;
    transition: border-color 0.15s;
}
.metric-card:hover { border-color: #484f58; }
.metric-card-name {
    font-size: 11px;
    text-transform: uppercase;
    color: #8b949e;
    letter-spacing: 0.8px;
    font-weight: 600;
    margin-bottom: 4px;
}
.metric-card-value {
    font-size: 20px;
    font-weight: 700;
    color: #e6edf3;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card-step {
    font-size: 11px;
    color: #484f58;
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── GPU Status Blocks (5 large lit/dim blocks) ─────────────────── */
.vd-gpu-blocks {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 12px;
}
@media (max-width: 900px) {
    .vd-gpu-blocks { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 520px) {
    .vd-gpu-blocks { grid-template-columns: repeat(2, 1fr); }
}
.vd-gpu-block {
    background: #161b22;
    border: 2px solid #30363d;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    transition: border-color 0.3s, box-shadow 0.3s, opacity 0.3s;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 6px;
}
.vd-gpu-block-lit {
    background: #0d1117;
}
.vd-gpu-block-dim {
    opacity: 0.45;
    border-color: #21262d;
}
.vd-block-name {
    font-size: 22px;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #484f58;
    letter-spacing: -0.5px;
}
.vd-gpu-block-lit .vd-block-name {
    color: inherit;
}
.vd-block-status {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #3fb950;
}
.vd-block-status-off {
    color: #f85149;
}
.vd-block-count {
    font-size: 12px;
    color: #8b949e;
    font-family: 'JetBrains Mono', monospace;
}
.vd-block-backends {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 2px;
}
.vd-block-chip {
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 8px;
    border: 1px solid;
}
.vd-block-last {
    font-size: 11px;
    color: #484f58;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2px;
}

/* ── Historical Line Charts ─────────────────────────────────────── */
.vd-history-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 8px;
}
@media (max-width: 1100px) {
    .vd-history-grid { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 700px) {
    .vd-history-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 480px) {
    .vd-history-grid { grid-template-columns: 1fr; }
}
.vd-history-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 14px;
    transition: border-color 0.2s;
}
.vd-history-card:hover {
    border-color: #484f58;
}
.vd-history-title {
    font-size: 13px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 8px;
}
.vd-history-svg {
    display: block;
    width: 100%;
    height: 60px;
    border-radius: 4px;
    background: #0d1117;
}
.vd-chart-axis-label {
    fill: #484f58;
    font-size: 7px;
    font-family: 'JetBrains Mono', monospace;
}
.vd-chart-legend {
    display: flex;
    gap: 8px;
    margin-top: 6px;
    flex-wrap: wrap;
}
.vd-chart-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 10px;
    color: #8b949e;
}
.vd-chart-legend-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
}
"""
