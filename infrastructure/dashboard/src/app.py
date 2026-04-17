"""Dash entrypoint for the hard-pruned availability dashboard."""

from __future__ import annotations

from datetime import UTC

import dash_mantine_components as dmc
from dash import Dash, dcc, html
from dash import Input as DashIn
from dash import Output as DashOut

from .models import DashboardConfig, DashboardSnapshot, GpuCard, LaneSnapshot, ProviderRow, SweepStatus
from .utils import build_dashboard_snapshot, build_history_figure, load_dashboard_config, start_sweep_scheduler


def money_label(value: float | None) -> str:
    if value is None or value <= 0:
        return "n/a"
    return f"${value:.3f}/hr"


def time_label(value) -> str:
    if value is None:
        return "never"
    return value.astimezone(UTC).strftime("%H:%M:%S UTC")


def age_label(seconds: int | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def lane_color(mode: str) -> str:
    if mode == "preemptible":
        return "#59C9A5"
    return "#F4A259"


def shell_theme() -> dict:
    return {
        "fontFamily": "IBM Plex Sans, Avenir Next, Segoe UI, sans-serif",
        "primaryColor": "teal",
        "defaultRadius": "lg",
        "colors": {
            "ink": [
                "#f5f7fb",
                "#d8dde8",
                "#b7c0d0",
                "#8791a3",
                "#677287",
                "#4b5568",
                "#2c3441",
                "#1d222d",
                "#141821",
                "#0d1017",
            ]
        },
    }


def header_kpis(snapshot: DashboardSnapshot) -> list:
    return [
        metric_chip("Preemptible", snapshot.preemptible.live_instance_count, lane_color("preemptible")),
        metric_chip("On-Demand", snapshot.on_demand.live_instance_count, lane_color("on-demand")),
        metric_chip("Hidden Unknown", snapshot.hidden_unknown_count, "#8B949E"),
    ]


def metric_chip(label: str, value, accent: str):
    return dmc.Paper(
        radius="xl",
        p="md",
        withBorder=True,
        style={
            "background": f"linear-gradient(145deg, {accent}22 0%, rgba(13,16,23,0.92) 82%)",
            "borderColor": f"{accent}55",
            "minWidth": "10rem",
        },
        children=dmc.Stack(
            gap=0,
            children=[
                dmc.Text(label, size="xs", tt="uppercase", fw=700, c="dimmed"),
                dmc.Text(str(value), size="xl", fw=800, c=accent),
            ],
        ),
    )


def source_badges(snapshot: DashboardSnapshot) -> list:
    badges: list = []
    for note in snapshot.source_notes:
        color = "gray"
        lowered = note.lower()
        if "ok" in lowered:
            color = "teal"
        elif "error" in lowered:
            color = "red"
        elif "skipped" in lowered or "missing" in lowered:
            color = "yellow"
        badges.append(dmc.Badge(note, color=color, variant="light", radius="sm"))
    if snapshot.hidden_unknown_labels:
        labels = ", ".join(snapshot.hidden_unknown_labels[:3])
        if len(snapshot.hidden_unknown_labels) > 3:
            labels = f"{labels}, +{len(snapshot.hidden_unknown_labels) - 3} more"
        badges.append(
            dmc.Badge(
                f"Hidden modes: {labels}",
                color="gray",
                variant="outline",
                radius="sm",
            )
        )
    return badges


def sweep_badges(sweep: SweepStatus) -> list:
    state_colors = {
        "running": ("blue", "Sweep running"),
        "error": ("red", "Sweep error"),
        "idle": ("teal", "Sweep idle"),
    }
    color, label = state_colors.get(sweep.state, ("gray", "Sweep idle"))
    badges = [
        dmc.Badge(label, color=color, variant="light", radius="sm"),
        dmc.Badge(f"Last success {time_label(sweep.last_success_at)}", color="gray", variant="outline", radius="sm"),
        dmc.Badge(f"Snapshot age {age_label(sweep.snapshot_age_seconds)}", color="gray", variant="outline", radius="sm"),
    ]
    if sweep.state == "running" and sweep.running_since is not None:
        badges.append(dmc.Badge(f"Started {time_label(sweep.running_since)}", color="blue", variant="outline", radius="sm"))
    if sweep.state == "error" and sweep.last_error_text:
        badges.append(dmc.Badge(sweep.last_error_text[:108], color="red", variant="outline", radius="sm"))
    return badges


def provider_row_component(row: ProviderRow):
    header = dmc.Group(
        justify="space-between",
        children=[
            dmc.Group(
                gap="xs",
                children=[
                    dmc.Badge(
                        row.provider_label,
                        variant="outline",
                        radius="sm",
                        style={"borderColor": row.provider_color, "color": row.provider_color},
                    ),
                    dmc.Text(row.regions_label, size="sm", c="dimmed"),
                ],
            ),
            dmc.Group(
                gap="md",
                children=[
                    dmc.Text(f"{row.current_count} live", size="sm", fw=700),
                    dmc.Text(money_label(row.cheapest_price), size="sm", fw=700, c=row.provider_color),
                ],
            ),
        ],
    )
    subline = dmc.Group(
        justify="space-between",
        children=[
            dmc.Text(row.instance_label, size="xs", c="dimmed"),
            dmc.Text(f"Last available {time_label(row.last_available_at)}", size="xs", c="dimmed"),
        ],
    )
    progress = dmc.Progress(
        value=row.availability_percent,
        color=row.provider_color,
        radius="xl",
        size="md",
        style={"background": "#151b22"},
    )
    footer = dmc.Group(
        justify="space-between",
        children=[
            dmc.Text(f"30m availability {row.availability_percent:.0f}%", size="xs", c="dimmed"),
            dmc.Text("Available" if row.available else "Waiting", size="xs", fw=700, c=row.provider_color),
        ],
    )
    return dmc.Paper(
        p="sm",
        radius="lg",
        withBorder=True,
        style={"background": "#11161f", "borderColor": f"{row.provider_color}33"},
        children=dmc.Stack(gap="xs", children=[header, subline, progress, footer]),
    )


def card_component(card: GpuCard):
    accent = lane_color(card.mode)
    figure = dcc.Graph(
        figure=build_history_figure(card),
        config={"displayModeBar": False, "responsive": True},
        style={"height": "220px"},
    )
    stats = dmc.Group(
        justify="space-between",
        children=[
            dmc.Stack(
                gap=0,
                children=[
                    dmc.Text(card.gpu, size="xl", fw=800),
                    dmc.Text(
                        f"{card.available_backends} backend{'s' if card.available_backends != 1 else ''} live",
                        size="sm",
                        c="dimmed",
                    ),
                ],
            ),
            dmc.Stack(
                gap=0,
                ta="right",
                children=[
                    dmc.Text(f"{card.total_available_count} instances", size="lg", fw=800, c=accent),
                    dmc.Text(f"Best now {money_label(card.cheapest_price)}", size="sm", c="dimmed"),
                ],
            ),
        ],
    )
    row_stack = dmc.Stack(
        gap="sm",
        children=[provider_row_component(row) for row in card.rows]
        or [dmc.Text("No provider data for this lane yet.", size="sm", c="dimmed")],
    )
    return dmc.Paper(
        radius="xl",
        p="lg",
        withBorder=True,
        style={
            "background": (
                f"radial-gradient(circle at top left, {accent}1d 0%, rgba(13,16,23,0.96) 42%, rgba(13,16,23,1) 100%)"
            ),
            "borderColor": f"{accent}44",
            "height": "100%",
        },
        children=dmc.Stack(
            gap="md",
            children=[
                stats,
                dmc.Text("30-minute backend availability", size="sm", fw=700, c="dimmed"),
                figure,
                row_stack,
            ],
        ),
    )


def lane_component(lane: LaneSnapshot):
    accent = lane_color(lane.mode)
    cards = [card_component(card) for card in lane.cards]
    body = (
        dmc.SimpleGrid(cols={"base": 1, "xl": 2}, spacing="lg", children=cards)
        if cards
        else dmc.Paper(
            radius="xl",
            p="xl",
            withBorder=True,
            style={"background": "#11161f", "borderColor": f"{accent}44"},
            children=dmc.Text("No GPU data has landed in this lane yet.", size="sm", c="dimmed"),
        )
    )
    return dmc.Stack(
        gap="lg",
        children=[
            dmc.Group(
                justify="space-between",
                children=[
                    dmc.Stack(
                        gap=0,
                        children=[
                            dmc.Text(lane.title, size="xl", fw=900, c=accent, tt="uppercase"),
                            dmc.Text(
                                f"{lane.live_gpu_count} GPU families live across {lane.live_provider_count} providers",
                                size="sm",
                                c="dimmed",
                            ),
                        ],
                    ),
                    dmc.Group(
                        gap="sm",
                        children=[
                            dmc.Badge(f"{lane.live_instance_count} instances", color="gray", variant="light"),
                            dmc.Badge(f"Best {money_label(lane.best_price)}", color="gray", variant="outline"),
                        ],
                    ),
                ],
            ),
            body,
        ],
    )


def render_dashboard(snapshot: DashboardSnapshot, config: DashboardConfig):
    refreshed = snapshot.generated_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    header = dmc.AppShellHeader(
        p="md",
        withBorder=False,
        style={
            "background": "linear-gradient(135deg, #0f1724 0%, #102331 45%, #12291f 100%)",
            "borderBottom": "1px solid rgba(148,163,184,0.16)",
        },
        children=dmc.Stack(
            gap="sm",
            children=[
                dmc.Group(
                    justify="space-between",
                    children=[
                        dmc.Stack(
                            gap=0,
                            children=[
                                dmc.Text("Verda Capacity Board", size="xs", tt="uppercase", fw=800, c="dimmed"),
                                dmc.Title("Availability at a glance", order=1, c="#F8FAFC"),
                                dmc.Text(
                                    "Dense preemptible and on-demand boards with SQL-backed sweep history, live price, and last-seen context.",
                                    size="sm",
                                    c="#C7D2DA",
                                ),
                            ],
                        ),
                        dmc.Stack(
                            gap=0,
                            ta="right",
                            children=[
                                dmc.Text("Refreshed", size="xs", tt="uppercase", fw=800, c="dimmed"),
                                dmc.Text(refreshed, size="sm", fw=700),
                                dmc.Text(f"Polling every {int(config.poll_seconds)}s on :{config.dashboard_port}", size="xs", c="dimmed"),
                            ],
                        ),
                    ],
                ),
                dmc.Group(gap="xs", children=sweep_badges(snapshot.sweep)),
                dmc.Group(gap="sm", children=header_kpis(snapshot)),
                dmc.Group(gap="xs", children=source_badges(snapshot)),
            ],
        ),
    )
    main = dmc.AppShellMain(
        p="lg",
        children=dmc.Stack(
            gap="xl",
            children=[
                lane_component(snapshot.preemptible),
                lane_component(snapshot.on_demand),
            ],
        ),
    )
    return dmc.AppShell(
        padding="lg",
        header={"height": 164},
        children=[header, main],
        style={"background": "#0B0F15", "minHeight": "100vh"},
    )


def build_app() -> Dash:
    config = load_dashboard_config()
    initial_snapshot = build_dashboard_snapshot(config)
    app = Dash(__name__, external_stylesheets=dmc.styles.ALL)
    app.title = "Verda Availability Dashboard"
    app.layout = dmc.MantineProvider(
        theme=shell_theme(),
        forceColorScheme="dark",
        children=[
            dcc.Interval(
                id="dashboard-tick",
                interval=config.poll_interval_ms,
                n_intervals=0,
            ),
            html.Div(id="dashboard-shell", children=render_dashboard(initial_snapshot, config)),
        ],
    )

    @app.callback(DashOut("dashboard-shell", "children"), DashIn("dashboard-tick", "n_intervals"))
    def refresh_dashboard(_: int):
        return render_dashboard(build_dashboard_snapshot(config), config)

    return app


app = build_app()


def main() -> None:
    config = load_dashboard_config()
    start_sweep_scheduler(config)
    app.run(host="0.0.0.0", port=config.dashboard_port, debug=False)


if __name__ == "__main__":
    main()
