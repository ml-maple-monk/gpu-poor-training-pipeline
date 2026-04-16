"""config.py — Central configuration loaded from defaults.toml.

Two-layer merge:
  1. Load [dashboard] from repo-root defaults.toml
  2. If /app/config-overlay.toml exists (container), deep-merge over defaults

Secrets (tokens) are the ONLY values still read from environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

_OVERLAY_PATH = Path("/app/config-overlay.toml")


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    """Recursively merge *overlay* into *base* in place."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _load_dashboard_config() -> dict[str, Any]:
    """Load [dashboard] from defaults.toml, optionally merged with overlay."""
    # 4 parents: config.py -> src -> dashboard -> infrastructure -> repo-root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    defaults_path = repo_root / "defaults.toml"
    if not defaults_path.is_file():
        defaults_path = Path("/app/defaults.toml")  # container fallback
    with open(defaults_path, "rb") as f:
        base = tomllib.load(f).get("dashboard", {})
    # Overlay (container deployment): deep-merge user TOML over defaults
    if _OVERLAY_PATH.is_file():
        with open(_OVERLAY_PATH, "rb") as f:
            overlay = tomllib.load(f).get("dashboard", {})
        _deep_merge(base, overlay)
    return base


_DEFAULTS = _load_dashboard_config()

# ── Service URLs ─────────────────────────────────────────────────────────────
DSTACK_SERVER_URL: str = _DEFAULTS["dstack_server_url"]
MLFLOW_URL: str = _DEFAULTS["mlflow_url"]

# ── Secrets (env-var only — never in TOML) ───────────────────────────────────
DSTACK_SERVER_ADMIN_TOKEN: str = os.environ.get("DSTACK_SERVER_ADMIN_TOKEN", "")

# ── Docker ───────────────────────────────────────────────────────────────────
TRAINER_CONTAINER: str = _DEFAULTS["trainer_container"]

# ── File paths ───────────────────────────────────────────────────────────────
CF_TUNNEL_URL_FILE: str = _DEFAULTS["cf_tunnel_url_file"]
SEEKER_DATA_DIR: str = _DEFAULTS["seeker_data_dir"]

# ── Gradio ───────────────────────────────────────────────────────────────────
GRADIO_PORT: int = _DEFAULTS["gradio_port"]
GRADIO_QUEUE_MAX_SIZE: int = _DEFAULTS["gradio_queue_max_size"]
LOG_RING_SIZE: int = _DEFAULTS["log_ring_size"]

# ── Collector cadences (seconds) ─────────────────────────────────────────────
COLLECTOR_CADENCE_TRAINING: float = _DEFAULTS["collector_cadence_training"]
COLLECTOR_CADENCE_LIVE_METRICS: float = _DEFAULTS["collector_cadence_live_metrics"]
COLLECTOR_CADENCE_DSTACK: float = _DEFAULTS["collector_cadence_dstack"]
COLLECTOR_CADENCE_SYSTEM: float = _DEFAULTS["collector_cadence_system"]
COLLECTOR_CADENCE_MLFLOW: float = _DEFAULTS["collector_cadence_mlflow"]
COLLECTOR_CADENCE_TUNNEL: float = _DEFAULTS["collector_cadence_tunnel"]
COLLECTOR_CADENCE_SEEKER: float = _DEFAULTS["collector_cadence_seeker"]
COLLECTOR_CADENCE_OFFERS: float = _DEFAULTS["collector_cadence_offers"]

# ── UI timer intervals (seconds) ─────────────────────────────────────────────
TIMER_FAST: float = _DEFAULTS["timer_fast"]
TIMER_MEDIUM: float = _DEFAULTS["timer_medium"]
TIMER_SLOW: float = _DEFAULTS["timer_slow"]
TIMER_SEEKER: float = _DEFAULTS["timer_seeker"]
TIMER_DETAIL: float = _DEFAULTS["timer_detail"]
TIMER_LOG: float = _DEFAULTS["timer_log"]
TIMER_DSTACK_LOG: float = _DEFAULTS["timer_dstack_log"]

# ── Shutdown ─────────────────────────────────────────────────────────────────
SHUTDOWN_GRACE_SECONDS: float = _DEFAULTS["shutdown_grace_seconds"]
MIN_WORKER_JOIN_TIMEOUT: float = _DEFAULTS["min_worker_join_timeout"]

# ── HTTP timeouts (seconds) ──────────────────────────────────────────────────
TIMEOUT_DSTACK_REST: float = _DEFAULTS["timeout_dstack_rest"]
TIMEOUT_DSTACK_STREAM: float = _DEFAULTS["timeout_dstack_stream"]
TIMEOUT_DSTACK_OFFERS: float = _DEFAULTS["timeout_dstack_offers"]
TIMEOUT_MLFLOW: int = _DEFAULTS["timeout_mlflow"]
TIMEOUT_BOOTSTRAP_REST: float = _DEFAULTS["timeout_bootstrap_rest"]
TIMEOUT_BOOTSTRAP_CLI: float = _DEFAULTS["timeout_bootstrap_cli"]

# ── API limits ───────────────────────────────────────────────────────────────
DSTACK_RUNS_LIMIT: int = _DEFAULTS["dstack_runs_limit"]
MLFLOW_MAX_RECENT_RUNS: int = _DEFAULTS["mlflow_max_recent_runs"]
MLFLOW_MAX_METRIC_HISTORY: int = _DEFAULTS["mlflow_max_metric_history"]
MLFLOW_MAX_METRICS_PER_RUN: int = _DEFAULTS["mlflow_max_metrics_per_run"]
MAX_OFFERS_PER_GPU: int = _DEFAULTS["max_offers_per_gpu"]
OFFER_HISTORY_MAXLEN: int = _DEFAULTS["offer_history_maxlen"]

# ── GPU specs ────────────────────────────────────────────────────────────────
# List of (probe_name, display_name, min_memory_mib | None) tuples
GPU_SPECS: list[tuple[str, str, int | None]] = [
    (
        spec["probe_name"],
        spec["display_name"],
        spec["min_memory_mib"] if spec["min_memory_mib"] != 0 else None,
    )
    for spec in _DEFAULTS.get("gpu_specs", [])
]

# ── Platform colors ──────────────────────────────────────────────────────────
# Dict mapping lowercase backend name -> (primary, muted, label)
_pc = _DEFAULTS.get("platform_colors", {})
PLATFORM_COLORS: dict[str, tuple[str, str, str]] = {
    k: (v["primary"], v["muted"], v["label"])
    for k, v in _pc.items()
    if k != "_default"
}
_dc = _pc.get("_default", {"primary": "#8b949e", "muted": "#21262d", "label": "Other"})
DEFAULT_PLATFORM_COLOR: tuple[str, str, str] = (_dc["primary"], _dc["muted"], _dc["label"])

# ── GPU display colors ───────────────────────────────────────────────────────
# Dict mapping display_name -> hex color
GPU_COLORS: dict[str, str] = dict(_DEFAULTS.get("gpu_colors", {}))
