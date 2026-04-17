"""Helpers for serializing fully-merged RunConfig as TOML."""

from __future__ import annotations

import base64
from dataclasses import asdict, fields
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

from gpupoor.config import RunConfig


def write_merged_toml(config: RunConfig, path: str | Path) -> None:
    """Write a fully-merged RunConfig as a TOML file.

    The output contains all values (defaults + user overrides) so the
    container-side code can read it with a simple tomllib.load() -- no
    defaults lookup needed.
    """
    data = _config_to_dict(config)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def merged_toml_b64(config: RunConfig) -> str:
    """Serialize a RunConfig as base64-encoded TOML for remote passing."""
    data = _config_to_dict(config)
    toml_bytes = tomli_w.dumps(data).encode("utf-8")
    return base64.b64encode(toml_bytes).decode("ascii")


def _sanitize_value(value: object) -> object:
    """Make a value safe for TOML serialization.

    - Convert tuples to lists (TOML only has arrays).
    - Convert Path objects to strings.
    - Recursively sanitize dicts and lists.
    - Return None unchanged (caller is responsible for omitting it).
    """
    if isinstance(value, tuple):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return _strip_nones({k: _sanitize_value(v) for k, v in value.items()})
    return value


def _strip_nones(d: dict) -> dict:
    """Remove keys whose values are None (TOML has no null type)."""
    return {k: v for k, v in d.items() if v is not None}


def _config_to_dict(config: RunConfig) -> dict:
    """Convert a RunConfig to a plain dict suitable for TOML serialization.

    Skips the ``source`` field (it holds the file path, not config data).
    Omits None values and converts tuples to lists for TOML compatibility.
    """
    result: dict = {"name": config.name}

    # Each section corresponds to a sub-dataclass on RunConfig.
    # We skip 'source' (Path to the config file) and 'name' (already added).
    _SKIP_FIELDS = {"name", "source"}

    for field in fields(config):
        if field.name in _SKIP_FIELDS:
            continue
        sub = getattr(config, field.name)
        raw = asdict(sub)
        sanitized = _sanitize_value(raw)
        result[field.name] = sanitized

    # Include gpu_profiles (array-of-tables from defaults.toml) so the
    # container can match its GPU against known peak TFLOPs values.
    from gpupoor.config import _DEFAULTS
    gpu_profiles = _DEFAULTS.get("gpu_profiles")
    if gpu_profiles:
        result["gpu_profiles"] = _sanitize_value(gpu_profiles)

    return result
