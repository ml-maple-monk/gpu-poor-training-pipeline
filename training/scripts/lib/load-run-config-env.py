#!/usr/bin/env python3
"""Emit shell exports for a generated gpupoor runtime config or a live TOML."""

from __future__ import annotations

import json
import shlex
import sys
import tomllib
from pathlib import Path

try:
    from gpupoor.config import runtime_env_from_tables
except ModuleNotFoundError:  # pragma: no cover - host/container path fallback
    SCRIPT_PATH = Path(__file__)
    candidate = SCRIPT_PATH.parent.parent.parent.parent / "src"
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
    from gpupoor.config import runtime_env_from_tables


def _env_from_toml(path: Path) -> dict[str, str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return runtime_env_from_tables(
        recipe=data.get("recipe", {}),
        training=data.get("training", {}),
        mlflow=data.get("mlflow", {}),
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: load-run-config-env.py <run-config.json>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if path.suffix == ".toml":
        env = _env_from_toml(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        env = payload.get("env")
        if not isinstance(env, dict):
            print("run config missing 'env' object", file=sys.stderr)
            return 1
    for key in sorted(env):
        value = env[key]
        if value is None:
            continue
        print(f"export {key}={shlex.quote(str(value))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
