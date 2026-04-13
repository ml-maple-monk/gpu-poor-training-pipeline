#!/usr/bin/env bash

# Shared dstack CLI resolver.
#
# Prefer the isolated uv-managed CLI when present, because host Python
# environments can break dstack imports via incompatible dependency resolution.

resolve_dstack_bin() {
    local -a candidates=()
    local candidate=""

    if [ -n "${DSTACK_BIN:-}" ]; then
        candidates+=("${DSTACK_BIN}")
    fi

    candidates+=("$HOME/.dstack-cli-venv/bin/dstack")

    if candidate=$(command -v dstack 2>/dev/null); then
        candidates+=("$candidate")
    fi

    for candidate in "${candidates[@]}"; do
        [ -n "$candidate" ] || continue
        [ -x "$candidate" ] || continue
        if "$candidate" --version >/dev/null 2>&1; then
            DSTACK_BIN="$candidate"
            export DSTACK_BIN
            return 0
        fi
    done

    return 1
}

require_dstack_bin() {
    if resolve_dstack_bin; then
        return 0
    fi

    echo "[dstack-cli] ERROR: no working dstack CLI found." >&2
    echo "[dstack-cli] Prefer: uv venv ~/.dstack-cli-venv --python 3.11" >&2
    echo "[dstack-cli] Then:   uv pip install --python ~/.dstack-cli-venv/bin/python 'dstack[verda]==0.20.*'" >&2
    return 1
}

dstack_cli() {
    require_dstack_bin || return 1
    "$DSTACK_BIN" "$@"
}
