#!/usr/bin/env bash
# doc-anchor-check.sh — verify every anchor referenced in READMEs resolves to a
# source-file comment of the form:  # doc-anchor: <name>
#
# Exit 0: all referenced anchors are defined in source.
# Exit 1: one or more referenced anchors have no matching source comment.
#
# Usage:
#   bash scripts/doc-anchor-check.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Collect all anchor names defined in source files
defined=$(grep -rh 'doc-anchor:' \
        infrastructure/ training/ dstack/ scripts/ \
    2>/dev/null \
    | grep -oP 'doc-anchor:\s*\K[\w-]+' \
    | sort -u)

# Collect all anchor names referenced in READMEs / TROUBLESHOOTING
referenced=$(grep -rh 'doc-anchor:' \
        README.md TROUBLESHOOTING.md \
        training/docs/README.md infrastructure/dashboard/docs/README.md infrastructure/local-emulator/docs/README.md infrastructure/mlflow/docs/README.md dstack/docs/README.md \
    2>/dev/null \
    | grep -oP 'doc-anchor:\s*\K[\w-]+' \
    | sort -u)

if [ -z "$referenced" ]; then
    echo "[anchor-check] No anchor references found in READMEs — nothing to verify."
    exit 0
fi

missing=$(comm -23 <(echo "$referenced") <(echo "$defined"))

if [ -n "$missing" ]; then
    echo "[anchor-check] FAIL — unresolved anchors referenced in READMEs but not defined in source:"
    echo "$missing" | sed 's/^/  - /'
    exit 1
fi

echo "[anchor-check] OK — all $(echo "$referenced" | wc -l | tr -d ' ') referenced anchors resolve to source comments."
