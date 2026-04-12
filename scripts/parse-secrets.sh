#!/usr/bin/env bash
set -euo pipefail

SECRETS_FILE="${1:-./secrets}"

if [ ! -f "$SECRETS_FILE" ]; then
    echo "ERROR: secrets file not found: $SECRETS_FILE" >&2
    exit 1
fi

# parse Secret value (client secret)
SECRET=$(grep -m1 'Secret:' "$SECRETS_FILE" | sed 's/^[Ss]ecret:\s*//' | tr -d '[:space:]') || true
# parse ClientID — tolerate "CliendID" typo
CLIENT_ID=$(grep -m1 -E '[Cc]lien[dt]\s*ID\s*:' "$SECRETS_FILE" | sed 's/^[Cc]lien[dt]\s*ID\s*:\s*//' | tr -d '[:space:]') || true

if [ -z "$SECRET" ]; then
    echo "ERROR: could not parse Secret from $SECRETS_FILE" >&2
    exit 1
fi
if [ -z "$CLIENT_ID" ]; then
    echo "ERROR: could not parse ClientID/CliendID from $SECRETS_FILE" >&2
    exit 1
fi

# generate a random local-dev inference token (not a Verda-issued token)
INFERENCE_TOKEN="local-dev-$(head -c16 /dev/urandom | base64 | tr -d '=/+')"

cat > .env.inference <<EOF
# LOCAL-ONLY bearer token for /infer — NOT a Verda-issued token.
# Delete this file and re-run parse-secrets.sh to rotate.
VERDA_INFERENCE_TOKEN=${INFERENCE_TOKEN}
EOF
chmod 600 .env.inference

cat > .env.mgmt <<EOF
VERDA_CLIENT_ID=${CLIENT_ID}
VERDA_CLIENT_SECRET=${SECRET}
EOF
chmod 600 .env.mgmt

echo "Written: .env.inference (mode 600)"
echo "Written: .env.mgmt (mode 600)"
