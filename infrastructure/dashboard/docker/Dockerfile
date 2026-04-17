# Verda Dashboard — read-only Dash container
# Shared image for the dashboard app and the dstack sidecar.
FROM python:3.11-slim

# ── Privacy/telemetry baked in ───────────────────────────────────────────────
ENV HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System deps (minimal) ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ────────────────────────────────────────────────────────────
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app
RUN mkdir -p /app && chown -R app:app /app

# ── Install Python deps ──────────────────────────────────────────────────────
WORKDIR /app
COPY config/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source + entrypoint ────────────────────────────────────────────────
COPY src/ ./src/
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ── Run as non-root ──────────────────────────────────────────────────────────
USER app

EXPOSE 7860

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "src.app"]
