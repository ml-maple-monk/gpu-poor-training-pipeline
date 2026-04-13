# Verda Dashboard — read-only Gradio container
# Target: <300 MB, non-root app user, no dstack CLI, no docker SDK
FROM python:3.11-slim

# ── Privacy/telemetry baked in ───────────────────────────────────────────────
ENV GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/mpl \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System deps (minimal) ────────────────────────────────────────────────────
# Note: docker CLI binary is mounted from host at /usr/bin/docker (read-only)
# so docker.io is NOT installed in the image (saves ~200MB and avoids split package issue)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ────────────────────────────────────────────────────────────
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app
RUN mkdir -p /tunnel /artifacts-pull && chown -R app:app /tunnel /artifacts-pull

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
