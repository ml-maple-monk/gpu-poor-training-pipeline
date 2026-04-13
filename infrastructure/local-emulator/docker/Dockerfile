ARG BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04

# --- builder stage ---
FROM ${BASE_IMAGE} AS builder

ARG APP_PORT=8000
ARG APP_USER=verda
ARG APP_UID=1000
ARG APP_GID=1000

# reject :latest tag at build time
RUN echo "$BASE_IMAGE" | grep -vq ':latest$'

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY config/requirements.txt .
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir -r requirements.txt

# --- runtime stage ---
FROM ${BASE_IMAGE} AS runtime

ARG APP_PORT=8000
ARG APP_USER=verda
ARG APP_UID=1000
ARG APP_GID=1000

# reject :latest in runtime stage too (ARG is per-stage)
RUN echo "$BASE_IMAGE" | grep -vq ':latest$'

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

RUN groupadd -g ${APP_GID} ${APP_USER} && \
    useradd -u ${APP_UID} -g ${APP_GID} -m -s /bin/sh ${APP_USER}

WORKDIR /app
COPY src/ /app/src/
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# /data owned by verda so the non-root user can write
RUN mkdir -p /data && chown ${APP_UID}:${APP_GID} /data

EXPOSE ${APP_PORT}

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${APP_PORT:-8000}/health || exit 1

USER ${APP_USER}

ENTRYPOINT ["/app/entrypoint.sh"]
