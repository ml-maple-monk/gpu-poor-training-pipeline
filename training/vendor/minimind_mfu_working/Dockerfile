FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/app/.venv/bin:/root/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    python3-pip \
    python3.12 \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml uv.lock README.md ARCHITECTURE.md ./
COPY minimind_mfu_experiment.py ./
COPY minimind_local ./minimind_local
COPY recipes ./recipes

RUN uv sync --frozen --no-dev

CMD ["minimind-train", "--help"]
