"""config.py — Central configuration from environment variables."""

from __future__ import annotations

import os

# dstack
DSTACK_SERVER_URL: str = os.environ.get("DSTACK_SERVER_URL", "http://localhost:3000")
DSTACK_SERVER_ADMIN_TOKEN: str = os.environ.get("DSTACK_SERVER_ADMIN_TOKEN", "")

# MLflow
MLFLOW_URL: str = os.environ.get("MLFLOW_URL", "http://localhost:5000")

# Docker
TRAINER_CONTAINER: str = os.environ.get("TRAINER_CONTAINER", "minimind-trainer")

# CF tunnel file path (mounted at /tunnel in container)
CF_TUNNEL_URL_FILE: str = os.environ.get("CF_TUNNEL_URL_FILE", "/tunnel/.cf-tunnel.url")

# Log ring size
LOG_RING_SIZE: int = int(os.environ.get("LOG_RING_SIZE", "500"))

# Gradio
GRADIO_PORT: int = int(os.environ.get("GRADIO_PORT", "7860"))
GRADIO_QUEUE_MAX_SIZE: int = int(os.environ.get("GRADIO_QUEUE_MAX_SIZE", "5"))
