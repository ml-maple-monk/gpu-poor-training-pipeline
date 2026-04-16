"""config.py — Central configuration from environment variables."""

from __future__ import annotations

import os

DEFAULT_DSTACK_SERVER_URL = "http://localhost:3000"
DEFAULT_MLFLOW_URL = "http://localhost:5000"
DEFAULT_TRAINER_CONTAINER = "minimind-trainer"
DEFAULT_CF_TUNNEL_URL_FILE = "/tunnel/.cf-tunnel.url"
DEFAULT_SEEKER_DATA_DIR = "/seeker-data"
DEFAULT_LOG_RING_SIZE = 500
DEFAULT_GRADIO_PORT = 7860
DEFAULT_GRADIO_QUEUE_MAX_SIZE = 5

# dstack
DSTACK_SERVER_URL: str = os.environ.get("DSTACK_SERVER_URL", DEFAULT_DSTACK_SERVER_URL)
DSTACK_SERVER_ADMIN_TOKEN: str = os.environ.get("DSTACK_SERVER_ADMIN_TOKEN", "")

# MLflow
MLFLOW_URL: str = os.environ.get("MLFLOW_URL", DEFAULT_MLFLOW_URL)

# Docker
TRAINER_CONTAINER: str = os.environ.get("TRAINER_CONTAINER", DEFAULT_TRAINER_CONTAINER)

# CF tunnel file path (mounted at /tunnel in container)
CF_TUNNEL_URL_FILE: str = os.environ.get("CF_TUNNEL_URL_FILE", DEFAULT_CF_TUNNEL_URL_FILE)

# Seeker data directory (mounted read-only into the dashboard container)
SEEKER_DATA_DIR: str = os.environ.get("SEEKER_DATA_DIR", DEFAULT_SEEKER_DATA_DIR)

# Log ring size
LOG_RING_SIZE: int = int(os.environ.get("LOG_RING_SIZE", str(DEFAULT_LOG_RING_SIZE)))

# Gradio
GRADIO_PORT: int = int(os.environ.get("GRADIO_PORT", str(DEFAULT_GRADIO_PORT)))
GRADIO_QUEUE_MAX_SIZE: int = int(os.environ.get("GRADIO_QUEUE_MAX_SIZE", str(DEFAULT_GRADIO_QUEUE_MAX_SIZE)))
