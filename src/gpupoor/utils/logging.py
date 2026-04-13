"""Shared logging helper.

Configured once in ``gpupoor.cli.main()``; modules should grab a logger
via :func:`get_logger` and emit plain messages — the formatter adds the
``[gpupoor]`` prefix, and the handler configuration routes INFO/DEBUG to
stdout and WARNING+ to stderr (matching the prior print-based behavior).
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_PREFIX = "[gpupoor]"
_ROOT_NAME = "gpupoor"


def _below_warning(record: logging.LogRecord) -> bool:
    return record.levelno < logging.WARNING


def configure_root(*, level: int = logging.INFO) -> None:
    """Configure the ``gpupoor`` logger once. Idempotent.

    INFO and DEBUG are routed to stdout; WARNING and above to stderr.
    A single ``[gpupoor]`` prefix is added by the formatter so call
    sites stay tidy (``log.info("message")``).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(_below_warning)
    stdout_handler.setFormatter(logging.Formatter(f"{_PREFIX} %(message)s"))

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter(f"{_PREFIX} %(message)s"))

    root.addHandler(stdout_handler)
    root.addHandler(stderr_handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger under the ``gpupoor`` namespace.

    Accepts either a dotted module name (``__name__``) or a bare suffix.
    Names already rooted at ``gpupoor`` are returned as-is; anything else
    is re-rooted so handlers configured on the ``gpupoor`` logger apply.
    """
    if name == _ROOT_NAME or name.startswith(f"{_ROOT_NAME}."):
        return logging.getLogger(name)
    suffix = name.rsplit(".", 1)[-1]
    return logging.getLogger(f"{_ROOT_NAME}.{suffix}")
