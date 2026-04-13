"""Tests for :mod:`gpupoor.utils.logging`."""

from __future__ import annotations

import logging

import pytest

from gpupoor.utils import logging as gp_logging


@pytest.fixture(autouse=True)
def _reset_logging_state() -> None:
    """Reset the module-level configured flag and handlers between tests.

    ``configure_root`` is intentionally idempotent, so tests that want to
    exercise fresh configuration must reset the singleton state first.
    """
    gp_logging._CONFIGURED = False  # noqa: SLF001 — test-only reset
    root = logging.getLogger("gpupoor")
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.propagate = True
    yield
    gp_logging._CONFIGURED = False  # noqa: SLF001
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.propagate = True


def test_configure_root_is_idempotent() -> None:
    gp_logging.configure_root()
    gp_logging.configure_root()
    gp_logging.configure_root()

    root = logging.getLogger("gpupoor")
    # Exactly one stdout + one stderr handler, regardless of call count.
    assert len(root.handlers) == 2


def test_info_goes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    gp_logging.configure_root()
    log = gp_logging.get_logger("gpupoor.demo")

    log.info("hello world")

    captured = capsys.readouterr()
    assert "[gpupoor] hello world" in captured.out
    assert captured.err == ""


def test_warning_goes_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    gp_logging.configure_root()
    log = gp_logging.get_logger("gpupoor.demo")

    log.warning("watch out")

    captured = capsys.readouterr()
    assert "[gpupoor] watch out" in captured.err
    assert captured.out == ""


def test_prefix_applied_once(capsys: pytest.CaptureFixture[str]) -> None:
    gp_logging.configure_root()
    log = gp_logging.get_logger("gpupoor.demo")

    log.info("only-once")

    captured = capsys.readouterr()
    # Prefix must appear exactly once per record — the formatter owns it,
    # so call sites that forget to strip their old literal prefix would
    # show up as "[gpupoor] [gpupoor] …" here.
    assert captured.out.count("[gpupoor]") == 1


def test_get_logger_namespace() -> None:
    # Dotted module path should be re-rooted under gpupoor.
    logger = gp_logging.get_logger("gpupoor.backends.dstack")
    assert logger.name == "gpupoor.backends.dstack"

    # Anything else gets rewritten to gpupoor.<suffix>.
    rewired = gp_logging.get_logger("some.other.module")
    assert rewired.name == "gpupoor.module"

    # Bare name (no dots) still lands under gpupoor.
    bare = gp_logging.get_logger("widget")
    assert bare.name == "gpupoor.widget"
