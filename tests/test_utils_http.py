"""Tests for gpupoor.utils.http."""

from __future__ import annotations

import urllib.error
from typing import Any
from unittest.mock import patch

import pytest

from gpupoor.utils import http as http_utils


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None


def test_http_ok_true_on_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """200 OK is the one status that counts as healthy."""
    monkeypatch.setattr(http_utils.urllib.request, "urlopen", lambda *a, **kw: _FakeResponse(200))
    assert http_utils.http_ok("http://example/") is True


def test_http_ok_false_on_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any URL-layer failure is swallowed into a False return."""

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise urllib.error.URLError("nope")

    monkeypatch.setattr(http_utils.urllib.request, "urlopen", boom)
    assert http_utils.http_ok("http://example/") is False


def test_wait_for_health_returns_true_once_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The poll loop must stop as soon as the endpoint replies with 200."""
    responses = [urllib.error.URLError("warming up"), _FakeResponse(200)]

    def urlopen(*_args: Any, **_kwargs: Any) -> Any:
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(http_utils.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(http_utils.time, "sleep", lambda _s: None)

    assert (
        http_utils.wait_for_health(
            "http://example/",
            total_timeout_seconds=5,
            per_check_timeout_seconds=1,
            sleep_seconds=0,
        )
        is True
    )


def test_wait_for_health_returns_false_on_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the deadline passes without success, the helper must return False."""

    def never_healthy(*_args: Any, **_kwargs: Any) -> Any:
        raise urllib.error.URLError("down")

    # Force the deadline to be hit on the first check: time.monotonic is
    # called once to set the deadline and again to compare against it.
    clock = iter([0.0, 10.0, 20.0])
    monkeypatch.setattr(http_utils.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(http_utils.urllib.request, "urlopen", never_healthy)
    monkeypatch.setattr(http_utils.time, "sleep", lambda _s: None)

    assert (
        http_utils.wait_for_health(
            "http://example/",
            total_timeout_seconds=1,
            per_check_timeout_seconds=1,
            sleep_seconds=0,
        )
        is False
    )


def test_wait_for_health_accepts_expected_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 503 response must count as success when ``expected_status=503``."""

    def raise_503(*_args: Any, **_kwargs: Any) -> Any:
        raise urllib.error.HTTPError("http://example/", 503, "Service Unavailable", hdrs=None, fp=None)

    monkeypatch.setattr(http_utils.urllib.request, "urlopen", raise_503)
    monkeypatch.setattr(http_utils.time, "sleep", lambda _s: None)

    with patch.object(http_utils.time, "monotonic", side_effect=[0.0, 0.5, 0.9]):
        assert (
            http_utils.wait_for_health(
                "http://example/",
                total_timeout_seconds=10,
                per_check_timeout_seconds=1,
                sleep_seconds=0,
                expected_status=503,
            )
            is True
        )
