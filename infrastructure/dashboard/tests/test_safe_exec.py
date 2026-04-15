"""test_safe_exec.py — F2 regression tests for safe_dstack_stream lifecycle.

Covers the client-leak bug in safe_exec.py:
  - safe_dstack_stream previously created an httpx.Client and returned only
    the stream context manager, so the caller's `with` exit closed the stream
    but left the client open.
  - After F2, safe_dstack_stream is itself a context manager that closes BOTH
    the stream and the underlying httpx.Client on exit (normal or exceptional).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src import safe_exec


@pytest.fixture
def fake_httpx_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the module-level httpx.Client with a MagicMock factory.

    Returns the MagicMock *instance* that `httpx.Client(...)` will yield, so
    tests can assert `.close()` / `.stream()` behaviour directly. The stream
    context manager returned by `client.stream(...)` yields a MagicMock
    response when entered.
    """
    client_instance = MagicMock(name="httpx_client")
    # client.stream(...) returns a context manager yielding a fake response.
    stream_cm = MagicMock(name="httpx_stream_cm")
    stream_cm.__enter__.return_value = MagicMock(name="httpx_response")
    stream_cm.__exit__.return_value = False
    client_instance.stream.return_value = stream_cm

    client_factory = MagicMock(name="httpx_Client_factory", return_value=client_instance)
    monkeypatch.setattr(safe_exec.httpx, "Client", client_factory)
    # Also stub _get_token / endpoint resolution so the test does not need env.
    monkeypatch.setattr(safe_exec, "_get_token", lambda: "test-token")
    monkeypatch.setattr(
        safe_exec,
        "_assert_endpoint",
        lambda endpoint: f"http://test.local/api/{endpoint}",
    )
    return client_instance


def test_safe_dstack_stream_closes_client_on_normal_exit(fake_httpx_client: MagicMock) -> None:
    """On a clean `with` exit, the httpx.Client must be closed (not leaked)."""
    with safe_exec.safe_dstack_stream("runs/get_logs", json={"foo": "bar"}) as resp:
        # resp is the MagicMock response; just confirm we got something usable.
        assert resp is not None

    assert fake_httpx_client.close.called, (
        "httpx.Client.close() must be called when the caller's `with` exits normally; "
        f"close.call_args_list={fake_httpx_client.close.call_args_list}"
    )


def test_safe_dstack_stream_closes_client_on_exception(fake_httpx_client: MagicMock) -> None:
    """If the caller's body raises, the httpx.Client must still be closed."""

    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom), safe_exec.safe_dstack_stream("runs/get_logs", json={"foo": "bar"}) as resp:
        assert resp is not None
        raise Boom("caller body failed mid-stream")

    assert fake_httpx_client.close.called, (
        "httpx.Client.close() must be called even when the caller's body raises; "
        f"close.call_args_list={fake_httpx_client.close.call_args_list}"
    )
