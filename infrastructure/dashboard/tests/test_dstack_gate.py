"""test_dstack_gate.py — Step 3.5 boot gate: access path selection."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bootstrap import _probe_dstack_cli, _probe_rest, choose_access_path


@pytest.mark.parametrize("response_shape", [{"runs": []}, []], ids=["dict-runs", "bare-list"])
def test_choose_access_path_prefers_rest_when_a_token_exists(monkeypatch, response_shape):
    """Proves the dashboard treats both accepted REST payload shapes as a healthy
    C2.2 path, so read-only dstack access is available before any CLI fallback."""
    monkeypatch.setenv("DSTACK_SERVER_ADMIN_TOKEN", "token123")

    mock_resp = MagicMock()
    mock_resp.json.return_value = response_shape

    with patch("src.bootstrap.safe_dstack_rest", return_value=mock_resp) as mock_rest:
        path = choose_access_path()

    mock_rest.assert_called_once_with(
        "runs/list",
        method="POST",
        json={"limit": 50},
        timeout=5.0,
    )
    assert path == "C2.2"


@pytest.mark.parametrize(
    ("token", "rest_side_effect", "cli_works", "expected"),
    [
        ("token123", httpx.ConnectError("connection refused"), True, "C2.1"),
        ("token123", httpx.ConnectError("refused"), False, "FAILED"),
        ("", None, False, "FAILED"),
    ],
    ids=["rest-fails-cli-recovers", "all-paths-fail", "no-token-skips-rest"],
)
def test_choose_access_path_falls_back_or_fails_closed(monkeypatch, token, rest_side_effect, cli_works, expected):
    """Proves the boot gate only degrades along the intended path order and
    fails closed when neither REST nor CLI access is usable."""
    monkeypatch.setenv("DSTACK_SERVER_ADMIN_TOKEN", token)

    rest_patch = patch("src.bootstrap.safe_dstack_rest")
    with rest_patch as mock_rest:
        if rest_side_effect is not None:
            mock_rest.side_effect = rest_side_effect
        with patch("src.bootstrap._probe_dstack_cli", return_value=cli_works):
            path = choose_access_path()

    if token:
        assert mock_rest.called
    else:
        mock_rest.assert_not_called()
    assert path == expected


@pytest.mark.parametrize(
    "payload",
    ["just a string", 42, None, 3.14],
    ids=["string", "int", "null", "float"],
)
def test_probe_rest_handles_non_dict_non_list_payload(payload):
    """Proves a scalar/None JSON body does not raise AttributeError from
    data.get(...); _probe_rest must treat it as a BAD_SHAPE and return False."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload

    with patch("src.bootstrap.safe_dstack_rest", return_value=mock_resp):
        assert _probe_rest() is False


def test_probe_rest_handles_missing_runs_key():
    """Proves a dict payload without the 'runs' key is rejected as BAD_SHAPE
    without raising — the narrow except must cover the KeyError/TypeError path."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"other": []}

    with patch("src.bootstrap.safe_dstack_rest", return_value=mock_resp):
        assert _probe_rest() is False


@pytest.mark.parametrize(
    "exc",
    [
        PermissionError("dstack binary not executable"),
        FileNotFoundError("dstack not on PATH"),
        __import__("subprocess").TimeoutExpired(cmd="dstack ps", timeout=10),
    ],
    ids=["permission-denied", "binary-missing", "probe-timeout"],
)
def test_probe_dstack_cli_handles_concrete_exceptions(exc):
    """Proves the narrowed except tuple covers the realistic failure
    set (OSError + subprocess.SubprocessError) without re-raising —
    CS5 replaced the bare `except Exception` with this tighter set.

    ``subprocess`` is imported inside the function under test, so we
    patch ``subprocess.run`` at the stdlib level rather than on the
    ``src.bootstrap`` module namespace.
    """
    with patch("subprocess.run", side_effect=exc):
        assert _probe_dstack_cli() is False
