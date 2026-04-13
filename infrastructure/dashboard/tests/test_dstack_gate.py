"""test_dstack_gate.py — Step 3.5 boot gate: access path selection."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bootstrap import choose_access_path


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
        ("token123", Exception("connection refused"), True, "C2.1a"),
        ("token123", Exception("refused"), False, "FAILED"),
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
