"""test_dstack_gate.py — Step 3.5 boot gate: access path selection."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bootstrap import choose_access_path


def test_c22_chosen_when_rest_works(monkeypatch):
    """C2.2 is chosen when REST probe succeeds."""
    monkeypatch.setenv("DSTACK_SERVER_ADMIN_TOKEN", "token123")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"runs": []}

    with patch("src.bootstrap.safe_dstack_rest", return_value=mock_resp) as mock_rest:
        path = choose_access_path()

    mock_rest.assert_called_once_with(
        "runs/list",
        method="POST",
        json={"limit": 50},
        timeout=5.0,
    )
    assert path == "C2.2"


def test_c21a_chosen_when_rest_fails_cli_works(monkeypatch):
    """C2.1a is chosen when REST fails but CLI works."""
    monkeypatch.setenv("DSTACK_SERVER_ADMIN_TOKEN", "token123")

    with patch("src.bootstrap.safe_dstack_rest", side_effect=Exception("connection refused")):
        with patch("src.bootstrap._probe_dstack_cli", return_value=True):
            path = choose_access_path()

    assert path == "C2.1a"


def test_failed_when_all_paths_fail(monkeypatch):
    """FAILED is returned when all access paths fail."""
    monkeypatch.setenv("DSTACK_SERVER_ADMIN_TOKEN", "token123")

    with patch("src.bootstrap.safe_dstack_rest", side_effect=Exception("refused")):
        with patch("src.bootstrap._probe_dstack_cli", return_value=False):
            path = choose_access_path()

    assert path == "FAILED"


def test_c22_skipped_when_no_token(monkeypatch):
    """C2.2 probe is skipped when DSTACK_SERVER_ADMIN_TOKEN is empty."""
    monkeypatch.setenv("DSTACK_SERVER_ADMIN_TOKEN", "")

    with patch("src.bootstrap.safe_dstack_rest") as mock_rest:
        with patch("src.bootstrap._probe_dstack_cli", return_value=False):
            path = choose_access_path()

    # REST should NOT have been called (no token)
    mock_rest.assert_not_called()
    assert path == "FAILED"


def test_c22_chosen_with_list_response(monkeypatch):
    """C2.2 is chosen when REST returns a list directly."""
    monkeypatch.setenv("DSTACK_SERVER_ADMIN_TOKEN", "tok")

    mock_resp = MagicMock()
    mock_resp.json.return_value = []  # direct list

    with patch("src.bootstrap.safe_dstack_rest", return_value=mock_resp):
        path = choose_access_path()

    assert path == "C2.2"
