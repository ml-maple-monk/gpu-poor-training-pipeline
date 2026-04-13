"""test_doctor.py — Preflight / doctor helper regression tests.

F6 item #6: `_read_windows_utc_timestamp` calls `powershell.exe` which can
hang indefinitely if the Windows host is wedged. The call must apply a
bounded timeout and degrade gracefully (return an empty string) so the
surrounding preflight can `reporter.warn(...)` instead of blocking forever.
"""

from __future__ import annotations

import subprocess

from gpupoor.ops import doctor


def test_doctor_powershell_times_out(monkeypatch):
    """_read_windows_utc_timestamp must tolerate a hung powershell.exe by
    returning an empty string, matching existing `win_ts = ""` failure
    semantics in _run_local_preflight (see doctor.py:238)."""

    def _fake_run(*args, **kwargs):
        # A PRE-condition of the fix: the call site passes `timeout=5`.
        # If it doesn't, this test will also fail because the TimeoutExpired
        # never fires — so the assertion is dual-purpose.
        assert "timeout" in kwargs and kwargs["timeout"] == 5, (
            f"expected _read_windows_utc_timestamp to pass timeout=5 to subprocess; got kwargs={kwargs!r}"
        )
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    # doctor.py currently uses subprocess.check_output; after the fix it will
    # use subprocess.run. We patch *both* entry points so the test is agnostic
    # to the exact call shape, and let the `timeout` assertion above keep the
    # fix honest.
    monkeypatch.setattr(subprocess, "check_output", _fake_run)
    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = doctor._read_windows_utc_timestamp()
    assert result == "", (
        "on powershell.exe timeout, _read_windows_utc_timestamp must return an "
        f"empty string so the preflight can fall through to reporter.warn(); got {result!r}"
    )
