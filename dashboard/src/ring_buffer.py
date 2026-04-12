"""ring_buffer.py — Bounded deque with monotonic sequence numbers.

Used by LogTailer to store log lines and enable per-session delta pushes.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Sequence


class RingBuffer:
    """Thread-safe bounded log ring with monotonic sequence counter."""

    def __init__(self, maxlen: int = 500) -> None:
        self._buf: deque[tuple[int, str]] = deque(maxlen=maxlen)
        self._seq: int = 0
        self._lock = threading.Lock()

    def append(self, line: str) -> None:
        """Append a line and increment the sequence counter."""
        with self._lock:
            self._seq += 1
            self._buf.append((self._seq, line))

    def snapshot(self) -> tuple[list[str], int]:
        """Return (all_lines, current_seq)."""
        with self._lock:
            lines = [line for _, line in self._buf]
            return lines, self._seq

    def snapshot_since(self, session_seq: int) -> tuple[list[str], int]:
        """Return (new_lines_since_session_seq, current_seq).

        Only returns lines whose seq > session_seq (delta semantics).
        """
        with self._lock:
            new_lines = [line for seq, line in self._buf if seq > session_seq]
            return new_lines, self._seq

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
