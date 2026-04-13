"""test_ring_buffer.py — RingBuffer unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import threading

from src.ring_buffer import RingBuffer


def test_append_and_snapshot():
    rb = RingBuffer(maxlen=10)
    rb.append("line1")
    rb.append("line2")
    lines, seq = rb.snapshot()
    assert lines == ["line1", "line2"]
    assert seq == 2


def test_snapshot_since_delta():
    rb = RingBuffer(maxlen=10)
    rb.append("a")
    rb.append("b")
    _, seq0 = rb.snapshot()
    rb.append("c")
    rb.append("d")
    lines, seq1 = rb.snapshot_since(seq0)
    assert lines == ["c", "d"]
    assert seq1 == 4


def test_snapshot_since_no_new():
    rb = RingBuffer(maxlen=10)
    rb.append("x")
    _, seq = rb.snapshot()
    lines, new_seq = rb.snapshot_since(seq)
    assert lines == []
    assert new_seq == seq


def test_bounded_maxlen():
    rb = RingBuffer(maxlen=3)
    for i in range(10):
        rb.append(str(i))
    lines, seq = rb.snapshot()
    assert len(lines) == 3
    # seq is still monotonic
    assert seq == 10


def test_monotonic_seq():
    rb = RingBuffer(maxlen=5)
    seqs = []
    for i in range(20):
        rb.append(f"line{i}")
        seqs.append(rb.seq)
    assert seqs == list(range(1, 21))


def test_thread_safe_concurrent_appends():
    rb = RingBuffer(maxlen=1000)

    def writer():
        for i in range(100):
            rb.append(f"line-{i}")

    threads = [threading.Thread(target=writer) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines, seq = rb.snapshot()
    assert seq == 500  # 5 * 100
    assert len(lines) == 500


def test_snapshot_since_with_overflow():
    """snapshot_since still works after ring overflows."""
    rb = RingBuffer(maxlen=3)
    rb.append("a")
    rb.append("b")
    seq_after_2 = rb.seq
    # Overflow: add 5 more, ring keeps only last 3
    for c in "cdefg":
        rb.append(c)
    # seq_after_2 = 2; all lines with seq > 2 are c(3),d(4),e(5),f(6),g(7)
    # but ring only holds last 3: e,f,g (seqs 5,6,7)
    lines, seq = rb.snapshot_since(seq_after_2)
    assert seq == 7
    # Only the 3 surviving lines have seq > 2
    assert set(lines).issubset({"c", "d", "e", "f", "g"})
