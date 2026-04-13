"""test_ring_buffer.py — RingBuffer unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import threading

from src.ring_buffer import RingBuffer


def test_snapshot_apis_preserve_append_order_and_delta_semantics():
    """Proves readers can observe the full ordered log stream and then fetch
    only new lines, which is the core dashboard log-delivery contract."""
    rb = RingBuffer(maxlen=10)
    rb.append("line1")
    rb.append("line2")
    lines, seq = rb.snapshot()
    assert lines == ["line1", "line2"]
    assert seq == 2

    _, seq0 = rb.snapshot()
    rb.append("line3")
    rb.append("line4")
    delta, seq1 = rb.snapshot_since(seq0)
    assert delta == ["line3", "line4"]
    no_new, seq2 = rb.snapshot_since(seq1)
    assert no_new == []
    assert seq2 == seq1


def test_ring_buffer_keeps_a_bounded_window_while_sequence_stays_monotonic():
    """Proves overflow discards old lines without rewinding sequence numbers,
    which is the safety property that lets sessions reason about deltas."""
    rb = RingBuffer(maxlen=3)
    seqs = []
    for i in range(10):
        rb.append(str(i))
        seqs.append(rb.seq)
    lines, seq = rb.snapshot()
    assert len(lines) == 3
    assert seq == 10
    assert seqs == list(range(1, 11))


def test_thread_safe_concurrent_appends():
    """Proves concurrent writers cannot corrupt the buffer, which protects the
    log fanout path used by multiple dashboard background threads."""
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
    """Proves delta reads still return the surviving tail after overflow, so
    late readers degrade gracefully instead of crashing or rewinding."""
    rb = RingBuffer(maxlen=3)
    rb.append("a")
    rb.append("b")
    seq_after_2 = rb.seq
    for c in "cdefg":
        rb.append(c)
    lines, seq = rb.snapshot_since(seq_after_2)
    assert seq == 7
    assert set(lines).issubset({"c", "d", "e", "f", "g"})
