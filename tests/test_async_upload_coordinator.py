"""Unit tests for AsyncUploadCoordinator."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass
from typing import Any

import pytest

from training_signal_processing.runtime.async_upload_coordinator import (
    AsyncUploadCoordinator,
)


# --------------------------------------------------------------------- fakes
class FakeS3Client:
    """In-memory fake aioboto3 S3 client."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes]] = []
        self.raise_on: set[int] = set()
        # Set pause_event to an unset event to make uploads block until set.
        self.pause_event: threading.Event | None = None
        self._lock = threading.Lock()
        self.closed: bool = False

    async def __aenter__(self) -> "FakeS3Client":
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.closed = True

    async def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        with self._lock:
            idx = len(self.calls)
            self.calls.append((Key, Body))
        if self.pause_event is not None:
            while not self.pause_event.is_set():
                await asyncio.sleep(0.005)
        if idx in self.raise_on:
            raise RuntimeError(f"injected failure on call {idx}")


class FakeSession:
    def __init__(self, client: FakeS3Client) -> None:
        self._client = client

    def client(self, *args: Any, **kwargs: Any) -> FakeS3Client:
        return self._client


@dataclass
class _FakeCfg:
    bucket: str = "test-bucket"
    endpoint_url: str = "https://example.invalid"
    access_key_id: str = "AKIA"
    secret_access_key: str = "SECRET"
    region: str = "auto"


class _FakeObjectStore:
    def __init__(self) -> None:
        self.config = _FakeCfg()


def make_fake_r2_object_store() -> _FakeObjectStore:
    return _FakeObjectStore()


# ---------------------------------------------------------------- utilities
def _patch_aioboto3(monkeypatch: pytest.MonkeyPatch, client: FakeS3Client) -> None:
    """Install a FakeSession-returning aioboto3 for the lazy import."""
    import aioboto3

    monkeypatch.setattr(aioboto3, "Session", lambda: FakeSession(client))


# ---------------------------------------------------------------- test: 1
def test_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeS3Client()
    _patch_aioboto3(monkeypatch, fake_client)

    store = make_fake_r2_object_store()
    with AsyncUploadCoordinator(
        object_store=store, max_in_flight=4, max_queued=16
    ) as c:
        for i in range(20):
            c.submit(f"key-{i}", b"data")
        c.drain()

    assert len(fake_client.calls) == 20
    keys = {key for key, _ in fake_client.calls}
    assert keys == {f"key-{i}" for i in range(20)}


# ---------------------------------------------------------------- test: 2
def test_backpressure_blocks_third_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeS3Client()
    pause = threading.Event()
    fake_client.pause_event = pause
    _patch_aioboto3(monkeypatch, fake_client)

    store = make_fake_r2_object_store()
    c = AsyncUploadCoordinator(
        object_store=store, max_in_flight=4, max_queued=2
    )
    try:
        # First two submits fill the queue immediately.
        c.submit("k1", b"a")
        c.submit("k2", b"b")

        third_returned = threading.Event()

        def third() -> None:
            c.submit("k3", b"c")
            third_returned.set()

        t = threading.Thread(target=third)
        t.start()
        # Third should be blocked because queue is full and pause_event
        # keeps uploads from completing.
        assert not third_returned.wait(timeout=0.3)

        # Release uploads — third should now unblock and submit.
        pause.set()
        assert third_returned.wait(timeout=5.0)
        t.join(timeout=5.0)
        c.drain()
    finally:
        c.close()

    assert len(fake_client.calls) == 3


# ---------------------------------------------------------------- test: 3
def test_failure_surfacing_and_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeS3Client()
    fake_client.raise_on = {1}
    # Hold uploads until all three submits have been queued, so that the
    # loop thread cannot race ahead and publish _first_error mid-submission
    # (which would cause the 3rd submit to fail-fast before drain runs).
    pause = threading.Event()
    fake_client.pause_event = pause
    _patch_aioboto3(monkeypatch, fake_client)

    store = make_fake_r2_object_store()
    c = AsyncUploadCoordinator(
        object_store=store, max_in_flight=2, max_queued=16
    )
    try:
        for i in range(3):
            c.submit(f"k{i}", b"x")

        # Now release the uploads so k1 can raise.
        pause.set()

        with pytest.raises(RuntimeError, match="injected failure"):
            c.drain()

        # Next submit must re-raise the same error (fail-fast chain).
        with pytest.raises(RuntimeError, match="injected failure"):
            c.submit("k-next", b"y")
    finally:
        c.close()


# ---------------------------------------------------------------- test: 4
def test_loop_thread_death_surfaces_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeS3Client()
    _patch_aioboto3(monkeypatch, fake_client)

    store = make_fake_r2_object_store()
    c = AsyncUploadCoordinator(
        object_store=store, max_in_flight=2, max_queued=4
    )
    try:
        # Stop the loop manually; the thread will exit.
        c._loop.call_soon_threadsafe(c._loop.stop)
        c._thread.join(timeout=5.0)
        # Coordinator doesn't know yet, but run_coroutine_threadsafe will
        # raise RuntimeError which submit() translates.
        with pytest.warns(RuntimeWarning):
            with pytest.raises(RuntimeError, match="closed or loop thread dead"):
                c.submit("k", b"x")
    finally:
        # close() is best-effort here — loop is already dead.
        try:
            c.close()
        except Exception:
            pass


# ---------------------------------------------------------------- test: 5
def test_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeS3Client()
    _patch_aioboto3(monkeypatch, fake_client)
    store = make_fake_r2_object_store()
    c = AsyncUploadCoordinator(object_store=store)
    c.close()
    # Second call must be a no-op (no exception).
    c.close()


# ---------------------------------------------------------------- test: 6
def test_context_manager_normal_and_exception_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeS3Client()
    _patch_aioboto3(monkeypatch, fake_client)
    store = make_fake_r2_object_store()

    # Normal exit.
    with AsyncUploadCoordinator(object_store=store) as c:
        c.submit("k1", b"a")
        c.drain()
    assert c._closed is True

    # Exception inside block — exit still closes.
    fake_client2 = FakeS3Client()
    _patch_aioboto3(monkeypatch, fake_client2)
    with pytest.raises(ValueError):
        with AsyncUploadCoordinator(object_store=store) as c2:
            c2.submit("k2", b"b")
            raise ValueError("boom")
    assert c2._closed is True


# ---------------------------------------------------------------- test: 7
def test_real_coroutine_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove run_coroutine_threadsafe bridge works with a real await."""

    class SleepyClient(FakeS3Client):
        async def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
            await asyncio.sleep(0.01)
            with self._lock:
                self.calls.append((Key, Body))

    sleepy = SleepyClient()
    _patch_aioboto3(monkeypatch, sleepy)
    store = make_fake_r2_object_store()
    with AsyncUploadCoordinator(
        object_store=store, max_in_flight=3, max_queued=8
    ) as c:
        for i in range(5):
            c.submit(f"k{i}", b"x")
        c.drain()
    assert len(sleepy.calls) == 5


# ---------------------------------------------------------------- test: 8
def test_drain_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeS3Client()
    # Never-set pause event blocks forever.
    fake_client.pause_event = threading.Event()
    _patch_aioboto3(monkeypatch, fake_client)

    store = make_fake_r2_object_store()
    c = AsyncUploadCoordinator(
        object_store=store, max_in_flight=2, max_queued=4, drain_timeout=0.1
    )
    try:
        c.submit("k1", b"x")
        with pytest.raises(concurrent.futures.TimeoutError):
            c.drain(timeout=0.1)
    finally:
        # Release the upload so close() can proceed quickly.
        fake_client.pause_event.set()
        c.close()


# ---------------------------------------------------------------- test: 9
def test_abort_unblocks_queued_submitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeS3Client()
    pause = threading.Event()
    fake_client.pause_event = pause
    _patch_aioboto3(monkeypatch, fake_client)

    store = make_fake_r2_object_store()
    c = AsyncUploadCoordinator(
        object_store=store, max_in_flight=1, max_queued=1
    )

    errors: list[BaseException] = []
    second_done = threading.Event()
    first_submitted = threading.Event()

    def submitter() -> None:
        try:
            c.submit("k1", b"x")  # Consumes the single queue slot.
            first_submitted.set()
            c.submit("k2", b"y")  # Blocks on queue_semaphore.
        except BaseException as exc:
            errors.append(exc)
        finally:
            second_done.set()

    t = threading.Thread(target=submitter)
    t.start()

    # Wait for the first submit to land so we know the second is blocked.
    assert first_submitted.wait(timeout=5.0)
    # Give the second call a moment to reach queue_semaphore.acquire().
    time.sleep(0.1)
    assert not second_done.is_set()

    cancelled = c.abort()
    # Blocked submitter should unblock quickly.
    assert second_done.wait(timeout=5.0)
    t.join(timeout=5.0)

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "closed or loop thread dead" in str(errors[0])
    # abort() returned an int (0 or 1 depending on whether the first
    # in-flight upload completed before cancel — both are acceptable).
    assert isinstance(cancelled, int)

    # Idempotence: subsequent abort() returns 0.
    assert c.abort() == 0

    pause.set()
    try:
        c.close()
    except Exception:
        pass
