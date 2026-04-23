"""Asyncio/aioboto3-backed upload coordinator for driver-side R2 writes.

This coordinator owns a single daemon thread that hosts a dedicated asyncio
event loop + a long-lived aioboto3 S3 client. Callers on the driver submit
``(key, body)`` pairs; uploads run concurrently (bounded by an asyncio
semaphore) and are surfaced as a single "first error" via :meth:`drain`.

Lifetime: one coordinator per run, owned by the executor. It is NOT designed
to be pickled — see :meth:`__getstate__`.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type-only
    from ..storage.object_store import R2ObjectStore


logger = logging.getLogger(__name__)


class AsyncUploadCoordinator:
    """Bounded async upload queue backed by aioboto3 on a dedicated loop thread."""

    def __init__(
        self,
        *,
        object_store: "R2ObjectStore",
        max_in_flight: int = 8,
        max_queued: int = 32,
        drain_timeout: float | None = 300.0,
    ) -> None:
        # Lazy import so worker environments that pickle other state never
        # pay for the aioboto3 import cost.
        import aioboto3  # noqa: PLC0415  (intentional lazy import)

        if max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1")
        if max_queued < 1:
            raise ValueError("max_queued must be >= 1")

        cfg = object_store.config
        self._bucket: str = cfg.bucket
        self._endpoint_url: str = cfg.endpoint_url
        self._access_key_id: str = cfg.access_key_id
        self._secret_access_key: str = cfg.secret_access_key
        self._region: str | None = cfg.region or None

        self._max_in_flight: int = max_in_flight
        self._max_queued: int = max_queued
        self._drain_timeout: float | None = drain_timeout

        self._queue_semaphore: threading.Semaphore = threading.Semaphore(max_queued)
        self._lock: threading.Lock = threading.Lock()
        self._pending: set[concurrent.futures.Future] = set()
        self._first_error: BaseException | None = None
        self._closed: bool = False
        self._loop_alive: bool = True

        # Track queue-semaphore slots acquired but not yet released (so
        # abort() can release exactly the right number of waiters).
        self._acquired_slots: int = 0

        # Start the loop thread.
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="AsyncUploadCoordinator-loop",
            daemon=True,
        )
        self._thread.start()
        self._loop_ready.wait()

        # Build aioboto3 session + client on the loop thread.
        self._session = aioboto3.Session()
        self._client_cm = self._session.client(
            "s3",
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._access_key_id,
            aws_secret_access_key=self._secret_access_key,
            region_name=self._region,
        )

        init_fut = asyncio.run_coroutine_threadsafe(self._init_async(), self._loop)
        try:
            init_fut.result()
        except BaseException:
            # Teardown the loop thread on failed init so we don't leak it.
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)
            self._loop_alive = False
            raise

        atexit.register(self.close)

    # ------------------------------------------------------------------ loop
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        try:
            self._loop.run_forever()
        finally:
            try:
                self._loop.close()
            except Exception:  # pragma: no cover - defensive
                pass

    async def _init_async(self) -> None:
        # Enter the aioboto3 client context manager to materialise the real
        # async client, and create the in-flight semaphore bound to this loop.
        self._client = await self._client_cm.__aenter__()
        self._inflight_semaphore: asyncio.Semaphore = asyncio.Semaphore(
            self._max_in_flight
        )

    # ------------------------------------------------------------------ api
    def submit(self, key: str, body: bytes) -> None:
        # Fail-fast: surface any previous upload error to every caller.
        if self._first_error is not None:
            raise self._first_error
        if self._closed or not self._loop_alive:
            raise RuntimeError("coordinator closed or loop thread dead")

        # Backpressure: block until a queue slot is available OR the
        # coordinator is aborted (abort releases all slots).
        self._queue_semaphore.acquire()

        # Re-check state AFTER acquiring (abort may have released us).
        if self._closed or not self._loop_alive:
            # Don't hold on to the slot — abort already accounted for it.
            raise RuntimeError("coordinator closed or loop thread dead")
        if self._first_error is not None:
            # Release the slot we just grabbed before re-raising.
            self._queue_semaphore.release()
            raise self._first_error

        with self._lock:
            self._acquired_slots += 1

        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._upload(key, body), self._loop
            )
        except RuntimeError:
            # Loop thread may have died between the earlier check and now.
            with self._lock:
                self._acquired_slots -= 1
            self._queue_semaphore.release()
            self._loop_alive = False
            raise RuntimeError("coordinator closed or loop thread dead") from None

        with self._lock:
            self._pending.add(fut)
        fut.add_done_callback(self._on_done)

    async def _upload(self, key: str, body: bytes) -> None:
        async with self._inflight_semaphore:
            await self._client.put_object(
                Bucket=self._bucket, Key=key, Body=body
            )

    def _on_done(self, fut: concurrent.futures.Future) -> None:
        with self._lock:
            self._pending.discard(fut)
            if self._acquired_slots > 0:
                self._acquired_slots -= 1
                release = True
            else:
                release = False
        if release:
            self._queue_semaphore.release()

        # Record first error (ignore cancellation — abort-caused).
        try:
            exc = fut.exception()
        except concurrent.futures.CancelledError:
            return
        except Exception:  # pragma: no cover - defensive
            return
        if exc is not None:
            with self._lock:
                if self._first_error is None:
                    self._first_error = exc

    def drain(self, timeout: float | None = None) -> None:
        effective = timeout if timeout is not None else self._drain_timeout

        with self._lock:
            snapshot = list(self._pending)

        for fut in snapshot:
            try:
                fut.result(timeout=effective)
            except concurrent.futures.TimeoutError:
                raise
            except concurrent.futures.CancelledError:
                # Treat as resolved (abort path).
                continue
            except BaseException as exc:
                # Capture directly: _on_done runs on the loop thread and may
                # not yet have fired by the time .result() returns. Recording
                # here guarantees _first_error is set before drain() returns.
                with self._lock:
                    if self._first_error is None:
                        self._first_error = exc

        if self._first_error is not None:
            raise self._first_error

    def abort(self) -> int:
        with self._lock:
            if self._closed:
                return 0
            self._closed = True
            snapshot = list(self._pending)

        cancelled = 0
        for fut in snapshot:
            if fut.done():
                continue
            if fut.cancel():
                cancelled += 1

        # Release ALL outstanding queue-semaphore slots so blocked
        # submitters unblock and re-raise. Best-effort: release up to
        # max_queued times to be safe, clamped by acquired_slots.
        with self._lock:
            to_release = self._acquired_slots
            self._acquired_slots = 0
        # Also release any extra waiters beyond in-flight work (blocked
        # submitters that grabbed ._queue_semaphore.acquire() above the
        # known pending count).
        for _ in range(to_release + self._max_queued):
            self._queue_semaphore.release()

        return cancelled

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            # Mark closed up-front to block new submits during teardown.
            self._closed = True

        # Best-effort drain of any work already in flight.
        try:
            with self._lock:
                snapshot = list(self._pending)
            for fut in snapshot:
                try:
                    fut.result(timeout=self._drain_timeout)
                except BaseException:
                    # Drain is best-effort in close(); log and continue.
                    pass
        except BaseException as exc:  # pragma: no cover - defensive
            logger.warning("AsyncUploadCoordinator close-drain raised: %r", exc)

        # Close aioboto3 client on the loop.
        if self._loop_alive and self._loop.is_running():
            try:
                close_fut = asyncio.run_coroutine_threadsafe(
                    self._client_cm.__aexit__(None, None, None), self._loop
                )
                close_fut.result(timeout=10)
            except BaseException as exc:
                logger.warning(
                    "AsyncUploadCoordinator client close raised: %r", exc
                )

            # Stop the loop + join the thread.
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                pass

        self._thread.join(timeout=10.0)
        self._loop_alive = False

        # Release any submitters still blocked on queue_semaphore so they
        # observe the closed state and raise RuntimeError.
        for _ in range(self._max_queued):
            self._queue_semaphore.release()

    def depth(self) -> int:
        with self._lock:
            return len(self._pending)

    # -------------------------------------------------------- context manager
    def __enter__(self) -> "AsyncUploadCoordinator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------ pickle guard
    def __getstate__(self) -> dict:
        raise TypeError(
            "AsyncUploadCoordinator is not picklable; it must only live on the driver"
        )
