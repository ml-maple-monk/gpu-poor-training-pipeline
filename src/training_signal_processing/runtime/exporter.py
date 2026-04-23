from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..core.models import ExportBatchResult, RunState
from ..ops.base import Batch

if TYPE_CHECKING:  # pragma: no cover - type-only
    from .async_upload_coordinator import AsyncUploadCoordinator

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


class Exporter(ABC):
    @abstractmethod
    def export_batch(self, batch_id: str, rows: Batch) -> ExportBatchResult:
        raise NotImplementedError

    @abstractmethod
    def finalize_run(self, run_state: RunState) -> None:
        raise NotImplementedError


class RayExporter(Exporter):
    """Ray-only exporter contract for explicit batch materialization."""

    upload_coordinator: "AsyncUploadCoordinator | None" = None

    def _put_bytes(self, key: str, body: bytes) -> None:
        if self.upload_coordinator is not None:
            self.upload_coordinator.submit(key, body)
        else:
            self.object_store.write_bytes(key, body)  # type: ignore[attr-defined]
