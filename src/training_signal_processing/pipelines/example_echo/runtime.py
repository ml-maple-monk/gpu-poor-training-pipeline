from __future__ import annotations

import json
from typing import Any

from ...core.execution import (
    ObjectStorePipelineRuntimeAdapter,
    OutputCompletionTracker,
    RayExporter,
)
from ...core.models import ExportBatchResult, RunArtifactLayout, RuntimeRunBindings
from ...core.storage import ObjectStore, R2ObjectStore
from ...core.utils import join_s3_key
from .models import EchoResult, RecipeConfig


def build_echo_output_key(output_root_key: str, source_id: str) -> str:
    return join_s3_key(join_s3_key(output_root_key, "outputs"), f"{source_id}.json")


class EchoExporter(RayExporter):
    def __init__(self, object_store: ObjectStore) -> None:
        self.object_store = object_store

    def export_batch(
        self,
        batch_id: str,
        rows: list[dict[str, object]],
    ) -> ExportBatchResult:
        output_keys: list[str] = []
        for row in rows:
            result = EchoResult.from_dict(row)
            if result.status != "success":
                continue
            payload = {
                "source_id": result.source_id,
                "message": result.message,
                "echoed_at": result.echoed_at,
            }
            self._put_bytes(
                result.output_r2_key,
                json.dumps(payload, sort_keys=True).encode("utf-8"),
            )
            output_keys.append(result.output_r2_key)
        return ExportBatchResult(
            batch_id=batch_id,
            row_count=len(rows),
            output_keys=output_keys,
        )


class EchoCompletionTracker(OutputCompletionTracker):
    def source_key_for_input(self, row: dict[str, Any]) -> str:
        return str(row["source_id"])

    def output_key_for_input(
        self,
        row: dict[str, Any],
        artifact_layout: RunArtifactLayout,
    ) -> str:
        return build_echo_output_key(artifact_layout.output_root_key, str(row["source_id"]))

    def output_listing_prefix(self, artifact_layout: RunArtifactLayout) -> str:
        return join_s3_key(artifact_layout.output_root_key, "outputs")


def build_adapter(
    config: RecipeConfig,
    bindings: RuntimeRunBindings,
    object_store: R2ObjectStore,
) -> ObjectStorePipelineRuntimeAdapter:
    return ObjectStorePipelineRuntimeAdapter(
        config=config,
        bindings=bindings,
        object_store=object_store,
        source_root_key="example_echo",
        exporter_factory=EchoExporter,
        completion_tracker_factory=EchoCompletionTracker,
    )
