from __future__ import annotations

import shlex

from ...core.submission import (
    ArtifactStore,
    BootstrapSpec,
    RemoteInvocationSpec,
    SubmissionAdapter,
    SubmissionManifest,
)
from .config import load_resolved_recipe_mapping
from .models import EchoTask, RecipeConfig


class EchoSubmissionAdapter(SubmissionAdapter):
    config: RecipeConfig

    def pipeline_family(self) -> str:
        return "example_echo"

    def build_new_run_manifest(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        dry_run: bool,
    ) -> SubmissionManifest:
        del artifact_store, run_id, dry_run
        tasks = self.build_tasks()
        return SubmissionManifest(
            rows=[task.to_dict() for task in tasks],
            discovered_items=len(tasks),
        )

    def load_resolved_recipe_mapping(self) -> dict[str, object]:
        return load_resolved_recipe_mapping(
            self.config_path,
            self.overrides,
            overlay_paths=self.overlay_paths,
        )

    def build_bootstrap_spec(self) -> BootstrapSpec:
        command = " && ".join(
            [
                "command -v uv >/dev/null",
                f"uv python install {shlex.quote(self.config.remote.python_version)}",
                "uv sync --group remote_ocr --no-dev",
            ]
        )
        return BootstrapSpec(command=command)

    def build_invocation_spec(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        config_object_key: str,
        input_manifest_key: str,
        uploaded_items: int,
    ) -> RemoteInvocationSpec:
        del artifact_store, uploaded_items
        command = shlex.join(
            [
                "uv",
                "run",
                "--group",
                "remote_ocr",
                "python",
                "-m",
                "training_signal_processing.pipelines.example_echo.cli",
                "remote-job",
                "--run-id",
                run_id,
                "--config-object-key",
                config_object_key,
                "--input-manifest-key",
                input_manifest_key,
            ]
        )
        return RemoteInvocationSpec(command=command, env={})

    def build_tasks(self) -> list[EchoTask]:
        return [
            EchoTask(source_id=str(item["source_id"]), message=str(item["message"]))
            for item in self.config.input.items
        ]
