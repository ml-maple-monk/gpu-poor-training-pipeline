from __future__ import annotations

import json
import shlex
from pathlib import Path
from urllib.parse import urlparse

from ...core.utils import join_s3_key, utc_timestamp
from ...runtime.submission import (
    ArtifactRef,
    ArtifactStore,
    BootstrapSpec,
    PreparedRun,
    RemoteInvocationSpec,
    SubmissionAdapter,
)
from .config import load_resolved_recipe_mapping
from .models import EchoTask, RecipeConfig


class EchoSubmissionAdapter(SubmissionAdapter):
    def __init__(
        self,
        *,
        config: RecipeConfig,
        config_path: Path,
        overrides: list[str] | None = None,
        overlay_paths: tuple[Path, ...] = (),
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.overrides = overrides or []
        self.overlay_paths = tuple(overlay_paths)

    def prepare_new_run(self, artifact_store: ArtifactStore, *, dry_run: bool) -> PreparedRun:
        run_id = utc_timestamp()
        input_manifest_key = self.build_control_key(run_id, "input_manifest.jsonl")
        config_object_key = self.build_control_key(run_id, "recipe.json")
        tasks = self.build_tasks()
        if not dry_run:
            artifact_store.write_jsonl(input_manifest_key, [task.to_dict() for task in tasks])
            artifact_store.write_json(
                config_object_key,
                load_resolved_recipe_mapping(
                    self.config_path,
                    self.overrides,
                    overlay_paths=self.overlay_paths,
                ),
            )
        return self.build_prepared_run(
            run_id=run_id,
            input_manifest_key=input_manifest_key,
            config_object_key=config_object_key,
            discovered_items=len(tasks),
            uploaded_items=0,
            is_resume=False,
        )

    def prepare_resume_run(self, artifact_store: ArtifactStore, run_id: str) -> PreparedRun:
        input_manifest_key = self.build_control_key(run_id, "input_manifest.jsonl")
        config_object_key = self.build_control_key(run_id, "recipe.json")
        if not artifact_store.exists(input_manifest_key):
            raise ValueError(f"Resume manifest not found in R2: {input_manifest_key}")
        if not artifact_store.exists(config_object_key):
            raise ValueError(f"Resume recipe object not found in R2: {config_object_key}")
        manifest_rows = artifact_store.read_jsonl(input_manifest_key)
        return self.build_prepared_run(
            run_id=run_id,
            input_manifest_key=input_manifest_key,
            config_object_key=config_object_key,
            discovered_items=len(manifest_rows),
            uploaded_items=0,
            is_resume=True,
        )

    def parse_remote_summary(self, stdout: str) -> dict[str, object]:
        stripped = stdout.strip()
        if not stripped:
            raise ValueError("Remote job returned no JSON summary on stdout.")
        return json.loads(stripped)

    def build_prepared_run(
        self,
        *,
        run_id: str,
        input_manifest_key: str,
        config_object_key: str,
        discovered_items: int,
        uploaded_items: int,
        is_resume: bool,
    ) -> PreparedRun:
        return PreparedRun(
            run_id=run_id,
            remote_root=self.config.remote.root_dir,
            sync_paths=("pyproject.toml", "uv.lock", "src", "config"),
            bootstrap=self.build_bootstrap_spec(),
            invocation=self.build_invocation_spec(
                run_id=run_id,
                config_object_key=config_object_key,
                input_manifest_key=input_manifest_key,
            ),
            artifacts=(
                ArtifactRef(name="input_manifest", key=input_manifest_key, kind="jsonl"),
                ArtifactRef(name="config_object", key=config_object_key, kind="json"),
            ),
            discovered_items=discovered_items,
            uploaded_items=uploaded_items,
            is_resume=is_resume,
            metadata={
                "pipeline_family": "example_echo",
                "input_manifest_key": input_manifest_key,
                "config_object_key": config_object_key,
            },
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
        run_id: str,
        config_object_key: str,
        input_manifest_key: str,
    ) -> RemoteInvocationSpec:
        command = shlex.join(
            [
                "uv",
                "run",
                "--group",
                "remote_ocr",
                "python",
                "-m",
                "training_signal_processing.pipelines.example_echo.remote_job",
                "--run-id",
                run_id,
                "--config-object-key",
                config_object_key,
                "--input-manifest-key",
                input_manifest_key,
            ]
        )
        env: dict[str, str] = {}
        reverse_tunnels: tuple[str, ...] = ()
        tracking_uri = self.resolve_remote_tracking_uri()
        if tracking_uri:
            env["MLFLOW_TRACKING_URI"] = tracking_uri
            reverse_tunnels = (self.build_reverse_tunnel_spec(),)
        return RemoteInvocationSpec(
            command=command,
            env=env,
            reverse_tunnels=reverse_tunnels,
        )

    def resolve_remote_tracking_uri(self) -> str:
        if not self.config.mlflow.enabled:
            return ""
        return f"http://127.0.0.1:{self.config.mlflow.remote_tunnel_port}"

    def build_reverse_tunnel_spec(self) -> str:
        parsed = urlparse(self.config.mlflow.local_tracking_uri)
        if not parsed.hostname or not parsed.port:
            raise ValueError("mlflow.local_tracking_uri must include an explicit host and port.")
        return f"{self.config.mlflow.remote_tunnel_port}:{parsed.hostname}:{parsed.port}"

    def build_tasks(self) -> list[EchoTask]:
        return [
            EchoTask(source_id=str(item["source_id"]), message=str(item["message"]))
            for item in self.config.input.items
        ]

    def build_control_key(self, run_id: str, name: str) -> str:
        run_root = join_s3_key(self.config.r2.output_prefix, run_id)
        return join_s3_key(join_s3_key(run_root, "control"), name)
