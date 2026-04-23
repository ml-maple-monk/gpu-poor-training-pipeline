from __future__ import annotations

from pathlib import Path

import pytest

from training_signal_processing.custom_ops import user_ops
from training_signal_processing.pipelines.ocr import config as ocr_config
from training_signal_processing.pipelines.ocr.config import load_recipe_config
from training_signal_processing.pipelines.ocr.exporter import OcrMarkdownExporter
from training_signal_processing.pipelines.ocr.submission import OcrSubmissionAdapter
from training_signal_processing.runtime.submission import (
    ArtifactStore,
    AsyncCommandHandle,
    AsyncCommandRunner,
    BootstrapSpec,
    CommandOutput,
    LocalAsyncUploadSpec,
    PreparedRun,
    RemoteInvocationSpec,
    RemoteTransport,
    SubmissionAdapter,
    SubmissionCoordinator,
)


class FakeArtifactStore(ArtifactStore):
    def __init__(self) -> None:
        self.bucket = "test-bucket"
        self.written_json: dict[str, dict[str, object]] = {}
        self.written_jsonl: dict[str, list[dict[str, object]]] = {}
        self.uploaded_files: list[tuple[Path, str]] = []

    def exists(self, key: str) -> bool:
        return key in self.written_json or key in self.written_jsonl

    def read_json(self, key: str) -> dict[str, object]:
        return self.written_json[key]

    def read_jsonl(self, key: str) -> list[dict[str, object]]:
        return self.written_jsonl[key]

    def write_json(self, key: str, value: dict[str, object]) -> None:
        self.written_json[key] = value

    def write_jsonl(self, key: str, rows: list[dict[str, object]]) -> None:
        self.written_jsonl[key] = rows

    def upload_file(self, path: Path, key: str) -> None:
        self.uploaded_files.append((path, key))

    def build_remote_env(self) -> dict[str, str]:
        return {"R2_BUCKET": self.bucket}


class FakeObjectStore:
    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}

    def write_bytes(self, key: str, body: bytes) -> None:
        self.payloads[key] = body


class FakeAsyncHandle(AsyncCommandHandle):
    def __init__(self) -> None:
        self.wait_called = False
        self.terminate_called = False
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> CommandOutput:
        self.wait_called = True
        self.returncode = 0 if self.returncode is None else self.returncode
        return CommandOutput(stdout="", stderr="")

    def terminate(self) -> None:
        self.terminate_called = True
        self.returncode = -15


class FakeAsyncRunner(AsyncCommandRunner):
    def __init__(self) -> None:
        self.started_commands: list[list[str]] = []
        self.started_envs: list[dict[str, str] | None] = []
        self.handle = FakeAsyncHandle()

    def start(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> AsyncCommandHandle:
        self.started_commands.append(command)
        self.started_envs.append(env)
        return self.handle


class FakeRemoteTransport(RemoteTransport):
    def __init__(self, *, fail_execute: bool = False) -> None:
        self.fail_execute = fail_execute
        self.events: list[str] = []

    def describe(self) -> dict[str, object]:
        return {"transport": "fake"}

    def sync(self, *, local_paths: tuple[str, ...], remote_root: str) -> None:
        self.events.append("sync")

    def bootstrap(self, *, remote_root: str, spec: BootstrapSpec) -> CommandOutput:
        self.events.append("bootstrap")
        return CommandOutput(stdout="", stderr="")

    def execute(self, *, remote_root: str, spec: RemoteInvocationSpec) -> CommandOutput:
        self.events.append("execute")
        if self.fail_execute:
            raise RuntimeError("remote execute failed")
        return CommandOutput(stdout='{"status":"success"}', stderr="")


class FakeSubmissionAdapter(SubmissionAdapter):
    def __init__(self, prepared_run: PreparedRun) -> None:
        self.prepared_run = prepared_run

    def prepare_new_run(self, artifact_store: ArtifactStore, *, dry_run: bool) -> PreparedRun:
        return self.prepared_run

    def prepare_resume_run(self, artifact_store: ArtifactStore, run_id: str) -> PreparedRun:
        return self.prepared_run

    def parse_remote_summary(self, stdout: str) -> dict[str, object]:
        return {"status": "success"}


@pytest.fixture()
def ocr_upload_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pdf_root = tmp_path / "pdfs"
    (pdf_root / "nested").mkdir(parents=True)
    (pdf_root / "alpha.pdf").write_bytes(b"%PDF-alpha")
    (pdf_root / "nested" / "beta.pdf").write_bytes(b"%PDF-beta")
    r2_config = tmp_path / "r2.env"
    r2_config.write_text(
        "\n".join(
            [
                "AWS_ACCESS_KEY_ID=test-access",
                "AWS_SECRET_ACCESS_KEY=test-secret",
                "AWS_DEFAULT_REGION=auto",
                "MLFLOW_S3_ENDPOINT_URL=https://example.r2.cloudflarestorage.com",
                "R2_BUCKET_NAME=test-bucket",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "ocr.yaml"
    config_path.write_text(
        f"""
run:
  name: test-ocr
  config_version: 1
ssh:
  host: localhost
  port: 22
  user: root
  identity_file: ~/.ssh/id_ed25519
remote:
  root_dir: /tmp/ocr
  python_version: "3.12"
ray:
  executor_type: ray
  batch_size: 1
  concurrency: 1
  target_num_blocks: 1
  ocr_worker_num_gpus: 1.0
  ocr_worker_num_cpus: 4
r2:
  config_file: {r2_config}
  bucket: test-bucket
  raw_pdf_prefix: dataset/raw/pdf
  output_prefix: dataset/processed/pdf_ocr
input:
  local_pdf_root: {pdf_root}
  include_glob: "**/*.pdf"
  max_files: 2
mlflow:
  enabled: false
  local_tracking_uri: http://127.0.0.1:5000
  remote_tunnel_port: 15000
  experiment_name: x
observability:
  flush_interval_sec: 5
  log_per_file_events: true
  heartbeat_interval_sec: 10
resumability:
  strategy: batch_manifest
  commit_every_batches: 1
  resume_mode: latest
ops:
  - name: prepare_pdf_document
    type: mapper
  - name: skip_existing
    type: filter
  - name: marker_ocr
    type: mapper
    force_ocr: true
  - name: export_markdown
    type: mapper
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(ocr_config, "CURRENT_MACHINE_PATH", tmp_path / "missing-machine")
    return config_path


def test_ocr_prepare_new_run_builds_async_upload_spec(
    monkeypatch: pytest.MonkeyPatch,
    ocr_upload_config: Path,
) -> None:
    config = load_recipe_config(ocr_upload_config)
    artifact_store = FakeArtifactStore()
    adapter = OcrSubmissionAdapter(
        config=config,
        config_path=ocr_upload_config,
        overrides=[],
    )
    monkeypatch.setattr(
        "training_signal_processing.pipelines.ocr.submission.shutil.which",
        lambda name: "/usr/bin/rclone" if name == "rclone" else None,
    )
    prepared = adapter.prepare_new_run(artifact_store, dry_run=False)

    assert prepared.uploaded_items == 0
    assert artifact_store.uploaded_files == []
    assert prepared.async_upload is not None
    assert prepared.async_upload.command[:4] == (
        "/usr/bin/rclone",
        "copy",
        config.input.local_pdf_root,
        "ocrinput:test-bucket/dataset/raw/pdf",
    )
    file_list_path = Path(prepared.async_upload.command[5])
    assert file_list_path.is_file()
    assert file_list_path.read_text(encoding="utf-8").splitlines() == [
        "alpha.pdf",
        "nested/beta.pdf",
    ]
    manifest_key = f"{config.r2.output_prefix}/{prepared.run_id}/control/input_manifest.jsonl"
    assert manifest_key in artifact_store.written_jsonl
    assert prepared.async_upload.env["RCLONE_CONFIG_OCRINPUT_PROVIDER"] == "Cloudflare"


def test_ocr_prepare_new_run_dry_run_skips_async_upload(ocr_upload_config: Path) -> None:
    config = load_recipe_config(ocr_upload_config)
    artifact_store = FakeArtifactStore()
    prepared = OcrSubmissionAdapter(
        config=config,
        config_path=ocr_upload_config,
        overrides=[],
    ).prepare_new_run(artifact_store, dry_run=True)

    assert prepared.async_upload is None
    assert artifact_store.written_json == {}
    assert artifact_store.written_jsonl == {}


def test_ocr_prepare_new_run_requires_rclone(
    monkeypatch: pytest.MonkeyPatch,
    ocr_upload_config: Path,
) -> None:
    config = load_recipe_config(ocr_upload_config)
    artifact_store = FakeArtifactStore()
    adapter = OcrSubmissionAdapter(
        config=config,
        config_path=ocr_upload_config,
        overrides=[],
    )
    monkeypatch.setattr(
        "training_signal_processing.pipelines.ocr.submission.shutil.which",
        lambda name: None,
    )

    with pytest.raises(RuntimeError, match="rclone is required"):
        adapter.prepare_new_run(artifact_store, dry_run=False)

    assert artifact_store.uploaded_files == []


def test_submission_coordinator_waits_for_async_upload_success(tmp_path: Path) -> None:
    cleanup_path = tmp_path / "upload-files.txt"
    cleanup_path.write_text("alpha.pdf\n", encoding="utf-8")
    prepared_run = PreparedRun(
        run_id="run-001",
        remote_root="/tmp/ocr",
        sync_paths=("src",),
        bootstrap=BootstrapSpec(command="echo bootstrap"),
        invocation=RemoteInvocationSpec(command="echo remote"),
        async_upload=LocalAsyncUploadSpec(
            command=("rclone", "copy"),
            cleanup_paths=(str(cleanup_path),),
        ),
    )
    async_runner = FakeAsyncRunner()
    transport = FakeRemoteTransport()
    result = SubmissionCoordinator(
        adapter=FakeSubmissionAdapter(prepared_run),
        artifact_store=FakeArtifactStore(),
        remote_transport=transport,
        async_command_runner=async_runner,
    ).submit(dry_run=False)

    assert result.mode == "executed"
    assert transport.events == ["sync", "bootstrap", "execute"]
    assert async_runner.started_commands == [["rclone", "copy"]]
    assert async_runner.handle.wait_called is True
    assert cleanup_path.exists() is False


def test_submission_coordinator_terminates_async_upload_on_remote_failure(tmp_path: Path) -> None:
    cleanup_path = tmp_path / "upload-files.txt"
    cleanup_path.write_text("alpha.pdf\n", encoding="utf-8")
    prepared_run = PreparedRun(
        run_id="run-001",
        remote_root="/tmp/ocr",
        sync_paths=("src",),
        bootstrap=BootstrapSpec(command="echo bootstrap"),
        invocation=RemoteInvocationSpec(command="echo remote"),
        async_upload=LocalAsyncUploadSpec(
            command=("rclone", "copy"),
            cleanup_paths=(str(cleanup_path),),
        ),
    )
    async_runner = FakeAsyncRunner()
    transport = FakeRemoteTransport(fail_execute=True)

    with pytest.raises(RuntimeError, match="remote execute failed"):
        SubmissionCoordinator(
            adapter=FakeSubmissionAdapter(prepared_run),
            artifact_store=FakeArtifactStore(),
            remote_transport=transport,
            async_command_runner=async_runner,
        ).submit(dry_run=False)

    assert async_runner.handle.terminate_called is True
    assert cleanup_path.exists() is False


def test_wait_for_source_object_polls_until_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}
    sleep_calls: list[float] = []

    class FakeObjectStore:
        def exists(self, key: str) -> bool:
            calls["count"] += 1
            return calls["count"] >= 3

    monkeypatch.setattr(user_ops, "sleep", lambda seconds: sleep_calls.append(seconds))

    user_ops.wait_for_source_object(
        FakeObjectStore(),
        key="dataset/raw/pdf/example.pdf",
        timeout_sec=5,
    )

    assert calls["count"] == 3
    assert len(sleep_calls) == 2


def test_wait_for_source_object_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeObjectStore:
        def exists(self, key: str) -> bool:
            return False

    times = iter([0.0, 0.0, 0.5, 0.5, 1.1, 1.1])
    monkeypatch.setattr(user_ops, "perf_counter", lambda: next(times))
    monkeypatch.setattr(user_ops, "sleep", lambda seconds: None)

    with pytest.raises(TimeoutError, match="did not appear"):
        user_ops.wait_for_source_object(
            FakeObjectStore(),
            key="dataset/raw/pdf/example.pdf",
            timeout_sec=1,
        )


def test_exporter_cleans_up_staged_pdfs_after_result_materialization(tmp_path: Path) -> None:
    success_pdf = tmp_path / "success.pdf"
    success_pdf.write_bytes(b"%PDF-success")
    failed_pdf = tmp_path / "failed.pdf"
    failed_pdf.write_bytes(b"%PDF-failed")
    object_store = FakeObjectStore()
    exporter = OcrMarkdownExporter(object_store)  # type: ignore[arg-type]

    result = exporter.export_batch(
        batch_id="batch-00001",
        rows=[
            {
                "run_id": "run-001",
                "source_r2_key": "dataset/raw/pdf/alpha.pdf",
                "relative_path": "alpha.pdf",
                "markdown_r2_key": "dataset/processed/pdf_ocr/run-001/markdown/alpha.md",
                "status": "success",
                "error_message": "",
                "source_sha256": "sha-alpha",
                "source_size_bytes": 10,
                "started_at": "",
                "finished_at": "",
                "duration_sec": 1.0,
                "marker_exit_code": 0,
                "markdown_text": "# alpha",
                "staged_pdf_path": str(success_pdf),
            },
            {
                "run_id": "run-001",
                "source_r2_key": "dataset/raw/pdf/beta.pdf",
                "relative_path": "beta.pdf",
                "markdown_r2_key": "dataset/processed/pdf_ocr/run-001/markdown/beta.md",
                "status": "failed",
                "error_message": "boom",
                "source_sha256": "sha-beta",
                "source_size_bytes": 20,
                "started_at": "",
                "finished_at": "",
                "duration_sec": 1.0,
                "marker_exit_code": 1,
                "markdown_text": "",
                "staged_pdf_path": str(failed_pdf),
            },
        ],
    )

    assert result.output_keys == ["dataset/processed/pdf_ocr/run-001/markdown/alpha.md"]
    assert (
        object_store.payloads["dataset/processed/pdf_ocr/run-001/markdown/alpha.md"]
        == b"# alpha"
    )
    assert success_pdf.exists() is False
    assert failed_pdf.exists() is False
