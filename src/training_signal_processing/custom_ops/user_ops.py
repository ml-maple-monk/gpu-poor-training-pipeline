from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

from ..models import ExecutionLogEvent
from ..ops.base import Batch
from ..ops.builtin import (
    BatchTransformOp,
    ExportMarkdownMapper,
    MarkerOcrMapper,
    SkipExistingFilter,
    SourcePreparationOp,
)
from ..pipelines.ocr.models import PdfTask
from ..utils import join_s3_key, utc_isoformat

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.

"""
ADD NEW CONCRETE USER OPS TO THIS FILE OR TO ANOTHER MODULE IN THIS DIRECTORY.

Design contract:
- Define one concrete subclass per user-customized op.
- Set `op_name` so the class auto-registers on import.
- Inherit from `SourcePreparationOp`, `BatchTransformOp`, `SkipExistingFilter`,
  or `ExportMarkdownMapper` so the executor can infer the stage automatically.
- The user should only need to modify files in `custom_ops/` and the YAML recipe
  to add a new op.
"""

OCR_CONVERSION_TIMEOUT_SEC = 300


class MarkerConversionError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def build_marker_diagnostics(options: dict[str, object]) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "device_option": options.get("device"),
        "dtype_option": options.get("dtype"),
        "attention_implementation_option": options.get("attention_implementation"),
        "force_ocr": options.get("force_ocr"),
        "mp_start_method": "spawn",
    }
    try:
        import torch

        diagnostics["torch_cuda_available"] = torch.cuda.is_available()
        diagnostics["torch_cuda_device_count"] = torch.cuda.device_count()
    except Exception as exc:
        diagnostics["torch_cuda_error"] = str(exc)
    return diagnostics


def get_marker_mp_context() -> mp.context.BaseContext:
    return mp.get_context("spawn")


def _run_marker_conversion(
    pdf_path: str,
    options: dict[str, object],
    result_queue: mp.Queue,
) -> None:
    diagnostics = build_marker_diagnostics(options)
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        converter = PdfConverter(
            artifact_dict=create_model_dict(
                device=options.get("device"),
                dtype=options.get("dtype"),
                attention_implementation=options.get("attention_implementation"),
            ),
            processor_list=options.get("processor_list"),
            renderer=options.get("renderer"),
            config=options,
        )
        rendered = converter(pdf_path)
        markdown_text, _, _ = text_from_rendered(rendered)
        result_queue.put(
            {
                "status": "success",
                "markdown_text": markdown_text,
                "diagnostics": diagnostics,
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "status": "failed",
                "error_message": str(exc),
                "diagnostics": diagnostics,
            }
        )


def convert_pdf_bytes_with_timeout(
    pdf_bytes: bytes,
    options: dict[str, object],
) -> tuple[str, dict[str, object]]:
    with NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(pdf_bytes)
        temp_path = Path(handle.name)
    mp_context = get_marker_mp_context()
    result_queue = mp_context.Queue()
    timeout_sec = int(options.get("timeout_sec", OCR_CONVERSION_TIMEOUT_SEC))
    process = mp_context.Process(
        target=_run_marker_conversion,
        args=(str(temp_path), options, result_queue),
    )
    try:
        process.start()
        process.join(timeout=timeout_sec)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise MarkerConversionError(
                f"Marker OCR conversion timed out after {timeout_sec} seconds.",
                diagnostics={"timeout_sec": timeout_sec},
            )
        if result_queue.empty():
            raise MarkerConversionError(
                "Marker OCR conversion exited without returning a result.",
            )
        payload = result_queue.get()
        diagnostics = (
            dict(payload["diagnostics"])
            if isinstance(payload.get("diagnostics"), dict)
            else {}
        )
        if payload.get("status") != "success":
            raise MarkerConversionError(
                str(payload.get("error_message", "Marker OCR conversion failed.")),
                diagnostics=diagnostics,
            )
        return str(payload.get("markdown_text", "")), diagnostics
    finally:
        if hasattr(result_queue, "close"):
            result_queue.close()
        if hasattr(result_queue, "join_thread"):
            result_queue.join_thread()
        temp_path.unlink(missing_ok=True)


class IdentityPreviewOp(BatchTransformOp):
    """
    Minimal working example for local Ray op testing.

    Copy this class, rename `op_name`, and replace `process_batch` with your own logic.
    """

    op_name = "identity_preview"

    def process_batch(self, batch: Batch) -> Batch:
        return list(batch)


class PreparePdfDocumentOp(SourcePreparationOp):
    op_name = "prepare_pdf_document"

    def process_row(self, row: dict[str, object]) -> dict[str, object]:
        runtime = self.require_runtime()
        task = PdfTask.from_dict(row)
        markdown_name = Path(task.relative_path).with_suffix(".md").as_posix()
        return {
            "run_id": runtime.run_id,
            "source_r2_key": task.source_r2_key,
            "relative_path": task.relative_path,
            "source_size_bytes": task.source_size_bytes,
            "source_sha256": task.source_sha256,
            "markdown_r2_key": join_s3_key(runtime.output_root_key, f"markdown/{markdown_name}"),
            "status": "pending",
            "error_message": "",
            "started_at": "",
            "finished_at": "",
            "duration_sec": 0.0,
            "marker_exit_code": 0,
            "markdown_text": "",
        }


class SkipExistingDocumentsOp(SkipExistingFilter):
    op_name = "skip_existing"

    def keep_row(self, row: dict[str, object]) -> bool:
        runtime = self.require_runtime()
        if runtime.allow_overwrite:
            return True
        return str(row["source_r2_key"]) not in runtime.completed_item_keys


class MarkerOcrDocumentOp(MarkerOcrMapper):
    op_name = "marker_ocr"

    def process_row(self, row: dict[str, object]) -> dict[str, object]:
        runtime = self.require_runtime()
        started_at = utc_isoformat()
        started_clock = perf_counter()
        diagnostics = build_marker_diagnostics(dict(self.options))
        try:
            self.log_runtime_event(
                runtime,
                code="ocr.pdf.read.start",
                message="Starting PDF read for OCR.",
                details={"source_r2_key": str(row["source_r2_key"]), "diagnostics": diagnostics},
            )
            pdf_bytes = runtime.get_object_store().read_bytes(str(row["source_r2_key"]))
            diagnostics["pdf_bytes_loaded"] = len(pdf_bytes)
            self.log_runtime_event(
                runtime,
                code="ocr.converter.init.start",
                message="Starting OCR converter process.",
                details={"diagnostics": diagnostics},
            )
            markdown_text, conversion_diagnostics = self.convert_pdf_bytes(pdf_bytes)
            diagnostics.update(conversion_diagnostics)
            status = "success"
            error_message = ""
            marker_exit_code = 0
            self.log_runtime_event(
                runtime,
                code="ocr.converter.init.finish",
                message="OCR converter process completed successfully.",
                details={"diagnostics": diagnostics},
            )
        except Exception as exc:
            markdown_text = ""
            status = "failed"
            error_message = str(exc)
            marker_exit_code = 1
            diagnostics.update(getattr(exc, "diagnostics", {}))
            diagnostics["failure"] = error_message
            self.log_runtime_event(
                runtime,
                code="ocr.converter.failed",
                message="OCR converter process failed.",
                details={"diagnostics": diagnostics, "error": error_message},
            )
        finished_at = utc_isoformat()
        return {
            **row,
            "run_id": runtime.run_id,
            "status": status,
            "error_message": error_message,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": perf_counter() - started_clock,
            "marker_exit_code": marker_exit_code,
            "markdown_text": markdown_text,
            "diagnostics": diagnostics,
        }

    def convert_pdf_bytes(self, pdf_bytes: bytes) -> tuple[str, dict[str, object]]:
        return convert_pdf_bytes_with_timeout(pdf_bytes, dict(self.options))

    def log_runtime_event(
        self,
        runtime: object,
        *,
        code: str,
        message: str,
        details: dict[str, object],
    ) -> None:
        logger = getattr(runtime, "logger", None)
        run_id = getattr(runtime, "run_id", "")
        if logger is None or not run_id:
            return
        logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code=code,
                message=message,
                run_id=run_id,
                op_name=self.op_name,
                details=details,
            )
        )


class ExportMarkdownResultOp(ExportMarkdownMapper):
    op_name = "export_markdown"

    def process_row(self, row: dict[str, object]) -> dict[str, object]:
        status = str(row["status"])
        if status != "success":
            return {**row, "markdown_text": ""}
        markdown_text = str(row.get("markdown_text", ""))
        if not markdown_text:
            raise ValueError("Successful OCR rows must include non-empty markdown_text.")
        if not str(row.get("markdown_r2_key", "")):
            raise ValueError("Successful OCR rows must include markdown_r2_key.")
        return dict(row)
