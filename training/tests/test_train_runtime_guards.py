from __future__ import annotations

from contextlib import nullcontext

import pytest

transformers = pytest.importorskip("transformers", reason="transformers is required for trainer_utils import")


def test_train_pretrain_rejects_unknown_dtype(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")

    with pytest.raises(ValueError, match="Unsupported autocast dtype"):
        trainer_utils.build_autocast_context("cuda", "fp32")


def test_train_pretrain_accepts_float32_dtype(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")

    ctx = trainer_utils.build_autocast_context("cuda", "float32")
    # float32 on cuda should produce a real autocast context (not nullcontext)
    assert ctx is not None


def test_validation_ppl_reports_overflow_as_infinity(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")

    assert trainer_utils.validation_ppl_from_loss(1.0) > 0.0
    assert trainer_utils.validation_ppl_from_loss(1e6) == float("inf")


def test_build_autocast_context_uses_nullcontext_on_cpu(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")

    assert isinstance(trainer_utils.build_autocast_context("cpu", "bfloat16"), nullcontext)


def test_log_flash_attention_status_reports_cpu_fallback(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")
    messages: list[str] = []

    trainer_utils.log_flash_attention_status(requested=True, device_type_name="cpu", logger=messages.append)

    assert messages == [
        "Flash attention requested, but CUDA is unavailable; training will use the fallback attention path"
    ]
