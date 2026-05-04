from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
)


class _FakeTokenizerFast:
    def __init__(self, *, tokenizer_file: str) -> None:
        payload = json.loads(Path(tokenizer_file).read_text(encoding="utf-8"))
        self._vocab_size = len(payload["model"]["vocab"])
        self._added_tokens = [item["content"] for item in payload["added_tokens"]]
        self._lookup = {token: self._vocab_size + index for index, token in enumerate(self._added_tokens)}
        self.eos_token_id = None
        self.pad_token_id = None

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def add_special_tokens(self, mapping: dict[str, object]) -> int:
        self.eos_token_id = self._lookup[str(mapping["eos_token"])]
        self.pad_token_id = self._lookup[str(mapping["pad_token"])]
        return 0

    def __len__(self) -> int:
        return self._vocab_size + len(self._added_tokens)


def _write_tokenizer_artifact(directory: Path, *, base_vocab_size: int = 50_000) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {"type": "BPE", "vocab": {f"tok{index}": index for index in range(base_vocab_size)}},
        "added_tokens": [
            {
                "id": base_vocab_size + index,
                "content": token,
                "special": True,
            }
            for index, token in enumerate(SPECIAL_TOKENS)
        ],
    }
    (directory / "tokenizer.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def test_load_tokenizer_artifact_accepts_native_superbpe_layout(import_minimind_module, tmp_path) -> None:
    pretrain_tokenizer = import_minimind_module("minimind.trainer.pretrain_tokenizer")
    artifact_dir = _write_tokenizer_artifact(tmp_path / "tokenizer")

    artifact = pretrain_tokenizer.load_tokenizer_artifact(artifact_dir)

    assert artifact.base_vocab_size == 50_000
    assert artifact.total_vocab_size == 50_014
    assert artifact.eos_token == "<|endoftext|>"
    assert artifact.pad_token == "<|vision_pad|>"


def test_load_pretrain_tokenizer_registers_distinct_eos_and_pad_tokens(
    import_minimind_module,
    monkeypatch,
    tmp_path,
) -> None:
    pretrain_tokenizer = import_minimind_module("minimind.trainer.pretrain_tokenizer")
    artifact_dir = _write_tokenizer_artifact(tmp_path / "tokenizer")
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(PreTrainedTokenizerFast=_FakeTokenizerFast),
    )

    tokenizer = pretrain_tokenizer.load_pretrain_tokenizer(artifact_dir)

    assert tokenizer.vocab_size == 50_000
    assert len(tokenizer) == 50_014
    assert tokenizer.eos_token_id == 50_000
    assert tokenizer.pad_token_id == 50_011
