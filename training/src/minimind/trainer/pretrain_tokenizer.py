"""Tokenizer loading for the MiniMind pretraining pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TOKENIZER_DIR = Path(
    "/home/geeyang/workspace/training-signal-processing/tokenizers/native_superbpe_1m_rows_max4w"
)
EXPECTED_BASE_VOCAB_SIZE = 50_000
EXPECTED_ADDED_SPECIAL_TOKEN_COUNT = 14
EXPECTED_TOTAL_VOCAB_SIZE = 50_014
EXPECTED_EOS_TOKEN = "<|endoftext|>"
DEFAULT_PAD_TOKEN = "<|vision_pad|>"


@dataclass(frozen=True)
class TokenizerArtifact:
    tokenizer_file: Path
    base_vocab_size: int
    added_special_tokens: tuple[str, ...]
    total_vocab_size: int
    eos_token: str
    pad_token: str


def load_pretrain_tokenizer(tokenizer_path: str | Path = DEFAULT_TOKENIZER_DIR) -> Any:
    artifact = load_tokenizer_artifact(tokenizer_path)
    try:
        from transformers import PreTrainedTokenizerFast
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("transformers is required to load the MiniMind tokenizer") from exc

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(artifact.tokenizer_file))
    special_tokens = [
        token
        for token in artifact.added_special_tokens
        if token not in {artifact.eos_token, artifact.pad_token}
    ]
    tokenizer.add_special_tokens(
        {
            "eos_token": artifact.eos_token,
            "pad_token": artifact.pad_token,
            "additional_special_tokens": special_tokens,
        }
    )
    _validate_loaded_tokenizer(tokenizer, artifact)
    return tokenizer


def load_tokenizer_artifact(tokenizer_path: str | Path) -> TokenizerArtifact:
    path = Path(tokenizer_path)
    tokenizer_file = path if path.name.endswith(".json") else path / "tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer JSON not found at {tokenizer_file}")

    payload = json.loads(tokenizer_file.read_text(encoding="utf-8"))
    vocab = payload.get("model", {}).get("vocab", {})
    added_tokens = tuple(item.get("content", "") for item in payload.get("added_tokens", ()))
    base_vocab_size = len(vocab)
    total_vocab_size = base_vocab_size + len(added_tokens)

    if base_vocab_size != EXPECTED_BASE_VOCAB_SIZE:
        raise ValueError(
            f"Expected base vocab size {EXPECTED_BASE_VOCAB_SIZE}, found {base_vocab_size} "
            f"in {tokenizer_file}"
        )
    if len(added_tokens) != EXPECTED_ADDED_SPECIAL_TOKEN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ADDED_SPECIAL_TOKEN_COUNT} added special tokens, "
            f"found {len(added_tokens)} in {tokenizer_file}"
        )
    if total_vocab_size != EXPECTED_TOTAL_VOCAB_SIZE:
        raise ValueError(
            f"Expected total vocab size {EXPECTED_TOTAL_VOCAB_SIZE}, found {total_vocab_size} "
            f"in {tokenizer_file}"
        )
    for token in (EXPECTED_EOS_TOKEN, DEFAULT_PAD_TOKEN):
        if token not in added_tokens:
            raise ValueError(f"Tokenizer artifact {tokenizer_file} does not include {token}")

    return TokenizerArtifact(
        tokenizer_file=tokenizer_file,
        base_vocab_size=base_vocab_size,
        added_special_tokens=added_tokens,
        total_vocab_size=total_vocab_size,
        eos_token=EXPECTED_EOS_TOKEN,
        pad_token=DEFAULT_PAD_TOKEN,
    )


def _validate_loaded_tokenizer(tokenizer: Any, artifact: TokenizerArtifact) -> None:
    if getattr(tokenizer, "vocab_size", artifact.base_vocab_size) != artifact.base_vocab_size:
        raise ValueError(
            f"Expected tokenizer base vocab size {artifact.base_vocab_size}, found {tokenizer.vocab_size}"
        )
    if len(tokenizer) != artifact.total_vocab_size:
        raise ValueError(f"Expected tokenizer total vocab size {artifact.total_vocab_size}, found {len(tokenizer)}")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must provide an EOS token id")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must provide a pad token id")
    if tokenizer.eos_token_id == tokenizer.pad_token_id:
        raise ValueError("Tokenizer EOS and pad token ids must be distinct for pretraining labels")
