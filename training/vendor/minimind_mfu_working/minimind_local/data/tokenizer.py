"""Tokenizer artifact loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minimind_local.model.config import MiniMindEndToEndConfig

EXPECTED_BASE_VOCAB_SIZE = 50_000
EXPECTED_ADDED_SPECIAL_TOKEN_COUNT = 14
EXPECTED_TOTAL_VOCAB_SIZE = 50_014
EXPECTED_EOS_TOKEN = "<|endoftext|>"
EXPECTED_BOS_TOKEN = "<|im_start|>"


@dataclass(frozen=True)
class TokenizerArtifact:
    tokenizer_file: Path
    base_vocab_size: int
    added_special_tokens: tuple[str, ...]
    total_vocab_size: int
    bos_token: str
    eos_token: str


def load_tokenizer_artifact(tokenizer_dir: str | Path) -> TokenizerArtifact:
    directory = Path(tokenizer_dir)
    tokenizer_file = directory if directory.name.endswith(".json") else directory / "tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer JSON not found at {tokenizer_file}")
    payload = json.loads(tokenizer_file.read_text(encoding="utf-8"))
    model = payload.get("model", {})
    vocab = model.get("vocab", {})
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
            f"Expected {EXPECTED_ADDED_SPECIAL_TOKEN_COUNT} added special tokens, found "
            f"{len(added_tokens)} in {tokenizer_file}"
        )
    if total_vocab_size != EXPECTED_TOTAL_VOCAB_SIZE:
        raise ValueError(
            f"Expected total vocab size {EXPECTED_TOTAL_VOCAB_SIZE}, found {total_vocab_size} "
            f"in {tokenizer_file}"
        )
    if EXPECTED_BOS_TOKEN not in added_tokens:
        raise ValueError(f"Tokenizer artifact {tokenizer_file} does not include {EXPECTED_BOS_TOKEN}")
    if EXPECTED_EOS_TOKEN not in added_tokens:
        raise ValueError(f"Tokenizer artifact {tokenizer_file} does not include {EXPECTED_EOS_TOKEN}")
    return TokenizerArtifact(
        tokenizer_file=tokenizer_file,
        base_vocab_size=base_vocab_size,
        added_special_tokens=added_tokens,
        total_vocab_size=total_vocab_size,
        bos_token=EXPECTED_BOS_TOKEN,
        eos_token=EXPECTED_EOS_TOKEN,
    )


def load_native_superbpe_tokenizer(tokenizer_dir: str | Path) -> Any:
    artifact = load_tokenizer_artifact(tokenizer_dir)
    try:
        from transformers import PreTrainedTokenizerFast
        from tokenizers.decoders import ByteLevel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "transformers is required to load the standalone MiniMind recipe tokenizer"
        ) from exc

    special_tokens = [
        token
        for token in artifact.added_special_tokens
        if token not in {artifact.bos_token, artifact.eos_token}
    ]
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(artifact.tokenizer_file))
    tokenizer.backend_tokenizer.decoder = ByteLevel()
    tokenizer.add_special_tokens(
        {
            "bos_token": artifact.bos_token,
            "eos_token": artifact.eos_token,
            "additional_special_tokens": special_tokens,
        }
    )
    if getattr(tokenizer, "vocab_size", artifact.base_vocab_size) != artifact.base_vocab_size:
        raise ValueError(
            f"Expected tokenizer base vocab size {artifact.base_vocab_size}, found {tokenizer.vocab_size}"
        )
    if len(tokenizer) != artifact.total_vocab_size:
        raise ValueError(
            f"Expected tokenizer total vocab size {artifact.total_vocab_size}, found {len(tokenizer)}"
        )
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must provide BOS and EOS token ids")
    return tokenizer


def validate_tokenizer_matches_config(tokenizer: Any, config: MiniMindEndToEndConfig) -> None:
    tokenizer_size = len(tokenizer)
    if config.vocab_size != tokenizer_size:
        raise ValueError(
            f"Model vocab_size {config.vocab_size} does not match tokenizer size {tokenizer_size}"
        )


__all__ = [
    "TokenizerArtifact",
    "load_native_superbpe_tokenizer",
    "load_tokenizer_artifact",
    "validate_tokenizer_matches_config",
]
