"""Print greedy checkpoint text continuations beside ground-truth text."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import click
import torch

from minimind_local.data.tokenized_parquet import TokenizedParquetDataset
from minimind_local.data.tokenizer import load_native_superbpe_tokenizer
from minimind_local.model.config import MiniMindEndToEndAxes, MiniMindEndToEndConfig
from minimind_local.model.module import build_minimind_end2end_module, unwrap_compiled_minimind_module


@dataclass(frozen=True)
class TextEvalCheckpointMetadata:
    config: MiniMindEndToEndConfig
    axes: MiniMindEndToEndAxes
    tokenizer_path: Path | None
    global_step: int


@click.command(name="minimind-eval-text", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--checkpoint", required=True, type=click.Path(path_type=Path, exists=True))
@click.option(
    "--tokenized-parquet-data",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    envvar="MINIMIND_TOKENIZED_PARQUET_DATA",
    show_envvar=True,
)
@click.option("--tokenizer", type=click.Path(path_type=Path), envvar="MINIMIND_TOKENIZER", show_envvar=True)
@click.option("--prompt-tokens", type=click.IntRange(1), default=64, show_default=True)
@click.option("--target-tokens", type=click.IntRange(1), default=64, show_default=True)
@click.option("--skip-samples", type=click.IntRange(0), default=0, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--dtype", type=click.Choice(("bfloat16", "float32")), default="bfloat16", show_default=True)
def text_eval_command(
    checkpoint: Path,
    tokenized_parquet_data: Path,
    tokenizer: Path | None,
    prompt_tokens: int,
    target_tokens: int,
    skip_samples: int,
    device: str,
    dtype: str,
) -> None:
    torch_device = torch.device(device)
    torch_dtype = _torch_dtype(dtype)
    model, metadata = load_checkpoint_for_text_eval(checkpoint, device=torch_device, dtype=torch_dtype)
    if prompt_tokens + target_tokens > metadata.config.sequence_length:
        raise click.ClickException(
            "prompt-tokens + target-tokens must not exceed checkpoint sequence_length "
            f"({metadata.config.sequence_length})"
        )

    tokenizer_path = tokenizer or metadata.tokenizer_path
    if tokenizer_path is None:
        raise click.ClickException(
            "tokenizer must be provided by --tokenizer, MINIMIND_TOKENIZER, or checkpoint metadata"
        )
    loaded_tokenizer = load_native_superbpe_tokenizer(tokenizer_path)
    try:
        source_ids = sample_ground_truth_ids(
            tokenized_parquet_data,
            eos_token_id=loaded_tokenizer.eos_token_id,
            min_tokens=prompt_tokens + target_tokens,
            skip_samples=skip_samples,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    prompt_ids = source_ids[:prompt_tokens]
    target_ids = source_ids[prompt_tokens : prompt_tokens + target_tokens]
    generated_ids = generate_greedy(
        model,
        prompt_ids,
        target_tokens=target_tokens,
        device=torch_device,
    )
    click.echo(
        format_text_eval_output(
            checkpoint_path=checkpoint,
            global_step=metadata.global_step,
            prompt_text=decode_ids(loaded_tokenizer, prompt_ids),
            generated_text=decode_ids(loaded_tokenizer, generated_ids),
            ground_truth_text=decode_ids(loaded_tokenizer, target_ids),
            prompt_token_count=int(prompt_ids.numel()),
            target_token_count=int(target_ids.numel()),
        )
    )


def load_checkpoint_for_text_eval(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, TextEvalCheckpointMetadata]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint_metadata_from_payload(payload)
    model = build_minimind_end2end_module(metadata.config, _inference_axes(metadata.axes), device, dtype)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, metadata


def checkpoint_metadata_from_payload(payload: dict[str, Any]) -> TextEvalCheckpointMetadata:
    try:
        config = MiniMindEndToEndConfig(**payload["config"])
        axes_payload = payload["axes"]
        axes = MiniMindEndToEndAxes(
            axes_payload["attention"],
            axes_payload["sparsity"],
            axes_payload["optimizer"],
            axes_payload.get("compile", "runtime"),
            axes_payload.get("precision", "bf16_training"),
        )
        tokenizer_path = payload.get("tokenizer_path")
        global_step = int(payload.get("global_step", 0))
    except KeyError as exc:
        raise ValueError(f"Checkpoint payload is missing required key {exc.args[0]!r}") from exc
    return TextEvalCheckpointMetadata(
        config=config,
        axes=axes,
        tokenizer_path=Path(tokenizer_path) if tokenizer_path else None,
        global_step=global_step,
    )


def _inference_axes(axes: MiniMindEndToEndAxes) -> MiniMindEndToEndAxes:
    if axes.precision == "fp8_training":
        return replace(axes, precision="bf16_training")
    return axes


def sample_ground_truth_ids(
    data_path: str | Path,
    *,
    eos_token_id: int,
    min_tokens: int,
    skip_samples: int,
) -> torch.Tensor:
    dataset = TokenizedParquetDataset(
        data_path,
        eos_token_id=eos_token_id,
        shuffle_buffer_size=0,
        shuffle_files=False,
    )
    for sample_index, token_ids in enumerate(dataset):
        if sample_index < skip_samples:
            continue
        if int(token_ids.numel()) >= min_tokens:
            return token_ids
    raise RuntimeError(
        f"No tokenized sample with at least {min_tokens} tokens found after skipping {skip_samples}"
    )


def generate_greedy(
    model: Any,
    prompt_ids: torch.Tensor,
    *,
    target_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    base_model = unwrap_compiled_minimind_module(model)
    generated = prompt_ids.to(device=device, dtype=torch.long).unsqueeze(0)
    max_context = int(getattr(base_model, "rope_cos").shape[0])
    with torch.inference_mode():
        for _ in range(target_tokens):
            if generated.size(1) >= max_context:
                break
            next_token = _next_greedy_token(base_model, generated)
            generated = torch.cat([generated, next_token], dim=1)
    return generated[0, prompt_ids.numel() :].detach().cpu()


def _next_greedy_token(base_model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
    position_embeddings = base_model.prepare_position_embeddings(position_ids)
    hidden = base_model._forward_hidden(input_ids, position_embeddings)
    logits = base_model.lm_head(base_model.norm(hidden))
    return torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)


def decode_ids(tokenizer: Any, token_ids: torch.Tensor) -> str:
    return tokenizer.decode(
        token_ids.detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def format_text_eval_output(
    *,
    checkpoint_path: Path,
    global_step: int,
    prompt_text: str,
    generated_text: str,
    ground_truth_text: str,
    prompt_token_count: int,
    target_token_count: int,
) -> str:
    return "\n".join(
        [
            "MiniMind checkpoint text evaluation",
            f"checkpoint: {checkpoint_path}",
            f"global_step: {global_step}",
            f"prompt_tokens: {prompt_token_count}",
            f"target_tokens: {target_token_count}",
            "",
            "=== Prompt ===",
            prompt_text,
            "",
            "=== Model Generated ===",
            generated_text,
            "",
            "=== Ground Truth ===",
            ground_truth_text,
        ]
    )


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else None
    return text_eval_command.main(args=args, prog_name="minimind-eval-text", standalone_mode=True)


__all__ = [
    "TextEvalCheckpointMetadata",
    "checkpoint_metadata_from_payload",
    "decode_ids",
    "format_text_eval_output",
    "generate_greedy",
    "load_checkpoint_for_text_eval",
    "main",
    "sample_ground_truth_ids",
    "text_eval_command",
]


if __name__ == "__main__":
    raise SystemExit(main())
