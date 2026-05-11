#!/usr/bin/env python3
"""Inspect MiniMind next-token predictions against real tokenized parquet samples."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).parent.parent
MINIMIND_SRC = REPO_ROOT / "training" / "src" / "minimind"
DEFAULT_CONFIG = Path("/tmp/gpupoor-minimind-1m-4096-muon8bit.toml")

sys.path.insert(0, str(MINIMIND_SRC))

from dataset.lm_dataset import PretrainDataCollator  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from trainer.pretrain_config import build_minimind_config, runtime_args_from_toml  # noqa: E402
from trainer.pretrain_data import TokenizedParquetDataset, is_tokenized_parquet_dataset  # noqa: E402
from trainer.pretrain_tokenizer import load_pretrain_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a MiniMind pretrain checkpoint read-only and print next-token predictions "
            "beside real ground-truth tokens from the tokenized parquet train split."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"Pretrain TOML. Default: {DEFAULT_CONFIG}")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Explicit checkpoint .pth to evaluate.")
    parser.add_argument("--device", default="cpu", help="Evaluation device. Default: cpu to avoid training GPU contention.")
    parser.add_argument("--samples", type=int, default=4, help="Number of real samples to display.")
    parser.add_argument("--positions-per-sample", type=int, default=8, help="Prediction positions to show per sample.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidate tokens to show per position.")
    parser.add_argument("--context-tokens", type=int, default=256, help="Real tokens to feed from each sample.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for sample selection.")
    parser.add_argument(
        "--sample-mode",
        choices=("shuffled", "sequential"),
        default="shuffled",
        help="Use the parquet train iterator shuffled or in file order.",
    )
    parser.add_argument(
        "--context-tail-chars",
        type=int,
        default=120,
        help="Maximum decoded context-tail characters displayed in each table row.",
    )
    return parser.parse_args()


def _die(message: str, exit_code: int = 2) -> int:
    print(f"error: {message}", file=sys.stderr)
    return exit_code


def _validate_cli(args: argparse.Namespace) -> int | None:
    if args.samples <= 0:
        return _die("--samples must be > 0")
    if args.positions_per_sample <= 0:
        return _die("--positions-per-sample must be > 0")
    if args.top_k <= 0:
        return _die("--top-k must be > 0")
    if args.context_tokens < 2:
        return _die("--context-tokens must be >= 2")
    if args.context_tail_chars <= 0:
        return _die("--context-tail-chars must be > 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        return _die(f"--device {args.device!r} requested, but CUDA is unavailable")
    return None


def _checkpoint_candidates(save_dir: Path, hidden_size: int) -> list[Path]:
    pattern = f"pretrain_{hidden_size}*.pth"
    candidates = [
        path
        for path in save_dir.glob(pattern)
        if path.is_file() and not path.name.endswith("_resume.pth") and not path.name.endswith(".tmp")
    ]
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def _resolve_checkpoint(explicit: Path | None, save_dir: str | os.PathLike[str], hidden_size: int) -> Path | None:
    if explicit is not None:
        return explicit.expanduser()
    candidates = _checkpoint_candidates(Path(save_dir).expanduser(), hidden_size)
    return candidates[0] if candidates else None


def _load_torch_file(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]
    if not isinstance(payload, dict) or not all(torch.is_tensor(value) for value in payload.values()):
        raise ValueError("checkpoint is not a plain model state_dict or resume dict with a 'model' state_dict")
    state_dict = dict(payload)
    if state_dict and all(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    return state_dict


def _load_model(checkpoint_path: Path, runtime_args: Any, device: torch.device) -> MiniMindForCausalLM:
    lm_config = build_minimind_config(runtime_args, MiniMindConfig)
    model = MiniMindForCausalLM(lm_config)
    payload = _load_torch_file(checkpoint_path)
    state_dict = _extract_state_dict(payload)
    try:
        load_result = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(f"checkpoint is incompatible with the configured MiniMind architecture: {exc}") from exc

    if load_result.missing_keys:
        print(f"warning: missing checkpoint keys: {len(load_result.missing_keys)}", file=sys.stderr)
    if load_result.unexpected_keys:
        print(f"warning: unexpected checkpoint keys: {len(load_result.unexpected_keys)}", file=sys.stderr)

    model.eval()
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device, dtype=torch.float32)
    return model


def _iter_real_samples(runtime_args: Any, tokenizer: Any, args: argparse.Namespace) -> Iterable[torch.Tensor]:
    shuffled = args.sample_mode == "shuffled"
    dataset = TokenizedParquetDataset(
        runtime_args.data_path,
        max_length=runtime_args.max_seq_len,
        eos_token_id=int(tokenizer.eos_token_id),
        split="train",
        validation_split_ratio=0.0,
        validation_split_seed=int(getattr(runtime_args, "_validation_split_seed", args.seed)),
        shuffle_buffer_size=int(getattr(runtime_args, "shuffle_buffer_size", 0)) if shuffled else 0,
        shuffle_seed=args.seed,
        shuffle_files=bool(getattr(runtime_args, "shuffle_files", False)) if shuffled else False,
    )
    dataset.set_epoch(0)
    for sample in dataset:
        sample = sample[: args.context_tokens].long()
        if sample.numel() >= 2:
            yield sample


def _select_evenly(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    selected: list[int] = []
    for idx in range(count):
        source_idx = round(idx * (len(values) - 1) / (count - 1))
        value = values[source_idx]
        if not selected or selected[-1] != value:
            selected.append(value)
    return selected


def _decode(tokenizer: Any, token_ids: Iterable[int]) -> str:
    ids = [int(token_id) for token_id in token_ids]
    try:
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(ids, skip_special_tokens=False)


def _cell(text: object, max_chars: int | None = None) -> str:
    rendered = str(text)
    if max_chars is not None and len(rendered) > max_chars:
        rendered = rendered[: max(0, max_chars - 3)] + "..."
    return rendered.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t").replace("|", "\\|")


def _token_cell(tokenizer: Any, token_id: int, *, max_chars: int = 32) -> str:
    text = _decode(tokenizer, [token_id])
    rendered = repr(text)[1:-1] if text else "<empty>"
    return f"`{token_id}` {_cell(rendered, max_chars=max_chars)}"


def _topk_cell(tokenizer: Any, token_ids: torch.Tensor, log_probs: torch.Tensor) -> str:
    chunks = []
    for token_id, log_prob in zip(token_ids.tolist(), log_probs.tolist(), strict=True):
        token_text = _decode(tokenizer, [int(token_id)])
        rendered = repr(token_text)[1:-1] if token_text else "<empty>"
        chunks.append(f"`{int(token_id)}`:{_cell(rendered, max_chars=24)}={math.exp(float(log_prob)):.3f}")
    return "<br>".join(chunks)


def _valid_prediction_positions(labels: torch.Tensor, real_token_count: int) -> list[int]:
    candidates = []
    for pos in range(max(0, real_token_count - 1)):
        if int(labels[pos + 1].item()) != -100:
            candidates.append(pos)
    return candidates


def _print_header(config_path: Path, checkpoint_path: Path, runtime_args: Any, device: torch.device, args: argparse.Namespace) -> None:
    print("# MiniMind Next-Token Prediction Eval")
    print()
    print(f"- config: `{config_path}`")
    print(f"- checkpoint: `{checkpoint_path}`")
    print(f"- dataset: `{runtime_args.data_path}`")
    print(f"- device: `{device}`")
    print(f"- samples: `{args.samples}`, positions/sample: `{args.positions_per_sample}`, top-k: `{args.top_k}`")
    print(f"- context tokens: `{args.context_tokens}`, sample mode: `{args.sample_mode}`, seed: `{args.seed}`")
    print()


def _evaluate_sample(
    *,
    sample_index: int,
    sample_tokens: torch.Tensor,
    model: MiniMindForCausalLM,
    tokenizer: Any,
    collator: PretrainDataCollator,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[int, int, int, float]:
    batch = collator([sample_tokens])
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"][0]
    position_ids = batch["position_ids"].to(device)
    attention_mask = batch["attention_mask"]
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_full_logits=True,
        )
    logits = output.logits[0].float().cpu()
    selected_positions = _select_evenly(
        _valid_prediction_positions(labels, int(sample_tokens.numel())),
        args.positions_per_sample,
    )

    print(f"## Sample {sample_index}")
    print()
    print("| pos | context tail | ground truth next | top-1 prediction | top-k predictions | hit |")
    print("| ---: | --- | --- | --- | --- | --- |")

    top1_hits = 0
    topk_hits = 0
    nll_sum = 0.0
    displayed = 0
    for pos in selected_positions:
        target_id = int(labels[pos + 1].item())
        log_probs = torch.log_softmax(logits[pos], dim=-1)
        k = min(args.top_k, log_probs.numel())
        top_log_probs, top_token_ids = torch.topk(log_probs, k=k)
        top1_id = int(top_token_ids[0].item())
        top1_prob = math.exp(float(top_log_probs[0].item()))
        topk_match = target_id in {int(token_id) for token_id in top_token_ids.tolist()}
        top1_match = top1_id == target_id

        context_start = max(0, pos + 1 - 48)
        context_tail = _decode(tokenizer, input_ids[0, context_start : pos + 1].tolist())
        target_nll = -float(log_probs[target_id].item())

        top1_hits += int(top1_match)
        topk_hits += int(topk_match)
        nll_sum += target_nll
        displayed += 1

        print(
            "| "
            f"{pos} | "
            f"{_cell(repr(context_tail)[1:-1], max_chars=args.context_tail_chars)} | "
            f"{_token_cell(tokenizer, target_id)} | "
            f"{_token_cell(tokenizer, top1_id)} ({top1_prob:.3f}) | "
            f"{_topk_cell(tokenizer, top_token_ids, top_log_probs)} | "
            f"{'top1' if top1_match else ('topk' if topk_match else '-')}"
            " |"
        )

    print()
    return displayed, top1_hits, topk_hits, nll_sum


def main() -> int:
    args = parse_args()
    validation_error = _validate_cli(args)
    if validation_error is not None:
        return validation_error
    if not args.config.is_file():
        return _die(f"config not found: {args.config}")

    runtime_args = runtime_args_from_toml(args.config, cuda_available=torch.cuda.is_available())
    if args.context_tokens >= int(runtime_args.max_seq_len):
        return _die(f"--context-tokens must be < configured max_seq_len ({runtime_args.max_seq_len})")
    if not is_tokenized_parquet_dataset(runtime_args.data_path):
        return _die(f"data_path is not a tokenized parquet dataset: {runtime_args.data_path}")

    checkpoint_path = _resolve_checkpoint(args.checkpoint, runtime_args.save_dir, int(runtime_args.hidden_size))
    if checkpoint_path is None or not checkpoint_path.is_file():
        return _die(
            "no compatible checkpoint found yet. "
            f"Expected a non-resume pretrain_{runtime_args.hidden_size}*.pth file in {runtime_args.save_dir}; "
            "run this after the first saved checkpoint."
        )

    device = torch.device(args.device)
    tokenizer = load_pretrain_tokenizer(runtime_args.tokenizer_path)
    collator = PretrainDataCollator(
        eos_token_id=int(tokenizer.eos_token_id),
        pad_token_id=int(tokenizer.pad_token_id),
        max_seq_len=args.context_tokens + 1,
    )
    model = _load_model(checkpoint_path, runtime_args, device)

    _print_header(args.config, checkpoint_path, runtime_args, device, args)

    total_displayed = 0
    total_top1_hits = 0
    total_topk_hits = 0
    total_nll = 0.0
    samples_seen = 0

    for samples_seen, sample_tokens in enumerate(_iter_real_samples(runtime_args, tokenizer, args), start=1):
        displayed, top1_hits, topk_hits, nll_sum = _evaluate_sample(
            sample_index=samples_seen,
            sample_tokens=sample_tokens,
            model=model,
            tokenizer=tokenizer,
            collator=collator,
            device=device,
            args=args,
        )
        total_displayed += displayed
        total_top1_hits += top1_hits
        total_topk_hits += topk_hits
        total_nll += nll_sum
        if samples_seen >= args.samples:
            break

    if samples_seen == 0 or total_displayed == 0:
        return _die("no evaluable samples with at least two real tokens were found")

    mean_nll = total_nll / total_displayed
    perplexity = math.exp(mean_nll) if mean_nll < math.log(sys.float_info.max) else float("inf")
    print("## Aggregate")
    print()
    print(f"- displayed positions: `{total_displayed}`")
    print(f"- top-1 accuracy: `{total_top1_hits / total_displayed:.4f}` ({total_top1_hits}/{total_displayed})")
    print(f"- top-{args.top_k} accuracy: `{total_topk_hits / total_displayed:.4f}` ({total_topk_hits}/{total_displayed})")
    print(f"- mean NLL: `{mean_nll:.4f}`")
    print(f"- approximate perplexity: `{perplexity:.4f}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
