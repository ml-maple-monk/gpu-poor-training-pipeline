"""
训练工具函数集合
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from model.model_minimind import MiniMindForCausalLM


def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, "n_routed_experts", getattr(config, "num_experts", 0))
    n_active = getattr(config, "num_experts_per_tok", 0)
    n_shared = getattr(config, "n_shared_experts", 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0." in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.shared_experts.0." in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total:
        Logger(f"Model Params: {total:.2f}M-A{active:.2f}M")
    else:
        Logger(f"Model Params: {total:.2f}M")


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


# doc-anchor: atomic-save-sigterm
def atomic_torch_save(obj, path):
    """Save checkpoints via a .tmp file so partial writes do not replace good weights."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def current_mlflow_step(epoch: int, step: int, iters: int) -> int:
    return epoch * iters + step


def validation_ppl_from_loss(val_loss: float) -> float:
    if math.isnan(val_loss):
        return float("nan")
    if math.isinf(val_loss):
        return float("inf")
    max_log = math.log(sys.float_info.max)
    if val_loss > max_log:
        return float("inf")
    return math.exp(val_loss)


def _resolve_autocast_dtype(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return None
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def build_autocast_context(device_type_name: str, dtype_name: str):
    if device_type_name != "cuda":
        return nullcontext()
    autocast_dtype = _resolve_autocast_dtype(dtype_name)
    if autocast_dtype is None:
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=autocast_dtype)


def build_grad_scaler(device_type_name: str, dtype_name: str):
    scaler_device = "cuda" if device_type_name == "cuda" else "cpu"
    scaler_enabled = device_type_name == "cuda" and dtype_name == "float16"
    return torch.amp.GradScaler(scaler_device, enabled=scaler_enabled)


def log_flash_attention_status(*, requested: bool, device_type_name: str, logger=Logger) -> None:
    has_sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    if not requested:
        logger("Flash attention disabled by config")
        return

    if device_type_name != "cuda":
        logger("Flash attention requested, but CUDA is unavailable; training will use the fallback attention path")
        return

    is_available = True
    if hasattr(torch.backends.cuda, "is_flash_attention_available"):
        is_available = bool(torch.backends.cuda.is_flash_attention_available())

    flash_sdp_enabled = True
    if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
        flash_sdp_enabled = bool(torch.backends.cuda.flash_sdp_enabled())

    if has_sdpa and is_available and flash_sdp_enabled:
        logger("Flash attention check passed: PyTorch flash attention is available")
        return

    logger(
        "Flash attention requested, but PyTorch flash attention is unavailable "
        f"(sdpa={has_sdpa}, available={is_available}, flash_sdp_enabled={flash_sdp_enabled}); "
        "training will use the fallback attention path"
    )


def get_lr(current_step, total_steps, lr, *, schedule="cosine", warmup_steps=0, min_lr_ratio=0.1):
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be >= 0.0 and <= 1.0")

    step = max(0, min(current_step, total_steps))
    if warmup_steps > 0 and step <= warmup_steps:
        return lr * (step / warmup_steps)

    if schedule == "constant":
        return lr
    if schedule != "cosine":
        raise ValueError(f"Unsupported lr schedule: {schedule}")

    if total_steps <= warmup_steps:
        return lr

    decay_progress = (step - warmup_steps) / (total_steps - warmup_steps)
    decay_progress = max(0.0, min(decay_progress, 1.0))
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lm_checkpoint(
    lm_config,
    weight="full_sft",
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    save_dir="../checkpoints",
    **kwargs,
):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = "_moe" if lm_config.use_moe else ""
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + ".tmp"
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        resume_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, "_orig_mod", raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location="cpu")
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}")
            return ckp_data
        return None


def init_model(lm_config, from_weight="pretrain", tokenizer_path="../model", save_dir="../out", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight != "none":
        moe_suffix = "_moe" if lm_config.use_moe else ""
        weight_path = f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


def build_packed_batches(indices, sample_lengths, packed_batch_size, max_seq_len, skip_batches=0, drop_last=False):
    if packed_batch_size <= 0:
        raise ValueError("packed_batch_size must be > 0")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0")

    batches = []
    current_batch = []
    current_row_tokens = 0
    current_row_count = 0
    iterator = tqdm(
        indices,
        total=len(indices) if hasattr(indices, "__len__") else None,
        desc="Building packed batches",
        unit="sample",
        disable=not is_main_process(),
        leave=False,
    )

    for idx in iterator:
        sample_length = min(int(sample_lengths[idx]), max_seq_len)
        next_row_tokens = current_row_tokens
        next_row_count = current_row_count

        if next_row_count == 0:
            next_row_count = 1
            next_row_tokens = sample_length
        elif next_row_tokens + sample_length <= max_seq_len:
            next_row_tokens += sample_length
        else:
            next_row_count += 1
            next_row_tokens = sample_length

        if current_batch and next_row_count > packed_batch_size:
            batches.append(current_batch)
            current_batch = [idx]
            current_row_tokens = sample_length
            current_row_count = 1
            continue

        current_batch.append(idx)
        current_row_tokens = next_row_tokens
        current_row_count = next_row_count

    if current_batch and (not drop_last or current_row_count == packed_batch_size):
        batches.append(current_batch)

    return batches[skip_batches:] if skip_batches > 0 else batches
