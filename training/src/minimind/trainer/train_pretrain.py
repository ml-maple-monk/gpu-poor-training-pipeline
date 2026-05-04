import glob
import importlib
import importlib.util
import math
import os
import signal
import sys
import time
from pathlib import Path

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel

from model.model_minimind import MiniMindConfig
from trainer._benchmark_metrics import (
    NvmlEnergyMeter,
    count_valid_tokens,
    ddp_sum,
    dense_model_flops_per_step,
    maybe_record_time_to_target,
    resolve_peak_flops_profile,
    should_log_dense_flops,
    world_size,
)
from trainer.mlflow_logger import MlflowLogger
from trainer.pretrain_config import (
    apply_dataset_environment,
    build_minimind_config,
    runtime_args_from_toml,
)
from trainer.pretrain_data import build_pretrain_data_pipeline
from trainer.trainer_utils import (
    Logger,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    log_flash_attention_status,
    setup_seed,
)
from trainer.trainer_utils import (
    atomic_torch_save as _atomic_torch_save,
)
from trainer.trainer_utils import (
    build_autocast_context as _build_autocast_context,
)
from trainer.trainer_utils import (
    build_grad_scaler as _build_grad_scaler,
)
from trainer.trainer_utils import (
    current_mlflow_step as _current_mlflow_step,
)
from trainer.trainer_utils import (
    validation_ppl_from_loss as _validation_ppl_from_loss,
)

mlflow_logger = MlflowLogger()
_MUON_STEP_PARAM = None


def _sigterm_handler(signum, frame):
    print("[SIGTERM] Received SIGTERM — shutting down gracefully", flush=True)

    save_dir = getattr(globals().get("args", None), "save_dir", "")
    if save_dir:
        for tmp_file in glob.glob(os.path.join(save_dir, "*.tmp")):
            try:
                os.remove(tmp_file)
            except OSError:
                pass

    try:
        mlflow_logger.finish(status="KILLED")
    except Exception as exc:
        print(f"[SIGTERM] Warning: MLflow finish failed: {exc}", flush=True)

    sys.exit(143)


def _reset_metric_window(state: dict[str, float | int | None]) -> None:
    state["window_start_time"] = time.perf_counter()
    state["window_loss_sum_local"] = 0.0
    state["window_logits_loss_sum_local"] = 0.0
    state["window_aux_loss_sum_local"] = 0.0
    state["window_tokens_local"] = 0
    state["window_sequences_local"] = 0
    state["window_optimizer_steps"] = 0
    state["window_grad_norm_sum"] = 0.0
    state["window_grad_norm_max"] = None
    state["window_grad_norm_count"] = 0
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)


def _build_metric_state(
    start_optimizer_step: int,
    *,
    resolved_peak_tflops_per_gpu: float | None,
    resolved_peak_fp8_tflops_per_gpu: float | None,
) -> dict[str, float | int | None | dict[str, float]]:
    state: dict[str, float | int | None | dict[str, float]] = {
        "job_start_time": time.perf_counter(),
        "consumed_tokens_local_total": 0,
        "optimizer_step": start_optimizer_step,
        "last_validation_update_step": -1,
        "time_to_target_hit": None,
        "resolved_peak_tflops_per_gpu": resolved_peak_tflops_per_gpu,
        "resolved_peak_fp8_tflops_per_gpu": resolved_peak_fp8_tflops_per_gpu,
    }
    _reset_metric_window(state)
    return state


def _save_checkpoint(epoch: int, step: int) -> None:
    model.eval()
    moe_suffix = "_moe" if lm_config.use_moe else ""
    ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    state_dict = raw_model.state_dict()
    _atomic_torch_save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
    lm_checkpoint(
        lm_config,
        weight=args.save_weight,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch,
        step=step,
        optimizer_step=int(metric_state["optimizer_step"]),
        save_dir=args.save_dir,
    )
    mlflow_logger.log_checkpoint(ckp, step)
    model.train()
    del state_dict


def _target_update_steps(iters: int) -> int:
    if args.max_steps > 0:
        return int(args.max_steps)
    return args.epochs * math.ceil(iters / max(args.accumulation_steps, 1))


def _target_reached() -> bool:
    return args.max_steps > 0 and int(metric_state["optimizer_step"]) >= int(args.max_steps)


def _load_muon_step_param():
    global _MUON_STEP_PARAM

    if _MUON_STEP_PARAM is not None:
        return _MUON_STEP_PARAM
    try:
        muon_module = importlib.import_module("architecture_optimisation_zoo.components.8bit_muon")
    except (ModuleNotFoundError, ImportError):
        module_path = Path(
            "/home/geeyang/workspace/architecture-optimisation-zoo/src/"
            "architecture_optimisation_zoo/components/8bit_muon.py"
        )
        if not module_path.is_file():
            raise ModuleNotFoundError(
                "optimizer='muon8bit' requires the local architecture-optimisation-zoo package. "
                "Install it with: python3 -m pip install --user --ignore-requires-python --no-deps "
                "-e /home/geeyang/workspace/architecture-optimisation-zoo"
            )
        spec = importlib.util.spec_from_file_location("gpupoor_aozoo_8bit_muon", module_path)
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"Unable to load Muon8Bit implementation from {module_path}")
        muon_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = muon_module
        spec.loader.exec_module(muon_module)
    _MUON_STEP_PARAM = getattr(muon_module, "_muon_step_param")
    return _MUON_STEP_PARAM


def _resolve_model_parameter_dtype(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported model parameter dtype: {dtype_name}")


def _is_fp8_eligible_linear(name: str, child: torch.nn.Module) -> bool:
    if not isinstance(child, torch.nn.Linear):
        return False
    if "embed_tokens" in name or "lm_head" in name:
        return False
    if getattr(child, "bias", None) is not None:
        return False
    return child.in_features % 16 == 0 and child.out_features % 16 == 0


def _maybe_apply_fp8_training(module: torch.nn.Module) -> None:
    if getattr(args, "precision", "bf16_training") != "fp8_training":
        setattr(module, "gpupoor_precision_split", {"precision": getattr(args, "precision", "bf16_training")})
        return
    if device_type != "cuda":
        raise RuntimeError("precision='fp8_training' requires CUDA")

    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as exc:
        raise ModuleNotFoundError(
            "precision='fp8_training' requires torchao.float8. Install torchao before launching this recipe."
        ) from exc

    selected: list[str] = []
    skipped: list[str] = []

    def module_filter_fn(child, name: str) -> bool:
        if _is_fp8_eligible_linear(name, child):
            selected.append(name)
            return True
        if isinstance(child, torch.nn.Linear):
            skipped.append(name)
        return False

    model_dtype = _resolve_model_parameter_dtype(args.dtype)
    module.to(dtype=model_dtype)
    config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
    convert_to_float8_training(module, config=config, module_filter_fn=module_filter_fn)

    split = {
        "precision": "fp8_training",
        "fp8_recipe": args.fp8_recipe,
        "model_parameter_dtype": str(model_dtype).removeprefix("torch."),
        "fp8_linears": len(selected),
        "skipped_linears": len(skipped),
        "selected_linears": tuple(sorted(selected)),
        "skipped_linear_names": tuple(sorted(skipped)),
        "backend": "torchao_float8_tensorwise_dynamic",
    }
    setattr(module, "gpupoor_precision_split", split)


def _is_muon_excluded_parameter(name: str) -> bool:
    return name.startswith(("model.embed_tokens.", "lm_head."))


def _split_muon8bit_parameters(module: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], dict]:
    muon_params: list[torch.nn.Parameter] = []
    aux_params: list[torch.nn.Parameter] = []
    muon_names: list[str] = []
    aux_names: list[str] = []
    seen: set[int] = set()
    for name, parameter in module.named_parameters():
        parameter_id = id(parameter)
        if parameter_id in seen or not parameter.requires_grad:
            continue
        seen.add(parameter_id)
        clean_name = name.removeprefix("_orig_mod.")
        if parameter.ndim == 2 and not _is_muon_excluded_parameter(clean_name):
            muon_params.append(parameter)
            muon_names.append(clean_name)
        else:
            aux_params.append(parameter)
            aux_names.append(clean_name)

    split = {
        "muon_param_count": int(sum(parameter.numel() for parameter in muon_params)),
        "muon_param_tensors": len(muon_params),
        "muon_param_names": muon_names,
        "sgd_aux_param_count": int(sum(parameter.numel() for parameter in aux_params)),
        "sgd_aux_param_tensors": len(aux_params),
        "sgd_aux_param_names": aux_names,
        "adamw_fallback": False,
        "excluded_reason": "Muon8Bit is matrix-only; tied vocab, norms, biases, and non-2D tensors use plain SGD.",
    }
    return muon_params, aux_params, split


class Muon8BitWithSgdAux(optim.Optimizer):
    """Single optimizer using 8-bit Muon for matrices and SGD for unsupported tensors."""

    def __init__(self, param_groups):
        _load_muon_step_param()
        normalized = []
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Each muon8bit param group must include a boolean 'use_muon' flag.")
            group = dict(group)
            if group["use_muon"]:
                group.setdefault("lr", 1e-3)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
                group.setdefault("block_size", 256)
                group.setdefault("quantize_state", True)
                group.setdefault("scale_dtype", torch.float16)
            else:
                group.setdefault("lr", 1e-3)
                group.setdefault("weight_decay", 0.0)
            normalized.append(group)
        super().__init__(normalized, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_step_param = _load_muon_step_param()
        for group in self.param_groups:
            if group["use_muon"]:
                for parameter in group["params"]:
                    muon_step_param(
                        parameter,
                        self.state[parameter],
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        momentum=group["momentum"],
                        nesterov=group["nesterov"],
                        ns_steps=group["ns_steps"],
                        quantize_state=group["quantize_state"],
                        block_size=group["block_size"],
                        scale_dtype=group["scale_dtype"],
                    )
                continue

            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(grad.to(dtype=parameter.dtype), alpha=-lr)

        return loss


def _build_muon8bit_optimizer(module: torch.nn.Module):
    muon_params, aux_params, split = _split_muon8bit_parameters(module)
    if not muon_params:
        raise ValueError("optimizer='muon8bit' found no eligible 2D non-vocab parameters")
    setattr(module, "gpupoor_optimizer_split", split)
    param_groups = [
        {
            "params": muon_params,
            "use_muon": True,
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
            "block_size": 256,
            "quantize_state": True,
            "scale_dtype": torch.float16,
        },
    ]
    if aux_params:
        param_groups.append(
            {
                "params": aux_params,
                "use_muon": False,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
    return Muon8BitWithSgdAux(param_groups)


def _build_optimizer(module: torch.nn.Module):
    model_parameters = module.parameters()
    if args.optimizer == "muon8bit":
        return _build_muon8bit_optimizer(module)
    if args.optimizer == "adamw":
        return optim.AdamW(model_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return optim.SGD(model_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def _log_train_window(epoch: int, step: int, iters: int, start_step: int) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize(args.device)

    global_tokens = ddp_sum(metric_state["window_tokens_local"], collective_device)
    current_lr = optimizer.param_groups[-1]["lr"]
    elapsed_window = time.perf_counter() - float(metric_state["window_start_time"])
    spend_time = time.time() - epoch_start_time
    eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

    if global_tokens <= 0:
        _reset_metric_window(metric_state)
        return

    global_loss_sum = ddp_sum(metric_state["window_loss_sum_local"], collective_device)
    global_logits_loss_sum = ddp_sum(metric_state["window_logits_loss_sum_local"], collective_device)
    global_aux_loss_sum = ddp_sum(metric_state["window_aux_loss_sum_local"], collective_device)
    consumed_tokens = ddp_sum(metric_state["consumed_tokens_local_total"], collective_device)

    current_loss = global_loss_sum / global_tokens
    current_logits_loss = global_logits_loss_sum / global_tokens
    current_aux_loss = global_aux_loss_sum / global_tokens

    extra_metrics = {
        "train/update_step": float(metric_state["optimizer_step"]),
    }
    if metric_state["resolved_peak_tflops_per_gpu"] is not None:
        extra_metrics["train/peak_tflops_per_gpu"] = float(metric_state["resolved_peak_tflops_per_gpu"])
    if metric_state["resolved_peak_fp8_tflops_per_gpu"] is not None:
        extra_metrics["train/peak_fp8_tflops_per_gpu"] = float(metric_state["resolved_peak_fp8_tflops_per_gpu"])
    summary = [
        f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters})",
        f"loss: {current_loss:.4f}",
        f"logits_loss: {current_logits_loss:.4f}",
        f"aux_loss: {current_aux_loss:.4f}",
        f"lr: {current_lr:.8f}",
        f"epoch_time: {eta_min:.1f}min",
        f"consumed_tokens: {int(consumed_tokens)}",
    ]

    if metric_state["window_optimizer_steps"] > 0 and elapsed_window > 0:
        step_time_s = elapsed_window / float(metric_state["window_optimizer_steps"])
        tokens_per_sec_per_gpu = global_tokens / elapsed_window / world_size()
        extra_metrics["train/step_time_s"] = step_time_s
        extra_metrics["train/tokens_per_sec_per_gpu"] = tokens_per_sec_per_gpu
        summary.append(f"tok/s/gpu: {tokens_per_sec_per_gpu:.2f}")

        if should_log_dense_flops(use_moe=bool(args.use_moe), peak_tflops_per_gpu=args.peak_tflops_per_gpu):
            global_sequences = ddp_sum(metric_state["window_sequences_local"], collective_device)
            avg_global_batch_seqs = global_sequences / float(metric_state["window_optimizer_steps"])
            model_flops = dense_model_flops_per_step(
                global_batch_seqs=avg_global_batch_seqs,
                seq_len=args.max_seq_len,
                num_layers=args.num_hidden_layers,
                hidden_size=args.hidden_size,
                vocab_size=lm_config.vocab_size,
            )
            model_tflops_per_gpu = model_flops / max(step_time_s, 1e-12) / world_size() / 1e12
            extra_metrics["train/model_tflops_per_gpu"] = model_tflops_per_gpu
            extra_metrics["train/mfu"] = model_tflops_per_gpu / args.peak_tflops_per_gpu

    grad_norm_count = int(metric_state["window_grad_norm_count"])
    if grad_norm_count > 0:
        avg_grad_norm = float(metric_state["window_grad_norm_sum"]) / grad_norm_count
        max_grad_norm = float(metric_state["window_grad_norm_max"])
        extra_metrics["train/grad_norm"] = avg_grad_norm
        extra_metrics["train/grad_norm_max"] = max_grad_norm
        summary.append(f"grad_norm: {avg_grad_norm:.4f}")

    if device_type == "cuda":
        extra_metrics["train/peak_allocated_gb"] = torch.cuda.max_memory_allocated(args.device) / 1e9
        extra_metrics["train/peak_reserved_gb"] = torch.cuda.max_memory_reserved(args.device) / 1e9

    local_energy_j = energy_meter.joules_since_start()
    if local_energy_j is not None:
        total_energy_j = ddp_sum(local_energy_j, collective_device)
        extra_metrics["train/total_energy_j"] = total_energy_j
        if consumed_tokens > 0:
            extra_metrics["train/joules_per_token"] = total_energy_j / consumed_tokens

    Logger(", ".join(summary))
    if is_main_process():
        mlflow_logger.log_step(
            step=_current_mlflow_step(epoch, step, iters),
            epoch=epoch + 1,
            loss=current_loss,
            logits_loss=current_logits_loss,
            aux_loss=current_aux_loss,
            lr=current_lr,
            tokens_seen=consumed_tokens,
            update_step=metric_state["optimizer_step"],
            extra_metrics=extra_metrics,
        )

    _reset_metric_window(metric_state)


def _run_validation(epoch: int, step: int, iters: int, val_loader) -> None:
    model.eval()
    val_loss_sum_local = 0.0
    val_tokens_local = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)
            position_ids = batch["position_ids"].to(args.device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            attention_mask = attention_mask.to(args.device, non_blocking=True) if attention_mask is not None else None
            with autocast_ctx:
                res = model(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    labels=labels,
                    return_full_logits=False,
                )
            valid_tokens = count_valid_tokens(labels)
            if valid_tokens > 0:
                val_loss_sum_local += float(res.loss.detach().float().item()) * valid_tokens
                val_tokens_local += valid_tokens
            del input_ids, labels, position_ids, attention_mask, res

    global_tokens = ddp_sum(val_tokens_local, collective_device)
    if global_tokens <= 0:
        model.train()
        return

    global_loss_sum = ddp_sum(val_loss_sum_local, collective_device)
    val_loss = global_loss_sum / global_tokens
    val_ppl = _validation_ppl_from_loss(val_loss)
    mlflow_step = _current_mlflow_step(epoch, step, iters)
    Logger(f"Validation(update={int(metric_state['optimizer_step'])}): loss={val_loss:.4f}, ppl={val_ppl:.4f}")

    if is_main_process():
        mlflow_logger.log_metrics(step=mlflow_step, metrics={"val/loss": val_loss, "val/ppl": val_ppl})

    time_to_target_hit = maybe_record_time_to_target(
        hit=metric_state["time_to_target_hit"],
        metric_name=args.time_to_target_metric,
        current_value=val_loss if args.time_to_target_metric == "val_loss" else val_ppl,
        target_value=args.time_to_target_value,
        consumed_tokens=ddp_sum(metric_state["consumed_tokens_local_total"], collective_device),
        wallclock_s=time.perf_counter() - float(metric_state["job_start_time"]),
    )
    if time_to_target_hit is not None and metric_state["time_to_target_hit"] is None:
        metric_state["time_to_target_hit"] = time_to_target_hit
        Logger(
            f"Reached {args.time_to_target_metric} target in {time_to_target_hit['wallclock_s']:.2f}s "
            f"at {int(time_to_target_hit['consumed_tokens'])} consumed tokens"
        )
        if is_main_process():
            mlflow_logger.log_metrics(
                step=mlflow_step,
                metrics={
                    "target/wallclock_s": time_to_target_hit["wallclock_s"],
                    "target/consumed_tokens": time_to_target_hit["consumed_tokens"],
                    "target/current_value": time_to_target_hit["current_value"],
                },
            )

    model.train()


def _maybe_run_validation(epoch: int, step: int, iters: int, val_loader, *, force: bool = False) -> None:
    if val_loader is None or args.validation_interval_steps <= 0 or metric_state["optimizer_step"] <= 0:
        return

    update_step = int(metric_state["optimizer_step"])
    due = force or (update_step % args.validation_interval_steps == 0)
    if not due or update_step == metric_state["last_validation_update_step"]:
        return

    metric_state["last_validation_update_step"] = update_step
    _run_validation(epoch, step, iters, val_loader)


def train_epoch(epoch, loader, iters, start_step=0, val_loader=None):
    global epoch_start_time

    epoch_start_time = time.time()
    last_step = start_step
    total_update_steps = _target_update_steps(iters)
    for step, batch in enumerate(loader, start=start_step + 1):
        if _target_reached():
            break
        input_ids = batch["input_ids"].to(args.device, non_blocking=True)
        labels = batch["labels"].to(args.device, non_blocking=True)
        position_ids = batch["position_ids"].to(args.device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        attention_mask = attention_mask.to(args.device, non_blocking=True) if attention_mask is not None else None
        last_step = step

        with autocast_ctx:
            res = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                return_full_logits=False,
            )
            aux_loss = res.aux_loss if res.aux_loss is not None else res.loss.new_zeros(())
            loss = (res.loss + aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        valid_tokens = count_valid_tokens(labels)
        logits_loss_value = float(res.loss.detach().float().item())
        aux_loss_value = float(aux_loss.detach().float().item())
        total_loss_value = logits_loss_value + aux_loss_value

        metric_state["window_logits_loss_sum_local"] += logits_loss_value * valid_tokens
        metric_state["window_aux_loss_sum_local"] += aux_loss_value * valid_tokens
        metric_state["window_loss_sum_local"] += total_loss_value * valid_tokens
        metric_state["window_tokens_local"] += valid_tokens
        metric_state["window_sequences_local"] += int(input_ids.size(0))
        metric_state["consumed_tokens_local_total"] += valid_tokens

        update_due = step % args.accumulation_steps == 0 or step == iters
        if update_due:
            next_update_step = int(metric_state["optimizer_step"]) + 1
            lr = get_lr(
                next_update_step,
                total_update_steps,
                args.learning_rate,
                schedule=args.lr_schedule,
                warmup_steps=args.lr_warmup_steps,
                min_lr_ratio=args.lr_min_ratio,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_norm_value = float(grad_norm.detach().float().item() if torch.is_tensor(grad_norm) else grad_norm)
            metric_state["window_grad_norm_sum"] += grad_norm_value
            metric_state["window_grad_norm_count"] += 1
            grad_norm_max = metric_state["window_grad_norm_max"]
            metric_state["window_grad_norm_max"] = (
                grad_norm_value if grad_norm_max is None else max(float(grad_norm_max), grad_norm_value)
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            metric_state["optimizer_step"] += 1
            metric_state["window_optimizer_steps"] += 1

        reached_target = _target_reached()

        if step % args.log_interval == 0 or step == iters or reached_target:
            _log_train_window(epoch, step, iters, start_step)

        save_due = update_due and int(metric_state["optimizer_step"]) % args.save_interval == 0
        if (save_due or step == iters or reached_target) and is_main_process():
            _save_checkpoint(epoch, step)

        if update_due:
            _maybe_run_validation(epoch, step, iters, val_loader, force=(step == iters or reached_target))

        del input_ids, labels, position_ids, attention_mask, res, loss
        if reached_target:
            break

    return last_step


def run_training(runtime_args):
    global \
        args, \
        autocast_ctx, \
        collective_device, \
        device_type, \
        energy_meter, \
        lm_config, \
        metric_state, \
        model, \
        optimizer, \
        scaler

    args = runtime_args
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = build_minimind_config(args, MiniMindConfig)
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.save_dir) if args.from_resume == 1 else None
    )

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    collective_device = torch.device(args.device if device_type == "cuda" else "cpu")
    autocast_ctx = _build_autocast_context(device_type, args.dtype)
    energy_meter = NvmlEnergyMeter(torch.cuda.current_device() if device_type == "cuda" else 0)

    resolved_peak_profile = None
    if device_type == "cuda":
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        resolved_peak_profile = resolve_peak_flops_profile(gpu_name, getattr(args, "_gpu_profiles", None))
        if resolved_peak_profile is not None:
            args.peak_fp8_tflops_per_gpu = resolved_peak_profile.fp8_tflops_per_gpu
            if args.peak_tflops_per_gpu is None:
                args.peak_tflops_per_gpu = resolved_peak_profile.training_tflops_per_gpu
            if is_main_process():
                fp8_summary = (
                    f", fp8_peak={resolved_peak_profile.fp8_tflops_per_gpu:.2f}"
                    if resolved_peak_profile.fp8_tflops_per_gpu is not None
                    else ""
                )
                Logger(
                    f"Auto-detected GPU peak flops from '{gpu_name}': "
                    f"train_peak={resolved_peak_profile.training_tflops_per_gpu:.2f}{fp8_summary}"
                )
        elif is_main_process():
            Logger(f"Peak TFLOPs auto-detect unavailable for GPU '{gpu_name}'")

    log_flash_attention_status(requested=bool(args.flash_attn), device_type_name=device_type, logger=Logger)

    # ========== 4. MLflow (no-op unless MLFLOW_TRACKING_URI is set) ==========
    if is_main_process():
        mlflow_logger.start(args, lm_config, getattr(args, "_mlflow_config", {}))

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(
        lm_config,
        args.from_weight,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
    )
    _maybe_apply_fp8_training(model)
    if is_main_process():
        precision_split = getattr(model, "gpupoor_precision_split", {})
        Logger(
            "Precision path: "
            f"{precision_split.get('precision', getattr(args, 'precision', 'bf16_training'))}"
            f" (architecture_variant={getattr(args, 'architecture_variant', '')})"
        )
        if precision_split.get("precision") == "fp8_training":
            Logger(
                "FP8 conversion: "
                f"backend={precision_split['backend']}; "
                f"recipe={precision_split['fp8_recipe']}; "
                f"linears={precision_split['fp8_linears']}; "
                f"skipped_linears={precision_split['skipped_linears']}"
            )
            mlflow_logger.log_params(
                {
                    "precision.name": precision_split["precision"],
                    "precision.backend": precision_split["backend"],
                    "precision.fp8_recipe": precision_split["fp8_recipe"],
                    "precision.model_parameter_dtype": precision_split["model_parameter_dtype"],
                    "precision.fp8_linears": precision_split["fp8_linears"],
                    "precision.skipped_linears": precision_split["skipped_linears"],
                    "precision.skipped_linear_names": precision_split["skipped_linear_names"],
                    "architecture.variant": getattr(args, "architecture_variant", ""),
                }
            )
    data_pipeline = build_pretrain_data_pipeline(args, tokenizer)
    val_loader = data_pipeline.val_loader
    scaler = _build_grad_scaler(device_type, args.dtype)
    optimizer = _build_optimizer(model)
    if is_main_process():
        Logger(f"Using optimizer: {args.optimizer}")
        optimizer_split = getattr(model, "gpupoor_optimizer_split", None)
        if optimizer_split:
            Logger(
                "Optimizer split: "
                f"Muon8Bit tensors={optimizer_split['muon_param_tensors']} "
                f"params={optimizer_split['muon_param_count']}; "
                f"SGD auxiliary tensors={optimizer_split['sgd_aux_param_tensors']} "
                f"params={optimizer_split['sgd_aux_param_count']}; "
                f"weight_decay={args.weight_decay}; AdamW fallback=false"
            )
            mlflow_logger.log_params(
                {
                    "optimizer.name": args.optimizer,
                    "optimizer.weight_decay": args.weight_decay,
                    "optimizer.muon_param_count": optimizer_split["muon_param_count"],
                    "optimizer.muon_param_tensors": optimizer_split["muon_param_tensors"],
                    "optimizer.sgd_aux_param_count": optimizer_split["sgd_aux_param_count"],
                    "optimizer.sgd_aux_param_tensors": optimizer_split["sgd_aux_param_tensors"],
                    "optimizer.adamw_fallback": False,
                    "optimizer.excluded_reason": optimizer_split["excluded_reason"],
                }
            )

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    start_optimizer_step = 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
        start_optimizer_step = ckp_data.get("optimizer_step", start_step // max(args.accumulation_steps, 1))

    metric_state = _build_metric_state(
        start_optimizer_step,
        resolved_peak_tflops_per_gpu=args.peak_tflops_per_gpu,
        resolved_peak_fp8_tflops_per_gpu=args.peak_fp8_tflops_per_gpu,
    )

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        compile_kwargs = {"fullgraph": True} if getattr(args, "compile_fullgraph", 0) else {}
        model = torch.compile(model, **compile_kwargs)
        Logger(f"torch.compile enabled (fullgraph={bool(getattr(args, 'compile_fullgraph', 0))})")
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    epoch = start_epoch
    while epoch < args.epochs or args.max_steps > 0:
        if _target_reached():
            break
        setup_seed(42 + epoch)
        indices = data_pipeline.epoch_indices(epoch)
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        loader = data_pipeline.train_loader(indices, skip_batches=skip)
        if skip > 0:
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始")
            train_epoch(epoch, loader, len(loader) + skip, start_step, val_loader=val_loader)
        else:
            train_epoch(epoch, loader, len(loader), 0, val_loader=val_loader)
        start_step = 0
        epoch += 1

    # ========== 9. 清理分布进程 ==========
    if is_main_process():
        mlflow_logger.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    """MiniMind pretraining entrypoint — reads config from a TOML file."""
    if len(sys.argv) != 2:
        print("usage: train_pretrain.py <config.toml>", file=sys.stderr)
        raise SystemExit(2)

    runtime_args = runtime_args_from_toml(sys.argv[1], cuda_available=torch.cuda.is_available())
    apply_dataset_environment(runtime_args)
    try:
        run_training(runtime_args)
    except Exception:
        if is_main_process():
            mlflow_logger.finish(status="FAILED")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise


if __name__ == "__main__":
    main()
