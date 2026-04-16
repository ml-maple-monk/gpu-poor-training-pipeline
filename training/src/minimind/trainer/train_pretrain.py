import glob
import math
import os
import signal
import sys
import time
from types import SimpleNamespace

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import click
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dataset.lm_dataset import PretrainDataCollator, PretrainDataset, pretokenized_sample_count
from model.model_minimind import MiniMindConfig
from trainer._benchmark_metrics import (
    NvmlEnergyMeter,
    count_valid_tokens,
    ddp_sum,
    dense_model_flops_per_step,
    dist_ready,
    maybe_record_time_to_target,
    resolve_peak_flops_profile,
    should_log_dense_flops,
    split_validation_indices,
    world_size,
)
from trainer._mlflow_helper import finish as _mlflow_finish
from trainer._mlflow_helper import log_checkpoint as _mlflow_log_ckpt
from trainer._mlflow_helper import log_metrics as _mlflow_log_metrics
from trainer._mlflow_helper import log_step as _mlflow_log_step
from trainer._mlflow_helper import start as _mlflow_start
from trainer.trainer_utils import (
    Logger,
    atomic_torch_save as _atomic_torch_save,
    build_autocast_context as _build_autocast_context,
    build_packed_batches,
    build_grad_scaler as _build_grad_scaler,
    current_mlflow_step as _current_mlflow_step,
    get_lr,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    log_flash_attention_status,
    setup_seed,
    validation_ppl_from_loss as _validation_ppl_from_loss,
)

VALIDATION_SPLIT_SEED = 42


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
        _mlflow_finish(status="KILLED")
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

def _build_pretrain_datasets(tokenizer):
    del tokenizer
    sample_count = pretokenized_sample_count(args.data_path)
    train_indices = None
    val_ds = None

    validation_requested = args.validation_split_ratio > 0.0 or args.validation_interval_steps > 0
    validation_enabled = args.validation_split_ratio > 0.0 and args.validation_interval_steps > 0

    if validation_enabled:
        if sample_count < 2:
            if is_main_process():
                Logger("Validation disabled: dataset has fewer than 2 samples after loading")
        else:
            train_indices, val_indices = split_validation_indices(
                sample_count, args.validation_split_ratio, seed=VALIDATION_SPLIT_SEED
            )
            if dist_ready():
                val_indices = val_indices[dist.get_rank() :: world_size()]
            val_ds = PretrainDataset(data_path=args.data_path, max_length=args.max_seq_len, sample_indices=val_indices)
            if is_main_process():
                Logger(
                    f"Validation enabled: {len(val_indices)} held-out samples, "
                    f"interval={args.validation_interval_steps} optimizer updates"
                )
    elif validation_requested and is_main_process():
        Logger("Validation disabled: set both validation_split_ratio > 0 and validation_interval_steps > 0")

    if val_ds is None and args.time_to_target_metric != "none" and is_main_process():
        Logger("Time-to-target disabled because validation is not active")

    train_ds = PretrainDataset(data_path=args.data_path, max_length=args.max_seq_len, sample_indices=train_indices)
    return train_ds, val_ds


def _save_checkpoint(epoch: int, step: int, wandb=None) -> None:
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
        wandb=wandb,
        optimizer_step=int(metric_state["optimizer_step"]),
        save_dir=args.save_dir,
    )
    _mlflow_log_ckpt(ckp, step)
    model.train()
    del state_dict


def _log_train_window(epoch: int, step: int, iters: int, start_step: int, wandb=None) -> None:
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
    if wandb:
        wandb.log(
            {
                "loss": current_loss,
                "logits_loss": current_logits_loss,
                "aux_loss": current_aux_loss,
                "learning_rate": current_lr,
                "epoch_time": eta_min,
                "consumed_tokens": consumed_tokens,
            }
        )
    if is_main_process():
        _mlflow_log_step(
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
            attention_mask = batch["attention_mask"].to(args.device, non_blocking=True)
            with autocast_ctx:
                res = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels)
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
        _mlflow_log_metrics(step=mlflow_step, metrics={"val/loss": val_loss, "val/ppl": val_ppl})

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
            _mlflow_log_metrics(
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


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, val_loader=None):
    global epoch_start_time

    epoch_start_time = time.time()
    last_step = start_step
    total_update_steps = args.epochs * math.ceil(iters / max(args.accumulation_steps, 1))
    for step, batch in enumerate(loader, start=start_step + 1):
        input_ids = batch["input_ids"].to(args.device, non_blocking=True)
        labels = batch["labels"].to(args.device, non_blocking=True)
        position_ids = batch["position_ids"].to(args.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(args.device, non_blocking=True)
        last_step = step

        with autocast_ctx:
            res = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            metric_state["optimizer_step"] += 1
            metric_state["window_optimizer_steps"] += 1

        if step % args.log_interval == 0 or step == iters:
            _log_train_window(epoch, step, iters, start_step, wandb=wandb)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            _save_checkpoint(epoch, step, wandb=wandb)

        if update_due:
            _maybe_run_validation(epoch, step, iters, val_loader, force=(step == iters))

        del input_ids, labels, position_ids, attention_mask, res, loss

    return last_step


def _coerce_args(options):
    runtime_args = SimpleNamespace(**options)
    positive_int_fields = (
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "max_position_embeddings",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    )
    for field_name in positive_int_fields:
        if getattr(runtime_args, field_name) <= 0:
            raise ValueError(f"{field_name} must be > 0")
    if runtime_args.rms_norm_eps <= 0:
        raise ValueError("rms_norm_eps must be > 0")
    if runtime_args.rope_theta <= 0:
        raise ValueError("rope_theta must be > 0")
    if runtime_args.router_aux_loss_coef < 0.0:
        raise ValueError("router_aux_loss_coef must be >= 0.0")
    if runtime_args.dropout < 0.0 or runtime_args.dropout >= 1.0:
        raise ValueError("dropout must be >= 0.0 and < 1.0")
    if runtime_args.hidden_size % runtime_args.num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if runtime_args.num_attention_heads % runtime_args.num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if runtime_args.num_experts_per_tok > runtime_args.num_experts:
        raise ValueError("num_experts_per_tok must be <= num_experts")
    if runtime_args.lr_warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be >= 0")
    if not 0.0 <= runtime_args.lr_min_ratio <= 1.0:
        raise ValueError("lr_min_ratio must be >= 0.0 and <= 1.0")
    runtime_args.peak_tflops_per_gpu = (
        runtime_args.peak_tflops_per_gpu if runtime_args.peak_tflops_per_gpu > 0 else None
    )
    runtime_args.time_to_target_value = (
        runtime_args.time_to_target_value if runtime_args.time_to_target_value > 0 else None
    )
    runtime_args.peak_fp8_tflops_per_gpu = None
    return runtime_args


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
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        dropout=args.dropout,
        vocab_size=args.vocab_size,
        flash_attn=bool(args.flash_attn),
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        hidden_act=args.hidden_act,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=args.rms_norm_eps,
        rope_theta=args.rope_theta,
        inference_rope_scaling=bool(args.inference_rope_scaling),
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_intermediate_size=args.moe_intermediate_size,
        norm_topk_prob=bool(args.norm_topk_prob),
        router_aux_loss_coef=args.router_aux_loss_coef,
    )
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
        resolved_peak_profile = resolve_peak_flops_profile(gpu_name)
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

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = (
            f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 4b. MLflow (no-op unless MLFLOW_TRACKING_URI is set) ==========
    if is_main_process():
        _mlflow_start(args, lm_config)

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds, val_ds = _build_pretrain_datasets(tokenizer)
    train_sample_lengths = train_ds.sample_lengths()
    drop_last_for_compile = bool(args.use_compile)
    pretrain_collator = PretrainDataCollator(
        eos_token_id=int(tokenizer.eos_token_id),
        pad_token_id=int(tokenizer.pad_token_id),
        max_seq_len=args.max_seq_len,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    val_loader = None
    if val_ds is not None:
        val_batches = build_packed_batches(
            list(range(len(val_ds))),
            val_ds.sample_lengths(),
            args.batch_size,
            args.max_seq_len,
            drop_last=drop_last_for_compile,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_batches,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=pretrain_collator,
            persistent_workers=True,
            prefetch_factor=8
            
        )
    scaler = _build_grad_scaler(device_type, args.dtype)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

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
        model = torch.compile(model)
        Logger("torch.compile enabled")
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = list(train_sampler) if train_sampler is not None else torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = build_packed_batches(
            indices,
            train_sample_lengths,
            args.batch_size,
            args.max_seq_len,
            skip_batches=skip,
            drop_last=drop_last_for_compile,
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=pretrain_collator,
            persistent_workers=True,
            prefetch_factor=8
            
        )
        if skip > 0:
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始")
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb, val_loader=val_loader)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb, val_loader=val_loader)

    # ========== 9. 清理分布进程 ==========
    if is_main_process():
        _mlflow_finish()
    if dist.is_initialized():
        dist.destroy_process_group()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--save_dir", default="../out", show_default=True, type=str, help="模型保存目录")
@click.option("--save_weight", default="pretrain", show_default=True, type=str, help="保存权重的前缀名")
@click.option("--epochs", default=2, show_default=True, type=int, help="训练轮数")
@click.option("--batch_size", default=32, show_default=True, type=int, help="batch size")
@click.option("--learning_rate", default=5e-4, show_default=True, type=float, help="初始学习率")
@click.option(
    "--device",
    default="cuda:0" if torch.cuda.is_available() else "cpu",
    show_default=True,
    type=str,
    help="训练设备",
)
@click.option(
    "--dtype",
    default="bfloat16",
    show_default=True,
    type=click.Choice(["float16", "bfloat16", "float32"]),
    help="混合精度类型",
)
@click.option("--num_workers", default=8, show_default=True, type=int, help="数据加载线程数")
@click.option("--accumulation_steps", default=8, show_default=True, type=int, help="梯度累积步数")
@click.option("--grad_clip", default=1.0, show_default=True, type=float, help="梯度裁剪阈值")
@click.option("--log_interval", default=100, show_default=True, type=int, help="日志打印间隔")
@click.option("--save_interval", default=1000, show_default=True, type=int, help="模型保存间隔")
@click.option(
    "--lr_schedule",
    default="cosine",
    show_default=True,
    type=click.Choice(["cosine", "constant"]),
    help="学习率调度器",
)
@click.option("--lr_warmup_steps", default=0, show_default=True, type=int, help="学习率 warmup 更新步数")
@click.option("--lr_min_ratio", default=0.1, show_default=True, type=float, help="余弦调度最小学习率比例")
@click.option("--hidden_size", default=768, show_default=True, type=int, help="隐藏层维度")
@click.option("--num_hidden_layers", default=8, show_default=True, type=int, help="隐藏层数量")
@click.option("--dropout", default=0.0, show_default=True, type=float, help="dropout 概率")
@click.option("--vocab_size", default=6400, show_default=True, type=int, help="词表大小")
@click.option(
    "--flash_attn",
    default=1,
    show_default=True,
    type=click.IntRange(0, 1),
    help="是否启用 scaled_dot_product_attention 快路径（0=否，1=是）",
)
@click.option("--num_attention_heads", default=8, show_default=True, type=int, help="注意力头数量")
@click.option("--num_key_value_heads", default=4, show_default=True, type=int, help="KV 头数量")
@click.option(
    "--hidden_act",
    default="silu",
    show_default=True,
    type=click.Choice(["silu", "gelu", "relu", "swish"]),
    help="MLP 激活函数",
)
@click.option("--intermediate_size", default=2432, show_default=True, type=int, help="MLP 中间层维度")
@click.option(
    "--max_position_embeddings",
    default=32768,
    show_default=True,
    type=int,
    help="模型最大位置编码长度",
)
@click.option("--rms_norm_eps", default=1e-6, show_default=True, type=float, help="RMSNorm epsilon")
@click.option("--rope_theta", default=1e6, show_default=True, type=float, help="RoPE base theta")
@click.option(
    "--inference_rope_scaling",
    default=0,
    show_default=True,
    type=click.IntRange(0, 1),
    help="是否启用 YaRN rope scaling 配置（0=否，1=是）",
)
@click.option(
    "--max_seq_len", default=340, show_default=True, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）"
)
@click.option(
    "--use_moe", default=0, show_default=True, type=click.IntRange(0, 1), help="是否使用MoE架构（0=否，1=是）"
)
@click.option("--num_experts", default=4, show_default=True, type=int, help="MoE expert 数量")
@click.option(
    "--num_experts_per_tok",
    default=1,
    show_default=True,
    type=int,
    help="每个 token 路由到多少个 expert",
)
@click.option(
    "--moe_intermediate_size",
    default=2432,
    show_default=True,
    type=int,
    help="MoE expert 的中间层维度",
)
@click.option(
    "--norm_topk_prob",
    default=1,
    show_default=True,
    type=click.IntRange(0, 1),
    help="是否对 top-k 路由概率归一化（0=否，1=是）",
)
@click.option(
    "--router_aux_loss_coef",
    default=5e-4,
    show_default=True,
    type=float,
    help="MoE router auxiliary loss 系数",
)
@click.option(
    "--data_path", default="../dataset/pretrain_t2t_mini", show_default=True, type=str, help="预训练数据路径"
)
@click.option("--from_weight", default="none", show_default=True, type=str, help="基于哪个权重训练，为none则从头开始")
@click.option(
    "--from_resume", default=0, show_default=True, type=click.IntRange(0, 1), help="是否自动检测&续训（0=否，1=是）"
)
@click.option("--use_wandb", is_flag=True, help="是否使用wandb")
@click.option("--wandb_project", default="MiniMind-Pretrain", show_default=True, type=str, help="wandb项目名")
@click.option(
    "--use_compile",
    default=0,
    show_default=True,
    type=click.IntRange(0, 1),
    help="是否使用torch.compile加速（0=否，1=是）",
)
@click.option("--validation_split_ratio", default=0.0, show_default=True, type=float, help="验证集切分比例")
@click.option("--validation_interval_steps", default=0, show_default=True, type=int, help="验证间隔（优化器步数）")
@click.option("--peak_tflops_per_gpu", default=0.0, show_default=True, type=float, help="每卡峰值TFLOPs，用于MFU")
@click.option(
    "--time_to_target_metric",
    default="none",
    show_default=True,
    type=click.Choice(["none", "val_loss", "val_ppl"]),
    help="time-to-target 使用的验证指标",
)
@click.option("--time_to_target_value", default=0.0, show_default=True, type=float, help="time-to-target 阈值")
def main(**options):
    """MiniMind pretraining entrypoint."""

    run_training(_coerce_args(options))


if __name__ == "__main__":
    main()
