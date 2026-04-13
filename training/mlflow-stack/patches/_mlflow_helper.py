"""MLflow integration for minimind — zero-behavior if tracking URI unset or unreachable."""
import os
import time
import traceback

_active = False
_start_time = None


def _safe(fn):
    def wrapper(*a, **kw):
        if not _active:
            return
        try:
            return fn(*a, **kw)
        except Exception as e:
            print(f"[mlflow] {fn.__name__} failed: {e}", flush=True)
    return wrapper


# doc-anchor: mlflow-helper-start
def start(args, model_config, script_name="train_pretrain"):
    global _active, _start_time
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        print("[mlflow] MLFLOW_TRACKING_URI not set — skipping integration", flush=True)
        return
    try:
        import mlflow
    except ImportError:
        print("[mlflow] mlflow python package not installed — skipping", flush=True)
        return
    try:
        import torch
        mlflow.set_tracking_uri(uri)
        exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "minimind-pretrain")
        mlflow.set_experiment(exp_name)
        run_name = (
            f"{script_name}-h{args.hidden_size}-L{args.num_hidden_layers}"
            f"-bs{args.batch_size}-lr{args.learning_rate}"
        )
        log_sys = os.environ.get("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "").lower() in ("1", "true", "yes")
        tags = {
            "script": script_name,
            "model_family": "minimind",
            "hidden_size": str(args.hidden_size),
            "num_hidden_layers": str(args.num_hidden_layers),
            "use_moe": str(bool(args.use_moe)),
            "dtype": args.dtype,
            "verda.profile": os.environ.get("VERDA_PROFILE", ""),
            "verda.emulation": os.environ.get("VERDA_EMULATION", "true"),
        }
        mlflow.start_run(run_name=run_name, log_system_metrics=log_sys, tags=tags)
        mlflow.log_params({k: str(v) for k, v in vars(args).items() if v is not None})
        try:
            cfg = {k: v for k, v in vars(model_config).items() if not k.startswith("_")}
            mlflow.log_dict(cfg, "config/model_config.json")
        except Exception:
            pass
        env_info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_version": getattr(torch.version, "cuda", None),
        }
        mlflow.log_dict(env_info, "config/env.json")
        _active = True
        _start_time = time.time()
        print(f"[mlflow] active — uri={uri} experiment={exp_name} run={run_name}", flush=True)
    except Exception:
        traceback.print_exc()
        _active = False


@_safe
def log_step(step, epoch, loss, logits_loss, aux_loss, lr, tokens_seen=None):
    import mlflow
    metrics = {
        "train/loss": float(loss),
        "train/logits_loss": float(logits_loss),
        "train/aux_loss": float(aux_loss),
        "train/lr": float(lr),
        "train/epoch": int(epoch),
    }
    if _start_time is not None:
        metrics["train/seconds_elapsed"] = time.time() - _start_time
    if tokens_seen is not None:
        metrics["train/tokens_seen"] = float(tokens_seen)
    mlflow.log_metrics(metrics, step=int(step))


@_safe
def log_checkpoint(path, step):
    import mlflow
    if path and os.path.exists(path):
        mlflow.log_artifact(path, artifact_path=f"checkpoints/step-{step}")


@_safe
def finish(status="FINISHED"):
    import mlflow
    mlflow.end_run(status=status)
