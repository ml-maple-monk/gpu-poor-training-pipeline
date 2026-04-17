"""MLflow integration for minimind — zero-behavior if tracking URI unset or unreachable."""

from __future__ import annotations

import importlib.util
import os
import queue
import sys
import threading
import time
import traceback

import torch

_active = False
_start_time = None
_mlflow_module = None
_metric_queue = None
_metric_worker = None
_metric_stop_event = None
_dropped_metric_events = 0

# Module-level metric-queue tunables, set from mlflow_config in start().
_metric_queue_maxsize = 256
_metric_queue_poll_seconds = 0.2
_metric_flush_timeout_seconds = 5.0

# Stashed reference so post-start helpers can read it without env vars.
_mlflow_config: dict = {}


def _recipe_config_from_runtime(args, mlflow_config: dict) -> dict[str, object]:
    return {
        "kind": str(mlflow_config.get("recipe_kind", "minimind_pretrain")),
        "prepare_data": bool(mlflow_config.get("recipe_prepare_data", False)),
        "dataset_path": getattr(args, "data_path", None),
        "output_dir": getattr(args, "save_dir", None),
        "runtime_dataset_path": getattr(args, "data_path", None),
        "runtime_output_dir": getattr(args, "save_dir", None),
        "time_cap_seconds": mlflow_config.get("time_cap_seconds"),
        "max_seq_len": getattr(args, "max_seq_len", None),
        "validation_split_ratio": getattr(args, "validation_split_ratio", None),
        "validation_interval_steps": getattr(args, "validation_interval_steps", None),
    }


def _log_runtime_config(mlflow, args, model_config, mlflow_config: dict) -> None:
    recipe_cfg = _recipe_config_from_runtime(args, mlflow_config)
    training_cfg = {k: v for k, v in vars(args).items() if v is not None}

    mlflow.log_params({k: str(v) for k, v in training_cfg.items()})
    mlflow.log_params({f"recipe.{k}": str(v) for k, v in recipe_cfg.items() if v is not None})
    mlflow.log_params({f"mlflow.{k}": str(v) for k, v in mlflow_config.items() if v is not None})
    try:
        cfg = {k: v for k, v in vars(model_config).items() if not k.startswith("_")}
        mlflow.log_dict(cfg, "config/model_config.json")
    except Exception:
        pass
    mlflow.log_dict(training_cfg, "config/training_args.json")
    mlflow.log_dict(recipe_cfg, "config/recipe_config.json")
    mlflow.log_dict(dict(mlflow_config), "config/mlflow_config.json")


def _reset_runtime_state() -> None:
    global \
        _active, \
        _start_time, \
        _mlflow_module, \
        _metric_queue, \
        _metric_worker, \
        _metric_stop_event, \
        _dropped_metric_events, \
        _mlflow_config

    _active = False
    _start_time = None
    _mlflow_module = None
    _metric_queue = None
    _metric_worker = None
    _metric_stop_event = None
    _dropped_metric_events = 0
    _mlflow_config = {}


def _metric_worker_loop() -> None:
    while _metric_queue is not None and _metric_stop_event is not None:
        if _metric_stop_event.is_set() and _metric_queue.empty():
            return
        try:
            payload = _metric_queue.get(timeout=_metric_queue_poll_seconds)
        except queue.Empty:
            continue
        try:
            if _mlflow_module is not None:
                _mlflow_module.log_metrics(payload["metrics"], step=payload["step"])
        except Exception as exc:
            print(f"[mlflow] async metric flush failed: {exc}", flush=True)
        finally:
            _metric_queue.task_done()


def _ensure_worker() -> None:
    global _metric_queue, _metric_worker, _metric_stop_event

    _metric_queue = queue.Queue(maxsize=_metric_queue_maxsize)
    _metric_stop_event = threading.Event()
    _metric_worker = threading.Thread(target=_metric_worker_loop, name="mlflow-metric-writer", daemon=True)
    _metric_worker.start()


def _enqueue_metrics(metrics, step) -> None:
    global _dropped_metric_events

    if not _active or _metric_queue is None:
        return
    try:
        _metric_queue.put_nowait({"metrics": metrics, "step": int(step)})
    except queue.Full:
        _dropped_metric_events += 1


def _load_mlflow_module():
    existing = sys.modules.get("mlflow")
    if existing is not None:
        return existing

    try:
        spec = importlib.util.find_spec("mlflow")
    except ValueError:
        return None

    if spec is None:
        return None

    import mlflow

    return mlflow


def _drain_metrics() -> None:
    global _dropped_metric_events

    if _metric_queue is None or _metric_stop_event is None:
        return

    _metric_stop_event.set()
    deadline = time.time() + _metric_flush_timeout_seconds
    while _metric_queue.unfinished_tasks > 0 and time.time() < deadline:
        time.sleep(0.05)

    abandoned = 0
    while not _metric_queue.empty():
        try:
            _metric_queue.get_nowait()
            _metric_queue.task_done()
            abandoned += 1
        except queue.Empty:
            break

    if abandoned > 0:
        _dropped_metric_events += abandoned

    if _metric_worker is not None:
        remaining = max(0.0, deadline - time.time())
        _metric_worker.join(timeout=remaining)

    if _dropped_metric_events > 0:
        print(f"[mlflow] dropped {_dropped_metric_events} metric event(s)", flush=True)


# doc-anchor: mlflow-helper-start
def start(runtime_args, model_config, mlflow_config: dict) -> None:
    global _active, _start_time, _mlflow_module, _mlflow_config
    global _metric_queue_maxsize, _metric_queue_poll_seconds, _metric_flush_timeout_seconds

    # Stash config for helpers that run after start().
    _mlflow_config = dict(mlflow_config)

    # Apply metric-queue tunables from config.
    _metric_queue_maxsize = int(mlflow_config.get("metric_queue_maxsize", 256))
    _metric_queue_poll_seconds = float(mlflow_config.get("metric_queue_poll_seconds", 0.2))
    _metric_flush_timeout_seconds = float(mlflow_config.get("metric_flush_timeout_seconds", 5.0))

    # Prefer MLFLOW_TRACKING_URI env var (set by launch code with tunnel URL)
    # over TOML tracking_uri (which has the local-only host.docker.internal).
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip() or str(mlflow_config.get("tracking_uri", "")).strip()
    if not uri:
        print("[mlflow] tracking_uri not set — skipping integration", flush=True)
        return

    # Set env vars that the MLflow SDK reads directly.
    os.environ["MLFLOW_TRACKING_URI"] = uri
    if mlflow_config.get("enable_system_metrics_logging"):
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = str(
        mlflow_config.get("system_metrics_sampling_interval", "5")
    )
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING"] = str(
        mlflow_config.get("system_metrics_samples_before_logging", "1")
    )
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = str(
        mlflow_config.get("http_request_max_retries", "7")
    )
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(
        mlflow_config.get("http_request_timeout_seconds", "120")
    )

    mlflow = _load_mlflow_module()
    if mlflow is None:
        print("[mlflow] mlflow python package not installed — skipping", flush=True)
        return

    timeout_seconds = int(mlflow_config.get("start_timeout_seconds", 180))
    retry_seconds = float(mlflow_config.get("start_retry_seconds", 5))
    deadline = time.time() + timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        try:
            mlflow.set_tracking_uri(uri)
            exp_name = str(mlflow_config.get("experiment_name", "minimind-pretrain"))
            mlflow.set_experiment(exp_name)
            script_name = str(mlflow_config.get("script_name", "train_pretrain"))
            run_name = (
                f"{script_name}-h{runtime_args.hidden_size}-L{runtime_args.num_hidden_layers}"
                f"-bs{runtime_args.batch_size}-lr{runtime_args.learning_rate}"
            )
            log_sys = bool(mlflow_config.get("enable_system_metrics_logging", False))
            tags = {
                "script": script_name,
                "model_family": "minimind",
                "hidden_size": str(runtime_args.hidden_size),
                "num_hidden_layers": str(runtime_args.num_hidden_layers),
                "use_moe": str(bool(runtime_args.use_moe)),
                "dtype": runtime_args.dtype,
                "recipe.kind": str(mlflow_config.get("recipe_kind", "minimind_pretrain")),
                "verda.profile": os.environ.get("VERDA_PROFILE", ""),
                "verda.emulation": os.environ.get("VERDA_EMULATION", "true"),
                "verda.run_name": os.environ.get("DSTACK_RUN_NAME", ""),
            }
            mlflow.start_run(run_name=run_name, log_system_metrics=log_sys, tags=tags)
            _log_runtime_config(mlflow, runtime_args, model_config, mlflow_config)
            env_info = {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "cuda_version": getattr(torch.version, "cuda", None),
            }
            mlflow.log_dict(env_info, "config/env.json")
            _mlflow_module = mlflow
            _active = True
            _start_time = time.time()
            _ensure_worker()
            print(f"[mlflow] active — uri={uri} experiment={exp_name} run={run_name}", flush=True)
            return
        except Exception as exc:
            traceback.print_exc()
            _active = False
            if time.time() >= deadline:
                print(
                    f"[mlflow] giving up after {attempt} attempts over {timeout_seconds}s: {exc}",
                    flush=True,
                )
                return
            print(
                f"[mlflow] start attempt {attempt} failed; retrying in {retry_seconds}s",
                flush=True,
            )
            time.sleep(retry_seconds)


def log_metrics(*, step, metrics) -> None:
    if not _active:
        return
    try:
        _enqueue_metrics(dict(metrics), step)
    except Exception as exc:
        print(f"[mlflow] log_metrics failed: {exc}", flush=True)


def log_step(step, epoch, loss, logits_loss, aux_loss, lr, tokens_seen=None, update_step=None, extra_metrics=None):
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
        metrics["train/consumed_tokens"] = float(tokens_seen)
    if update_step is not None:
        metrics["train/update_step"] = float(update_step)
    if extra_metrics:
        metrics.update(extra_metrics)
    log_metrics(step=step, metrics=metrics)


def log_checkpoint(path, step):
    if not _active or _mlflow_module is None:
        return
    if not bool(_mlflow_config.get("artifact_upload", False)):
        return
    try:
        if path and os.path.exists(path):
            _mlflow_module.log_artifact(path, artifact_path=f"checkpoints/step-{step}")
    except Exception as exc:
        print(f"[mlflow] log_checkpoint failed: {exc}", flush=True)


def finish(status="FINISHED"):
    global _active

    if not _active or _mlflow_module is None:
        return

    try:
        try:
            _drain_metrics()
        except Exception as exc:
            print(f"[mlflow] metric drain failed during finish: {exc}", flush=True)
    finally:
        _active = False
        try:
            _mlflow_module.end_run(status=status)
        except Exception as exc:
            print(f"[mlflow] finish failed: {exc}", flush=True)
        finally:
            _reset_runtime_state()
