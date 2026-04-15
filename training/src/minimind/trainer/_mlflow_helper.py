"""MLflow integration for minimind — zero-behavior if tracking URI unset or unreachable."""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback

_active = False
_start_time = None
_mlflow_module = None
_metric_queue = None
_metric_worker = None
_metric_stop_event = None
_dropped_metric_events = 0

_METRIC_QUEUE_MAXSIZE = 256
_METRIC_QUEUE_POLL_SECONDS = 0.2
_METRIC_FLUSH_TIMEOUT_SECONDS = 5.0


def _reset_runtime_state() -> None:
    global \
        _active, \
        _start_time, \
        _mlflow_module, \
        _metric_queue, \
        _metric_worker, \
        _metric_stop_event, \
        _dropped_metric_events

    _active = False
    _start_time = None
    _mlflow_module = None
    _metric_queue = None
    _metric_worker = None
    _metric_stop_event = None
    _dropped_metric_events = 0


def _metric_worker_loop() -> None:
    while _metric_queue is not None and _metric_stop_event is not None:
        if _metric_stop_event.is_set() and _metric_queue.empty():
            return
        try:
            payload = _metric_queue.get(timeout=_METRIC_QUEUE_POLL_SECONDS)
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

    _metric_queue = queue.Queue(maxsize=_METRIC_QUEUE_MAXSIZE)
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


def _drain_metrics() -> None:
    global _dropped_metric_events

    if _metric_queue is None or _metric_stop_event is None:
        return

    _metric_stop_event.set()
    deadline = time.time() + _METRIC_FLUSH_TIMEOUT_SECONDS
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
def start(args, model_config, script_name="train_pretrain"):
    global _active, _start_time, _mlflow_module

    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        print("[mlflow] MLFLOW_TRACKING_URI not set — skipping integration", flush=True)
        return
    try:
        import mlflow
    except ImportError:
        print("[mlflow] mlflow python package not installed — skipping", flush=True)
        return

    timeout_seconds = int(os.environ.get("MLFLOW_START_TIMEOUT_SECONDS", "180"))
    retry_seconds = float(os.environ.get("MLFLOW_START_RETRY_SECONDS", "5"))
    deadline = time.time() + timeout_seconds
    attempt = 0
    while True:
        attempt += 1
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
                "verda.run_name": os.environ.get("DSTACK_RUN_NAME", ""),
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
        _drain_metrics()
    finally:
        _active = False
        try:
            _mlflow_module.end_run(status=status)
        except Exception as exc:
            print(f"[mlflow] finish failed: {exc}", flush=True)
        finally:
            _reset_runtime_state()
