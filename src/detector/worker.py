"""SQS worker: collect images into a batch, run the model, commit results, ack messages.

Each batch goes through three steps:
    prepare: receive -> skip finished -> load (download + decode + letterbox, in a pool)
    infer:   inference -> postprocess
    finish:  write part file + commit to DynamoDB -> copy permanent failures to the DLQ
             -> delete

run_once() runs them one after another. run_pipelined() (PIPELINE=1) runs each step in its
own thread, joined by one-slot queues, so the GPU works on batch N while batch N+1 is
prepared and batch N-1 is written: at most three batches are in flight.

A message is deleted only once every image it names is committed, already done, or has
failed permanently (corrupt file, missing object). Anything else (throttling, a missing
batch record, a crash) leaves the message on the queue; SQS retries it and moves it to
the DLQ after the queue's maxReceiveCount.

Run in a container with:  python -m detector.worker
"""

from __future__ import annotations

import json
import logging
import os
import queue
import signal
import socket
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from botocore.exceptions import ClientError

from detector.gpu_monitor import GpuMonitor
from detector.inference_core import (
    DecodeError,
    LetterboxMeta,
    StageTimer,
    decode,
    letterbox,
    postprocess,
    to_batch,
)
from detector.messages import BadMessage, ImageRef, parse_message, results_key
from detector.runtimes import Runtime
from detector.store import BatchNotFound, BatchStore, ImageResult, is_complete
from detector.thread_monitor import ThreadMonitor

log = logging.getLogger("detector.worker")

ImageKey = tuple[str, str]  # (batch_id, image_name)
Loaded = tuple[np.ndarray, LetterboxMeta]  # a letterboxed image ready for the batch

RUNTIMES = ("pytorch", "onnx-cuda", "onnx-trt")


def _env_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class WorkerConfig:
    queue_url: str
    dlq_url: str
    bucket: str
    table: str
    batch_size: int = 16
    max_wait_s: float = 1.0
    wait_time_s: int = 20  # long poll while idle
    model: str = "yolov8m"
    device: str = "auto"
    half: bool = False
    runtime: str = "pytorch"  # pytorch | onnx-cuda | onnx-trt
    pipeline: bool = False  # run prepare / infer / finish in parallel threads
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300
    multi_label: bool = False  # every class above conf per anchor (mAP eval runs)
    thread_profile: bool = False  # log CPU use per thread name with each batch
    trt_options: str = ""  # JSON TensorRT provider options over the defaults (experiments)
    trt_prebuild: bool = True  # build the TensorRT engine in a subprocess, then load it
    trt_cache_dir: str = "/tmp/trt-cache"  # detector.runtimes.TRT_CACHE_DIR (no torch import)
    download_threads: int = 8
    finalize_function: str = ""  # Lambda to start when a commit completes a batch; "" = off
    worker_id: str = field(default_factory=lambda: f"{socket.gethostname()}-{os.getpid()}")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> WorkerConfig:
        env = dict(os.environ if env is None else env)
        missing = [k for k in ("QUEUE_URL", "DLQ_URL", "BUCKET", "TABLE") if not env.get(k)]
        if missing:
            raise ValueError(f"missing environment variables: {', '.join(missing)}")
        kwargs: dict[str, Any] = {
            "queue_url": env["QUEUE_URL"],
            "dlq_url": env["DLQ_URL"],
            "bucket": env["BUCKET"],
            "table": env["TABLE"],
        }
        casts: dict[str, tuple[str, Callable[[str], Any]]] = {
            "batch_size": ("BATCH_SIZE", int),
            "max_wait_s": ("MAX_WAIT_S", float),
            "wait_time_s": ("WAIT_TIME_S", int),
            "model": ("MODEL", str),
            "device": ("DEVICE", str),
            "half": ("HALF", _env_bool),
            "runtime": ("RUNTIME", str),
            "pipeline": ("PIPELINE", _env_bool),
            "conf": ("CONF", float),
            "iou": ("IOU", float),
            "max_det": ("MAX_DET", int),
            "multi_label": ("MULTI_LABEL", _env_bool),
            "thread_profile": ("THREAD_PROFILE", _env_bool),
            "trt_options": ("TRT_OPTIONS", str),
            "trt_prebuild": ("TRT_PREBUILD", _env_bool),
            "trt_cache_dir": ("TRT_CACHE_DIR", str),
            "download_threads": ("DOWNLOAD_THREADS", int),
            "finalize_function": ("FINALIZE_FUNCTION", str),
            "worker_id": ("WORKER_ID", str),
        }
        for attr, (name, cast) in casts.items():
            if env.get(name):
                kwargs[attr] = cast(env[name])
        cfg = cls(**kwargs)
        if not 1 <= cfg.batch_size <= 99:
            raise ValueError("BATCH_SIZE must be between 1 and 99")
        if cfg.runtime not in RUNTIMES:
            raise ValueError(f"RUNTIME must be one of {', '.join(RUNTIMES)}")
        return cfg


@dataclass
class BatchReport:
    """What one batch did; logged as one JSON line."""

    messages: int = 0
    images: int = 0
    skipped: int = 0
    processed: int = 0
    failed: int = 0
    retry: int = 0
    deleted: int = 0
    dead_lettered: int = 0
    batch_ids: list[str] = field(default_factory=list)
    stages_ms: dict[str, float] = field(default_factory=dict)
    # Wall clock (epoch seconds): when collection started and when results were committed.
    # The benchmark measures a run from the first t_receive to the last t_written.
    t_receive: float = 0.0
    t_written: float = 0.0
    gpu: dict[str, float] | None = None  # GpuMonitor.take(): mean utilization, peak memory
    pipelined: bool = False
    threads: dict | None = None  # ThreadMonitor.take(): CPU % per thread name (THREAD_PROFILE)


@dataclass
class PreparedBatch:
    """One batch between the prepare, infer and finish steps."""

    report: BatchReport
    timer: StageTimer
    messages: list[dict]
    images: dict[ImageKey, ImageRef]  # every image once
    msgs_for: dict[ImageKey, set[str]]  # image -> messages that name it
    retry_msgs: set[str] = field(default_factory=set)  # leave on the queue for SQS to retry
    results: dict[ImageKey, ImageResult] = field(default_factory=dict)
    keys: list[ImageKey] = field(default_factory=list)  # one per row of batch
    batch: np.ndarray | None = None  # (B, 3, 640, 640) float32
    metas: list[LetterboxMeta] = field(default_factory=list)


class Worker:
    def __init__(
        self,
        cfg: WorkerConfig,
        runtime: Runtime,
        sqs: Any,
        s3: Any,
        store: BatchStore,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        gpu_monitor: GpuMonitor | None = None,
        thread_monitor: ThreadMonitor | None = None,
        finalizer: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self.sqs = sqs
        self.s3 = s3
        self.store = store
        self.clock = clock
        self.wall = wall
        self.gpu_monitor = gpu_monitor
        self.thread_monitor = thread_monitor
        self.finalizer = finalizer
        self._stop = threading.Event()
        self._seq = 0
        # Part files are named <worker_id>-<run token>-<seq>: a restarted worker with the same
        # id starts seq at 1 again and must not overwrite its previous run's files.
        self._part_prefix = f"{cfg.worker_id}-{uuid.uuid4().hex[:8]}"
        self._pool = ThreadPoolExecutor(max_workers=cfg.download_threads)

    # Lifecycle -------------------------------------------------------------

    def request_stop(self, *_: Any) -> None:
        """Finish the current batch, then stop (SIGTERM from ECS scale-in)."""
        log.info("stop requested; finishing current batch")
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def run_forever(self) -> None:
        signal.signal(signal.SIGTERM, self.request_stop)
        mode = "pipelined" if self.cfg.pipeline else "serial"
        log.info("worker %s started: %s, %s", self.cfg.worker_id, self.runtime.name, mode)
        if self.cfg.pipeline:
            self.run_pipelined()
        else:
            while not self.stopping:
                try:
                    self.run_once()
                except Exception:
                    # Messages from a failed batch stay on the queue and are retried.
                    log.exception("batch failed; its messages will be retried")
                    time.sleep(1)
        self._pool.shutdown(wait=True)
        log.info("worker %s stopped", self.cfg.worker_id)

    # Serial -----------------------------------------------------------------

    def run_once(self) -> BatchReport | None:
        """Process one batch, one step after another. Returns None if no messages arrived."""
        prepared = self._prepare()
        if prepared is None:
            return None
        return self._finish(self._infer(prepared))

    # Pipelined --------------------------------------------------------------

    def run_pipelined(self, idle_exit: bool = False) -> list[BatchReport]:
        """Prepare, infer and finish in three threads until stop is requested.

        Each step hands its batch to the next through a one-slot queue, so a step blocks
        once the next one falls behind and at most three batches are in flight. A batch
        whose step raises is dropped there; its messages were never deleted, so SQS
        redelivers them after the visibility timeout. On stop, collection ends and the
        batches already in flight are finished. idle_exit=True also ends the run when a
        poll returns no messages (for tests).
        """
        to_infer: queue.Queue[PreparedBatch | None] = queue.Queue(maxsize=1)
        to_finish: queue.Queue[PreparedBatch | None] = queue.Queue(maxsize=1)
        reports: list[BatchReport] = []

        def prepare_loop() -> None:
            try:
                while not self.stopping:
                    try:
                        prepared = self._prepare(pipelined=True)
                    except Exception:
                        log.exception("prepare failed; the batch's messages will be retried")
                        time.sleep(1)
                        continue
                    if prepared is None:
                        if idle_exit:
                            break
                        continue
                    to_infer.put(prepared)
            finally:
                to_infer.put(None)

        def finish_loop() -> None:
            while (prepared := to_finish.get()) is not None:
                try:
                    reports.append(self._finish(prepared))
                except Exception:
                    log.exception("finish failed; the batch's messages will be retried")

        preparer = threading.Thread(target=prepare_loop, name="prepare", daemon=True)
        finisher = threading.Thread(target=finish_loop, name="finish", daemon=True)
        preparer.start()
        finisher.start()
        try:
            while True:
                with_wait = time.perf_counter()
                prepared = to_infer.get()
                if prepared is None:
                    break
                prepared.timer.totals["wait_input"] = time.perf_counter() - with_wait
                try:
                    to_finish.put(self._infer(prepared))
                except Exception:
                    log.exception("inference failed; the batch's messages will be retried")
        finally:
            to_finish.put(None)
            finisher.join()
            preparer.join(timeout=30)
        return reports

    # Steps --------------------------------------------------------------------

    def _prepare(self, pipelined: bool = False) -> PreparedBatch | None:
        """Receive a batch, drop finished images, load the rest. None if nothing arrived."""
        # No GPU sync here: in the pipeline this runs while the GPU works on another batch.
        timer = StageTimer()
        report = BatchReport(t_receive=self.wall(), pipelined=pipelined)

        with timer.stage("receive"):
            refs, messages = self._collect(report)
        if not messages:
            self._gpu_window()  # idle time doesn't count toward the next batch
            self._thread_window()
            return None
        report.messages = len(messages)

        # Every image once, even if two messages name it (S3 can deliver an event twice).
        images: dict[ImageKey, ImageRef] = {}
        msgs_for: dict[ImageKey, set[str]] = defaultdict(set)
        for ref in refs:
            key = (ref.batch_id, ref.image_name)
            images.setdefault(key, ref)
            msgs_for[key].add(ref.message_id)
        report.images = len(images)
        report.batch_ids = sorted({k[0] for k in images})
        p = PreparedBatch(
            report=report, timer=timer, messages=messages, images=images, msgs_for=msgs_for
        )

        with timer.stage("skip_check"):
            done: set[ImageKey] = set()
            for batch_id in report.batch_ids:
                names = [n for b, n in images if b == batch_id]
                done |= {(batch_id, n) for n in self.store.existing(batch_id, names)}
        report.skipped = len(done)
        todo = [k for k in images if k not in done]

        # Download, decode and letterbox each image in the pool: S3 reads and OpenCV
        # release the GIL, so the instance's cores share the CPU work.
        rows: list[np.ndarray] = []
        with timer.stage("load"):
            for key, outcome in zip(todo, self._pool.map(self._load, todo), strict=True):
                if isinstance(outcome, tuple):
                    rows.append(outcome[0])
                    p.metas.append(outcome[1])
                    p.keys.append(key)
                elif isinstance(outcome, ImageResult):
                    p.results[key] = outcome
                else:
                    log.warning("download of %s failed, will retry: %s", images[key].key, outcome)
                    p.retry_msgs |= msgs_for[key]
        if rows:
            with timer.stage("preprocess"):
                p.batch = to_batch(rows)
        return p

    def _infer(self, p: PreparedBatch) -> PreparedBatch:
        if p.batch is not None:
            with p.timer.stage("inference"):
                out = self.runtime.infer(p.batch)
                self.runtime.sync()  # charge the GPU work to this stage
            with p.timer.stage("postprocess"):
                dets = postprocess(
                    out,
                    p.metas,
                    conf=self.cfg.conf,
                    iou=self.cfg.iou,
                    max_det=self.cfg.max_det,
                    multi_label=self.cfg.multi_label,
                )
            for key, det in zip(p.keys, dets, strict=True):
                p.results[key] = ImageResult.done(key[1], det)
            p.batch = None  # free the input while the batch waits to be written
        return p

    def _finish(self, p: PreparedBatch) -> BatchReport:
        report, timer, results, msgs_for = p.report, p.timer, p.results, p.msgs_for
        newly_failed: list[ImageKey] = []
        with timer.stage("write"):
            by_batch: dict[str, list[ImageResult]] = defaultdict(list)
            for (batch_id, _), result in results.items():
                by_batch[batch_id].append(result)
            for batch_id, batch_results in by_batch.items():
                try:
                    self._write_part(batch_id, batch_results)
                    outcome = self.store.commit(
                        batch_id,
                        batch_results,
                        worker_id=self.cfg.worker_id,
                        runtime=self.runtime.name,
                        precision=self.runtime.precision,
                    )
                except BatchNotFound as e:
                    log.warning("%s; leaving its messages for retry", e)
                    for r in batch_results:
                        p.retry_msgs |= msgs_for[(batch_id, r.image_name)]
                    continue
                report.processed += sum(r.status == "done" for r in batch_results)
                report.skipped += outcome.skipped
                if outcome.written:
                    self._maybe_finalize(batch_id)
                for r in batch_results:
                    if r.status == "failed":
                        report.failed += 1
                        newly_failed.append((batch_id, r.image_name))
        report.t_written = self.wall()

        with timer.stage("ack"):
            by_id = {m["MessageId"]: m for m in p.messages}
            for key in newly_failed:
                for msg_id in msgs_for[key] - p.retry_msgs:
                    self._dead_letter(by_id[msg_id], results[key].error or "failed")
                    report.dead_lettered += 1
            to_delete = [m for m in p.messages if m["MessageId"] not in p.retry_msgs]
            report.deleted = self._delete(to_delete)
            report.retry = len(p.messages) - len(to_delete)

        report.stages_ms = {name: round(1000 * total, 2) for name, total in timer.totals.items()}
        report.gpu = self._gpu_window()
        report.threads = self._thread_window()
        log.info(json.dumps({"event": "batch", "worker_id": self.cfg.worker_id, **vars(report)}))
        return report

    def _maybe_finalize(self, batch_id: str) -> None:
        """Start finalize (asynchronously) if this commit completed the batch.

        Several workers can see the batch complete at once; the finalize step's conditional
        claim lets only one of them run. A failure here never fails the batch: GET
        /batches/{id} starts finalize too.
        """
        if self.finalizer is None:
            return
        try:
            batch = self.store.get_batch(batch_id)
            if batch and batch["status"] == "pending" and is_complete(batch):
                self.finalizer(batch_id)
                log.info(json.dumps({"event": "finalize_started", "batch_id": batch_id}))
        except Exception:
            log.exception("could not start finalize for %s", batch_id)

    def _gpu_window(self) -> dict[str, float] | None:
        return self.gpu_monitor.take() if self.gpu_monitor else None

    def _thread_window(self) -> dict | None:
        return self.thread_monitor.take() if self.thread_monitor else None

    def _collect(self, report: BatchReport) -> tuple[list[ImageRef], list[dict]]:
        """Receive until batch_size images are in hand or max_wait_s after the first one.

        A message received twice (its visibility timeout ran out mid-collection) is kept
        once, with its newest receipt handle; older handles can no longer delete it.
        """
        refs_by_msg: dict[str, list[ImageRef]] = {}
        messages: dict[str, dict] = {}
        deadline: float | None = None
        while sum(map(len, refs_by_msg.values())) < self.cfg.batch_size and not self.stopping:
            if deadline is None:
                wait = self.cfg.wait_time_s
            else:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    break
                wait = min(int(remaining), 20)
            have = sum(map(len, refs_by_msg.values()))
            resp = self.sqs.receive_message(
                QueueUrl=self.cfg.queue_url,
                MaxNumberOfMessages=max(1, min(10, self.cfg.batch_size - have)),
                WaitTimeSeconds=wait,
                AttributeNames=["ApproximateReceiveCount"],
            )
            received = resp.get("Messages", [])
            if not received:
                if deadline is None or wait == 0:
                    break  # idle, or the queue is empty right now
                continue
            for msg in received:
                try:
                    parsed = parse_message(msg)
                except BadMessage as e:
                    log.warning("bad message %s: %s", msg.get("MessageId"), e)
                    self._dead_letter(msg, str(e))
                    report.dead_lettered += 1
                    self._delete([msg])
                    continue
                if not parsed:  # S3 test event
                    self._delete([msg])
                    continue
                messages[msg["MessageId"]] = msg
                refs_by_msg[msg["MessageId"]] = parsed
                if deadline is None:
                    deadline = self.clock() + self.cfg.max_wait_s
        refs = [ref for parsed in refs_by_msg.values() for ref in parsed]
        return refs, list(messages.values())

    def _load(self, key: ImageKey) -> Loaded | ImageResult | Exception:
        """A letterboxed image, a permanent failure, or a temporary error to retry."""
        batch_id, name = key
        s3_key = f"input/{batch_id}/{name}"
        try:
            data = self.s3.get_object(Bucket=self.cfg.bucket, Key=s3_key)["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                return ImageResult.failed(name, "object missing from S3")
            return e
        except Exception as e:  # network errors etc.: retry
            return e
        try:
            return letterbox(decode(data))
        except DecodeError as e:
            return ImageResult.failed(name, f"decode failed: {e}")

    def _write_part(self, batch_id: str, results: list[ImageResult]) -> None:
        self._seq += 1
        lines = [
            json.dumps(
                {
                    "batch_id": batch_id,
                    **r.to_json(),
                    "worker_id": self.cfg.worker_id,
                    "runtime": self.runtime.name,
                    "precision": self.runtime.precision,
                }
            )
            for r in results
        ]
        self.s3.put_object(
            Bucket=self.cfg.bucket,
            Key=results_key(batch_id, self._part_prefix, self._seq),
            Body=("\n".join(lines) + "\n").encode(),
            ContentType="application/x-ndjson",
        )

    def _dead_letter(self, msg: dict, error: str) -> None:
        self.sqs.send_message(
            QueueUrl=self.cfg.dlq_url,
            MessageBody=msg.get("Body") or "",
            MessageAttributes={
                "error": {"DataType": "String", "StringValue": error[:256] or "error"},
                "worker_id": {"DataType": "String", "StringValue": self.cfg.worker_id},
            },
        )

    def _delete(self, messages: list[dict]) -> int:
        deleted = 0
        for i in range(0, len(messages), 10):
            chunk = messages[i : i + 10]
            resp = self.sqs.delete_message_batch(
                QueueUrl=self.cfg.queue_url,
                Entries=[
                    {"Id": str(n), "ReceiptHandle": m["ReceiptHandle"]} for n, m in enumerate(chunk)
                ],
            )
            deleted += len(resp.get("Successful", []))
            for failure in resp.get("Failed", []):
                # It will be redelivered and then skipped as already done.
                log.warning("delete failed: %s", failure)
        return deleted


def _lambda_finalizer(function_name: str, boto_cfg: Any) -> Callable[[str], None] | None:
    if not function_name:
        return None
    import boto3

    from detector.finalize import invoke_async

    client = boto3.client("lambda", config=boto_cfg)
    return lambda batch_id: invoke_async(client, function_name, batch_id)


def main() -> None:
    import boto3
    from botocore.config import Config

    from detector.runtimes import make_runtime, parse_trt_options

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = WorkerConfig.from_env()
    boto_cfg = Config(retries={"mode": "standard", "max_attempts": 5})
    started = time.monotonic()
    # Loading plus warm-up: for TensorRT this includes building the engine (minutes on a T4).
    runtime = make_runtime(
        cfg.runtime,
        model=cfg.model,
        device=cfg.device,
        half=cfg.half,
        batch_size=cfg.batch_size,
        trt_options=parse_trt_options(cfg.trt_options),
        trt_prebuild=cfg.trt_prebuild,
        trt_cache_dir=cfg.trt_cache_dir,
    )
    warmup_s = round(time.monotonic() - started, 1)
    monitor = GpuMonitor() if runtime.device.startswith("cuda") else None
    if monitor:
        monitor.start()
    threads = ThreadMonitor() if cfg.thread_profile else None
    if threads and not threads.start():
        threads = None
    log.info(
        json.dumps(
            {
                "event": "startup",
                "worker_id": cfg.worker_id,
                "batch_size": cfg.batch_size,
                "pipelined": cfg.pipeline,
                "warmup_s": warmup_s,
                **runtime.describe(),
                **(monitor.info() if monitor else {}),
            }
        )
    )
    worker = Worker(
        cfg,
        runtime,
        sqs=boto3.client("sqs", config=boto_cfg),
        s3=boto3.client("s3", config=boto_cfg),
        store=BatchStore(boto3.client("dynamodb", config=boto_cfg), cfg.table),
        gpu_monitor=monitor,
        thread_monitor=threads,
        finalizer=_lambda_finalizer(cfg.finalize_function, boto_cfg),
    )
    worker.run_forever()
    if monitor:
        monitor.stop()


if __name__ == "__main__":
    main()
