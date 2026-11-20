#!/usr/bin/env python3
"""Benchmark rows on AWS: one worker, a full queue, timed from its logs.

    uv run python scripts/benchmark.py stage                  # once: val2017 -> datasets/
    uv run python scripts/benchmark.py run --kind gpu --batch-size 64 --runtime onnx-trt --half
    uv run python scripts/benchmark.py run --kind gpu --batch-size 64 --half --eval   # mAP
    uv run python scripts/benchmark.py sweep [--pipeline]     # GPU PyTorch FP32, batch 1..64
    uv run python scripts/benchmark.py milestone6             # pipelined sweep, runtimes, mAP
    uv run python scripts/benchmark.py trt-ab                 # TensorRT engine-build A/B
    uv run python scripts/benchmark.py report                 # docs/benchmark.md + charts

One run (one report row):
  1. Cap the GPU service at 0 (it owns the queue) and check nothing else is running.
  2. Write the batch record, then copy the staged images into input/<batch_id>/ server-side,
     and wait until the queue holds every message.
  3. Start ONE worker task of the chosen kind (ECS RunTask with BATCH_SIZE, HALF, RUNTIME
     and PIPELINE overrides).
  4. Wait until completed + failed reaches the image count, then stop the task.
  5. Read its JSON logs, compute the row, write benchmarks/runs/<batch_id>.json.
  6. Restore the GPU service's scaling (always, even after an error or Ctrl-C). A series of
     runs (sweep, milestone6) caps once and restores once, after the last run.

An --eval run is the same, with Ultralytics' validation settings (confidence 0.001,
multi-label): it isn't a timed row. Its part files are scored for mAP50-95 against the local
COCO annotations, and the result goes to benchmarks/evals/<batch_id>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import boto3

from detector.benchmark import (
    SWEEP_BATCH_SIZES,
    attach_map,
    bench_batch_id,
    best_gpu_batch_size,
    latest_per_setup,
    load_records,
    plot_runtimes,
    plot_sweep,
    render_report,
)
from detector.evaluation import EvalSettings, coco_map, part_lines_to_coco
from detector.ops import (
    HOURLY_USD,
    PRICES_CHECKED,
    credentials_env,
    fetch_worker_events,
    list_images,
    list_keys,
    load_deployment,
    queue_depth,
    run_metrics,
    run_s5cmd,
    run_worker_task,
    running_tasks,
    s5cmd_copy_commands,
    s5cmd_upload_commands,
    scaling_max,
    set_scaling_max,
    task_instance_type,
    wait_for_alarm_clear,
    wait_for_batch,
    wait_for_task,
)
from detector.runtimes import TRT_CACHE_DIR
from detector.store import BatchStore

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "benchmarks" / "runs"
EVALS_DIR = ROOT / "benchmarks" / "evals"
NOTES = ROOT / "benchmarks" / "notes.md"  # appended to the report
SMOKE_DIR = ROOT / "benchmarks" / "smoke"  # experiments: kept, but not in the report
REPORT = ROOT / "docs" / "benchmark.md"
SWEEP_CHART = ROOT / "docs" / "img" / "throughput_by_batch.png"
RUNTIMES_CHART = ROOT / "docs" / "img" / "runtimes.png"
ANNOTATIONS = ROOT / "data" / "datasets" / "coco" / "annotations" / "instances_val2017.json"
STAGE_PREFIX = "datasets/coco-val2017"
RUNTIMES = ("pytorch", "onnx-cuda", "onnx-trt")


def log(msg: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')}  {msg}", flush=True)


def poll(check, *, timeout_s: float, poll_s: float = 5, what: str):
    """Call check() until it returns something truthy."""
    deadline = time.monotonic() + timeout_s
    while True:
        result = check()
        if result:
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out after {timeout_s:.0f} s waiting for {what}")
        time.sleep(poll_s)


@dataclass(frozen=True)
class RunSpec:
    """One run: a timed report row, or an eval run that scores mAP."""

    kind: str
    batch_size: int
    half: bool = False
    runtime: str = "pytorch"
    pipeline: bool | None = None  # default: pipelined on GPU, one step at a time on CPU
    eval_run: bool = False
    images: int = 5000
    timeout_s: float = 7200
    extra_env: tuple[tuple[str, str], ...] = ()  # e.g. (("TRT_OPTIONS", "{...}"),)
    smoke: bool = False  # write the run file to benchmarks/smoke/ instead of runs/

    @property
    def pipelined(self) -> bool:
        return self.kind == "gpu" if self.pipeline is None else self.pipeline

    @property
    def precision(self) -> str:
        return "fp16" if self.half else "fp32"

    def describe(self) -> str:
        mode = "pipelined" if self.pipelined else "one step at a time"
        what = "eval" if self.eval_run else "timed"
        return f"{self.kind} {self.runtime} {self.precision} bs{self.batch_size}, {mode}, {what}"

    def environment(self) -> dict[str, str]:
        env = {
            "BATCH_SIZE": str(self.batch_size),
            "HALF": "1" if self.half else "0",
            "RUNTIME": self.runtime,
            "PIPELINE": "1" if self.pipelined else "0",
        }
        env |= dict(self.extra_env)
        if self.eval_run:
            s = EvalSettings()
            env |= {
                "CONF": str(s.conf),
                "IOU": str(s.iou),
                "MAX_DET": str(s.max_det),
                "MULTI_LABEL": "1",
            }
        return env


class Bench:
    def __init__(self, env: str, profile: str) -> None:
        self.session = boto3.Session(profile_name=profile)
        c = self.session.client
        self.d = load_deployment(c("cloudformation"), env)
        self.s3, self.sqs, self.ecs = c("s3"), c("sqs"), c("ecs")
        self.appscaling, self.logs = c("application-autoscaling"), c("logs")
        self.cloudwatch = c("cloudwatch")
        self.env = env
        self.store = BatchStore(c("dynamodb"), self.d.table)

    # stage ---------------------------------------------------------------------

    def stage(self, source: Path, force: bool) -> None:
        files = list_images(source)
        staged = set(list_keys(self.s3, self.d.bucket, STAGE_PREFIX))
        todo = files if force else [f for f in files if f.name not in staged]
        if not todo:
            log(f"all {len(files)} images already in s3://{self.d.bucket}/{STAGE_PREFIX}/")
            return
        log(f"uploading {len(todo)} images to s3://{self.d.bucket}/{STAGE_PREFIX}/")
        start = time.monotonic()
        commands = s5cmd_upload_commands(todo, self.d.bucket, STAGE_PREFIX)
        run_s5cmd(commands, credentials_env(self.session))
        log(f"staged in {time.monotonic() - start:.0f} s")

    # series ----------------------------------------------------------------------

    def series(self, specs: list[RunSpec]) -> list[dict]:
        """Run specs in order with the GPU service capped once for the whole series."""
        for spec in specs:
            self._check(spec)
        gpu = self.d.workers["gpu"]
        restore_max = scaling_max(self.appscaling, self.d.cluster, gpu.service)
        log(f"capping {gpu.service} at 0 tasks for {len(specs)} run(s) (was max {restore_max})")
        set_scaling_max(self.appscaling, self.ecs, self.d.cluster, gpu.service, 0)
        results = []
        try:
            for i, spec in enumerate(specs, 1):
                log(f"run {i}/{len(specs)}: {spec.describe()}")
                results.append(self._run(spec))
        finally:
            if restore_max is not None:
                self._restore_gpu_scaling(restore_max)
        return results

    def _check(self, spec: RunSpec) -> None:
        workers = self.d.workers.get(spec.kind)
        if workers is None or not workers.task_family:
            raise SystemExit(f"no {spec.kind} workers in the deployed stack: run make deploy")
        if spec.kind == "cpu" and (spec.half or spec.runtime != "pytorch"):
            raise SystemExit("the CPU worker runs PyTorch FP32 only")
        if spec.eval_run and not ANNOTATIONS.is_file():
            raise SystemExit(f"--eval needs {ANNOTATIONS} (see data/dl_data.sh)")

    # one run -----------------------------------------------------------------------

    def _run(self, spec: RunSpec) -> dict:
        workers = self.d.workers[spec.kind]
        cluster = self.d.cluster
        names = list_keys(self.s3, self.d.bucket, STAGE_PREFIX)[: spec.images]
        if len(names) < spec.images:
            raise SystemExit(f"only {len(names)} images staged: run `benchmark.py stage` first")

        task_arn: str | None = None
        stopped = False
        try:
            poll(
                lambda: not running_tasks(self.ecs, cluster),
                timeout_s=600,
                what="running worker tasks to stop",
            )

            # SQS's approximate counts lag behind deletes, so give them a few minutes.
            def queue_empty() -> bool:
                depth = queue_depth(self.sqs, self.d.queue_url)
                return not depth["visible"] and not depth["in_flight"]

            try:
                poll(queue_empty, timeout_s=180, poll_s=10, what="the queue to empty")
            except TimeoutError:
                depth = queue_depth(self.sqs, self.d.queue_url)
                raise SystemExit(
                    f"queue isn't empty ({depth}); drain it before benchmarking"
                ) from None

            batch_id = bench_batch_id(
                spec.kind,
                spec.batch_size,
                spec.precision,
                runtime=spec.runtime,
                pipelined=spec.pipelined,
                eval_run=spec.eval_run,
            )
            self.store.create_batch(batch_id, len(names), "yolov8m")
            log(f"batch {batch_id}: copying {len(names)} staged images into input/")
            commands = s5cmd_copy_commands(names, self.d.bucket, STAGE_PREFIX, batch_id)
            run_s5cmd(commands, credentials_env(self.session))
            poll(
                lambda: queue_depth(self.sqs, self.d.queue_url)["visible"] >= len(names),
                timeout_s=600,
                what=f"{len(names)} messages in the queue",
            )
            log(f"queue holds {len(names)} messages; starting one {spec.kind} worker")

            requested = time.monotonic()
            task_arn = run_worker_task(
                self.ecs, cluster, workers, spec.environment(), started_by="benchmark"
            )
            task = wait_for_task(self.ecs, cluster, task_arn, "RUNNING", timeout_s=1800)
            startup_s = time.monotonic() - requested
            instance_type = task_instance_type(self.ecs, cluster, task)
            log(f"worker running on {instance_type} after {startup_s:.0f} s (then warm-up)")

            last = [0.0]

            def progress(batch: dict) -> None:
                if time.monotonic() - last[0] >= 30:
                    last[0] = time.monotonic()
                    done = batch["completed"] + batch["failed"]
                    log(f"  {done}/{batch['expected_count']} (failed {batch['failed']})")

            batch = wait_for_batch(
                self.store, batch_id, timeout_s=spec.timeout_s, poll_s=5, on_progress=progress
            )
            self.ecs.stop_task(cluster=cluster, task=task_arn, reason="benchmark finished")
            wait_for_task(self.ecs, cluster, task_arn, "STOPPED", timeout_s=300)
            stopped = True

            # CloudWatch Logs can lag a few seconds behind the task.
            def logged() -> list[dict] | None:
                events = fetch_worker_events(self.logs, workers.log_group, task_arn)
                work = sum(e["processed"] + e["failed"] for e in events if e["event"] == "batch")
                return events if work >= len(names) else None

            events = poll(logged, timeout_s=180, poll_s=10, what="the worker's log lines")
        finally:
            if task_arn and not stopped:
                log("stopping the benchmark task")
                self.ecs.stop_task(cluster=cluster, task=task_arn, reason="benchmark aborted")

        startup = next((e for e in events if e["event"] == "startup"), {})
        common = {
            "batch_id": batch_id,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "kind": spec.kind,
            "instance_type": instance_type,
            "runtime": startup.get("runtime", spec.runtime),
            "precision": spec.precision,
            "batch_size": spec.batch_size,
            "pipelined": spec.pipelined,
        }
        if spec.eval_run:
            return self._score(common)

        record = {
            **common,
            "hourly_usd": HOURLY_USD[instance_type],
            "prices_checked": PRICES_CHECKED,
            "task_startup_s": round(startup_s, 1),
            "warmup_s": startup.get("warmup_s"),
            "batch_record": {k: batch[k] for k in ("expected_count", "completed", "failed")},
            "metrics": run_metrics(events, HOURLY_USD[instance_type]),
            "startup": startup,
        }
        record["extra_env"] = dict(spec.extra_env)
        out_dir = SMOKE_DIR if spec.smoke else RUNS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{batch_id}.json").write_text(json.dumps(record, indent=2) + "\n")
        m = record["metrics"]
        gpu_util = "" if m["gpu_util_mean"] is None else f", GPU {m['gpu_util_mean']:.0f}%"
        log(
            f"{spec.describe()}: {m['images_per_s']} images/s, {m['seconds']:.0f} s, "
            f"${m['cost_per_1000_usd']:.4f} per 1,000{gpu_util}, failed {m['failed']}"
            f"  -> {out_dir.relative_to(ROOT)}/{batch_id}.json"
        )
        return record

    def _score(self, common: dict) -> dict:
        """mAP50-95 of an eval run from its part files in results/<batch_id>/."""
        batch_id = common["batch_id"]
        prefix = f"results/{batch_id}"
        lines: list[str] = []
        for name in list_keys(self.s3, self.d.bucket, prefix):
            body = self.s3.get_object(Bucket=self.d.bucket, Key=f"{prefix}/{name}")["Body"]
            lines += body.read().decode().splitlines()
        detections, image_ids, failed = part_lines_to_coco(lines)
        log(f"scoring {len(detections):,} detections on {len(image_ids):,} images")
        metrics = coco_map(ANNOTATIONS, detections, image_ids)
        record = {
            **common,
            "settings": EvalSettings().__dict__ | {"limit": None},
            "images_scored": len(image_ids),
            "failed": failed,
            "detections": len(detections),
            "metrics": metrics,
        }
        EVALS_DIR.mkdir(parents=True, exist_ok=True)
        (EVALS_DIR / f"{batch_id}.json").write_text(json.dumps(record, indent=2) + "\n")
        log(
            f"mAP50-95 {metrics['map50_95']:.4f} (mAP50 {metrics['map50']:.4f}) for "
            f"{common['kind']} {common['runtime']} {common['precision']}"
            f"  -> benchmarks/evals/{batch_id}.json"
        )
        return record

    def _restore_gpu_scaling(self, restore_max: int) -> None:
        """Lift the cap only once the backlog alarm has cleared, or it starts idle workers."""
        gpu = self.d.workers["gpu"]
        alarm = f"detector-gpu-backlog-{self.env}"  # cfg.name("gpu-backlog")
        log(f"waiting for {alarm} to clear before restoring scaling")
        if not wait_for_alarm_clear(self.cloudwatch, alarm):
            log(f"{alarm} still in ALARM after 10 min; leaving {gpu.service} capped at 0")
            log("restore it later with: make scale-restore")
            return
        set_scaling_max(self.appscaling, self.ecs, self.d.cluster, gpu.service, restore_max)
        log(f"restored {gpu.service} scaling (max {restore_max})")

    # milestone 6 ----------------------------------------------------------------------

    def milestone6(self, sizes: list[int], images: int, cpu_eval: bool) -> None:
        """Pipelined PyTorch FP32 sweep; FP16 / ONNX Runtime rows at its best batch size;
        eval runs for mAP. One capped series per phase, since the second depends on the first.
        """
        sweep = [RunSpec("gpu", bs, images=images) for bs in sizes]
        self.series(sweep)
        records = latest_per_setup(load_records(RUNS_DIR))
        best = best_gpu_batch_size(records, pipelined_rows=True)
        log(f"best pipelined batch size: {best}")
        rows = [
            RunSpec("gpu", best, half=True, images=images),
            RunSpec("gpu", best, half=True, runtime="onnx-cuda", images=images),
            RunSpec("gpu", best, half=True, runtime="onnx-trt", images=images),
        ]
        evals = [
            RunSpec("gpu", best, runtime=s.runtime, half=s.half, eval_run=True, images=images)
            for s in [RunSpec("gpu", best), *rows]
        ]
        if cpu_eval:
            evals.append(RunSpec("cpu", 1, eval_run=True, images=images))
        self.series(rows + evals)

    def trt_ab(self, images: int, batch_size: int) -> list[dict]:
        """A/B: TensorRT engine built in the worker process vs. in a subprocess, plus a
        PyTorch FP16 reference, in one capped series (so on one instance)."""
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        specs = trt_ab_specs(stamp, images, batch_size)
        records = self.series(specs)
        print_ab(specs, records)
        return records

    # report --------------------------------------------------------------------

    @staticmethod
    def report() -> None:
        records = latest_per_setup(load_records(RUNS_DIR))
        if not records:
            raise SystemExit(f"no runs in {RUNS_DIR}")
        evals = load_records(EVALS_DIR) if EVALS_DIR.is_dir() else []
        records = attach_map(records, evals)
        charts = {}
        for key, path, plot in (
            ("sweep", SWEEP_CHART, plot_sweep),
            ("runtimes", RUNTIMES_CHART, plot_runtimes),
        ):
            if plot(records, path):
                charts[key] = path.relative_to(REPORT.parent).as_posix()
        notes = NOTES.read_text() if NOTES.is_file() else ""
        REPORT.write_text(render_report(records, charts, notes))
        print(f"wrote {REPORT.relative_to(ROOT)} with {len(charts)} chart(s)")


AB_LABELS = ("A: TensorRT, engine built in the worker", "B: TensorRT, built in a subprocess",
             "C: PyTorch FP16 reference")  # fmt: skip


def trt_ab_specs(stamp: str, images: int = 500, batch_size: int = 32) -> list[RunSpec]:
    """The A/B runs. Each TensorRT run gets its own, new cache directory inside the cache
    volume, so both start with no engine whichever instance they land on."""

    def spec(runtime: str, env: dict[str, str]) -> RunSpec:
        return RunSpec(
            "gpu",
            batch_size,
            half=True,
            runtime=runtime,
            images=images,
            extra_env=tuple({"THREAD_PROFILE": "1", **env}.items()),
            smoke=True,
        )

    cache = f"{TRT_CACHE_DIR}/ab-{stamp}"
    return [
        spec("onnx-trt", {"TRT_PREBUILD": "0", "TRT_CACHE_DIR": f"{cache}-a"}),
        spec("onnx-trt", {"TRT_PREBUILD": "1", "TRT_CACHE_DIR": f"{cache}-b"}),
        spec("pytorch", {}),
    ]


def print_ab(specs: list[RunSpec], records: list[dict]) -> None:
    print(
        f"\n{'run':42} {'img/s':>6} {'GPU%':>5} {'load':>5} {'wait':>5} {'infer':>5} "
        f"{'warmup':>7} {'prebuild':>8}  engine from cache"
    )
    for label, record in zip(AB_LABELS, records, strict=False):
        m, st = record["metrics"], record.get("startup", {})
        per = m["stage_ms_per_image"]
        gpu = "-" if m["gpu_util_mean"] is None else f"{m['gpu_util_mean']:.0f}"
        print(
            f"{label:42} {m['images_per_s']:6.1f} {gpu:>5} {per.get('load', 0):5.1f} "
            f"{per.get('wait_input', 0):5.1f} {per.get('inference', 0):5.1f} "
            f"{record.get('warmup_s') or 0:7.0f} {st.get('prebuild_s') or 0:8.0f}  "
            f"{st.get('engine_from_cache', '-')}"
        )
    print("(ms per image: load = download + decode + letterbox, wait = GPU waiting for input)")


def _env_pairs(items: list[str]) -> tuple[tuple[str, str], ...]:
    pairs = []
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set needs KEY=VALUE, got {item!r}")
        pairs.append((key, value))
    return tuple(pairs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--env", default="dev")
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    sub = p.add_subparsers(dest="command", required=True)

    st = sub.add_parser("stage", help="upload val2017 to datasets/ (not queued)")
    st.add_argument("--source", type=Path, default=ROOT / "data/datasets/coco/val2017")
    st.add_argument("--force", action="store_true", help="re-upload images already staged")

    for name in ("run", "sweep"):
        r = sub.add_parser(name)
        if name == "run":
            r.add_argument("--kind", choices=["gpu", "cpu"], required=True)
            r.add_argument("--batch-size", type=int, required=True)
            r.add_argument("--eval", action="store_true", help="score mAP instead of timing")
        else:
            r.add_argument("--batch-sizes", type=int, nargs="+", default=list(SWEEP_BATCH_SIZES))
        r.add_argument("--runtime", choices=RUNTIMES, default="pytorch")
        r.add_argument("--half", action="store_true", help="FP16 (GPU only)")
        r.add_argument(
            "--pipeline",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="default: pipelined on GPU, one step at a time on CPU",
        )
        r.add_argument("--images", type=int, default=5000)
        r.add_argument("--timeout", type=float, default=7200, help="seconds per run")
        r.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="extra worker environment, e.g. THREAD_PROFILE=1 (repeatable)",
        )
        r.add_argument("--smoke", action="store_true", help="keep the run out of the report")

    m6 = sub.add_parser("milestone6", help="pipelined sweep, FP16/ONNX Runtime rows, mAP evals")
    m6.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64])
    m6.add_argument("--images", type=int, default=5000)
    m6.add_argument("--cpu-eval", action="store_true", help="also score the CPU row (~30 min)")

    ab = sub.add_parser("trt-ab", help="TensorRT engine built in-process vs. in a subprocess")
    ab.add_argument("--images", type=int, default=500)
    ab.add_argument("--batch-size", type=int, default=32)

    sub.add_parser("report", help="write docs/benchmark.md and the charts")
    args = p.parse_args()

    if args.command == "report":
        Bench.report()
        return
    bench = Bench(args.env, args.profile)
    if args.command == "stage":
        bench.stage(args.source, args.force)
    elif args.command == "run":
        spec = RunSpec(
            args.kind,
            args.batch_size,
            args.half,
            args.runtime,
            args.pipeline,
            args.eval,
            args.images,
            args.timeout,
            _env_pairs(args.set),
            args.smoke,
        )
        bench.series([spec])
    elif args.command == "sweep":
        specs = [
            RunSpec(
                "gpu",
                bs,
                args.half,
                args.runtime,
                args.pipeline,
                False,
                args.images,
                args.timeout,
                _env_pairs(args.set),
                args.smoke,
            )
            for bs in args.batch_sizes
        ]
        bench.series(specs)
        Bench.report()
    elif args.command == "trt-ab":
        bench.trt_ab(args.images, args.batch_size)
    else:
        bench.milestone6(args.batch_sizes, args.images, args.cpu_eval)
        Bench.report()


if __name__ == "__main__":
    main()
