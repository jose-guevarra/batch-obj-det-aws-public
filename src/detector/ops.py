"""Operator helpers shared by scripts/upload_batch.py, batch_status.py and benchmark.py."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from detector.messages import BATCH_ID_RE, input_key
from detector.store import BatchStore, is_complete

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


KINDS = ("cpu", "gpu")


@dataclass(frozen=True)
class Workers:
    """One worker kind's ECS and EC2 names."""

    kind: str
    service: str
    asg: str
    log_group: str
    task_family: str
    capacity_provider: str


@dataclass(frozen=True)
class Deployment:
    """Names the scripts need, read from the deployed stacks' outputs."""

    bucket: str
    table: str
    queue_url: str
    dlq_url: str
    cluster: str
    workers: dict[str, Workers]


def stack_outputs(cloudformation: Any, stack_name: str) -> dict[str, str]:
    stacks = cloudformation.describe_stacks(StackName=stack_name)["Stacks"]
    return {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}


def load_deployment(cloudformation: Any, env: str) -> Deployment:
    batch = stack_outputs(cloudformation, f"DetectorBatch-{env}")
    data = stack_outputs(cloudformation, f"DetectorData-{env}")
    workers = {}
    for kind in KINDS:
        k = kind.capitalize()
        if f"{k}ServiceName" not in batch:
            continue  # stack deployed before this kind existed
        workers[kind] = Workers(
            kind=kind,
            service=batch[f"{k}ServiceName"],
            asg=batch[f"{k}AsgName"],
            log_group=batch[f"{k}LogGroup"],
            task_family=batch.get(f"{k}TaskFamily", ""),
            capacity_provider=batch.get(f"{k}CapacityProvider", ""),
        )
    return Deployment(
        bucket=batch["BucketName"],
        table=data["TableName"],
        queue_url=batch["QueueUrl"],
        dlq_url=batch["DlqUrl"],
        cluster=batch["ClusterName"],
        workers=workers,
    )


def default_batch_id(prefix: str = "coco") -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"


def list_images(source: Path, limit: int | None = None) -> list[Path]:
    """Image files directly in source, sorted so --limit picks the same files every run."""
    files = sorted(p for p in Path(source).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return files[:limit] if limit else files


def write_corrupt_files(directory: Path, count: int) -> list[Path]:
    """Files with an image name but garbage content, to exercise failure handling."""
    paths = []
    for i in range(count):
        path = Path(directory) / f"corrupt-{i:03d}.jpg"
        path.write_bytes(os.urandom(2048))
        paths.append(path)
    return paths


def s5cmd_commands(files: Sequence[Path], bucket: str, batch_id: str) -> str:
    """An `s5cmd run` command file: one cp per image into input/{batch_id}/."""
    if not BATCH_ID_RE.match(batch_id):
        raise ValueError(f"batch id {batch_id!r} must match {BATCH_ID_RE.pattern}")
    lines = [
        f"cp {shlex.quote(str(p))} {shlex.quote(f's3://{bucket}/{input_key(batch_id, p.name)}')}"
        for p in files
    ]
    return "\n".join(lines) + "\n"


def s5cmd_upload_commands(files: Sequence[Path], bucket: str, prefix: str) -> str:
    """Upload files to s3://bucket/prefix/<name> (prefix must not be under input/)."""
    if prefix.startswith("input/"):
        raise ValueError("staging under input/ would queue every image")
    lines = [
        f"cp {shlex.quote(str(p))} {shlex.quote(f's3://{bucket}/{prefix.rstrip("/")}/{p.name}')}"
        for p in files
    ]
    return "\n".join(lines) + "\n"


def s5cmd_copy_commands(
    names: Sequence[str], bucket: str, source_prefix: str, batch_id: str
) -> str:
    """Server-side copies from a staged prefix into input/{batch_id}/ (queues one message each)."""
    if not BATCH_ID_RE.match(batch_id):
        raise ValueError(f"batch id {batch_id!r} must match {BATCH_ID_RE.pattern}")
    src = source_prefix.rstrip("/")
    lines = [
        f"cp {shlex.quote(f's3://{bucket}/{src}/{n}')} "
        f"{shlex.quote(f's3://{bucket}/{input_key(batch_id, n)}')}"
        for n in names
    ]
    return "\n".join(lines) + "\n"


def list_keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    """Object names (without the prefix) under prefix, sorted."""
    prefix = prefix.rstrip("/") + "/"
    names = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        names += [o["Key"][len(prefix) :] for o in page.get("Contents", [])]
    return sorted(n for n in names if n and "/" not in n)


def run_s5cmd(commands: str, credentials_env: dict[str, str]) -> None:
    """Run the commands with s5cmd, passing credentials through the environment.

    s5cmd can't read SSO profiles itself, so the caller resolves them with boto3.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(commands)
        path = f.name
    try:
        subprocess.run(
            ["s5cmd", "--numworkers", "64", "run", path],
            check=True,
            env={**os.environ, **credentials_env},
            stdout=subprocess.DEVNULL,
        )
    finally:
        os.unlink(path)


def credentials_env(session: Any) -> dict[str, str]:
    creds = session.get_credentials().get_frozen_credentials()
    env = {
        "AWS_ACCESS_KEY_ID": creds.access_key,
        "AWS_SECRET_ACCESS_KEY": creds.secret_key,
        "AWS_REGION": session.region_name,
    }
    if creds.token:
        env["AWS_SESSION_TOKEN"] = creds.token
    return env


def wait_for_batch(
    store: BatchStore,
    batch_id: str,
    *,
    timeout_s: float = 3600,
    poll_s: float = 5,
    on_progress: Callable[[dict], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Poll the batch record until completed + failed reaches expected_count."""
    deadline = time.monotonic() + timeout_s
    while True:
        batch = store.get_batch(batch_id)
        if batch is None:
            raise LookupError(f"no batch record for {batch_id!r}")
        if on_progress:
            on_progress(batch)
        if is_complete(batch):
            return batch
        if time.monotonic() >= deadline:
            raise TimeoutError(f"batch {batch_id!r} not finished after {timeout_s:.0f} s")
        sleep(poll_s)


def queue_depth(sqs: Any, queue_url: str) -> dict[str, int]:
    attrs = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"],
    )["Attributes"]
    return {
        "visible": int(attrs["ApproximateNumberOfMessages"]),
        "in_flight": int(attrs["ApproximateNumberOfMessagesNotVisible"]),
    }


# Scaling and tasks ------------------------------------------------------------------


def _scalable(cluster: str, service: str) -> dict[str, str]:
    return {
        "ServiceNamespace": "ecs",
        "ScalableDimension": "ecs:service:DesiredCount",
        "ResourceId": f"service/{cluster}/{service}",
    }


def scaling_max(appscaling: Any, cluster: str, service: str) -> int | None:
    """The service's current scaling maximum, or None if it doesn't autoscale."""
    s = _scalable(cluster, service)
    targets = appscaling.describe_scalable_targets(
        ServiceNamespace="ecs",
        ResourceIds=[s["ResourceId"]],
        ScalableDimension=s["ScalableDimension"],
    )["ScalableTargets"]
    return targets[0]["MaxCapacity"] if targets else None


def set_scaling_max(
    appscaling: Any, ecs: Any, cluster: str, service: str, max_capacity: int
) -> None:
    """Cap the service at max_capacity tasks; 0 also stops its running tasks right away."""
    appscaling.register_scalable_target(
        **_scalable(cluster, service), MinCapacity=0, MaxCapacity=max_capacity
    )
    if max_capacity == 0:
        ecs.update_service(cluster=cluster, service=service, desiredCount=0)


def wait_for_alarm_clear(
    cloudwatch: Any,
    alarm_name: str,
    *,
    timeout_s: float = 600,
    poll_s: float = 15,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Wait until the alarm leaves ALARM. False on timeout (the caller decides what to do).

    The backlog alarm stays in ALARM for a minute or two after the queue empties (SQS
    metrics lag); lifting a scaling cap while it's in ALARM starts workers on an empty queue.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        alarms = cloudwatch.describe_alarms(AlarmNames=[alarm_name])["MetricAlarms"]
        if not alarms or alarms[0]["StateValue"] != "ALARM":
            return True
        if time.monotonic() >= deadline:
            return False
        sleep(poll_s)


def running_tasks(ecs: Any, cluster: str) -> list[str]:
    """ARNs of every task in the cluster that hasn't fully stopped.

    A task ECS was told to stop is listed under desiredStatus=STOPPED right away, but it
    keeps running (and receiving messages) until it finishes its batch and exits. So stopping
    tasks count too, until their lastStatus is STOPPED.
    """
    arns: list[str] = []
    for status in ("RUNNING", "STOPPED"):
        for page in ecs.get_paginator("list_tasks").paginate(cluster=cluster, desiredStatus=status):
            arns += page["taskArns"]
    alive = []
    for i in range(0, len(arns), 100):
        tasks = ecs.describe_tasks(cluster=cluster, tasks=arns[i : i + 100])["tasks"]
        alive += [t["taskArn"] for t in tasks if t["lastStatus"] != "STOPPED"]
    return alive


def run_worker_task(
    ecs: Any, cluster: str, workers: Workers, environment: dict[str, str], started_by: str
) -> str:
    """Start one worker task outside its service (benchmarks). Returns the task ARN."""
    resp = ecs.run_task(
        cluster=cluster,
        taskDefinition=workers.task_family,
        count=1,
        capacityProviderStrategy=[{"capacityProvider": workers.capacity_provider, "weight": 1}],
        overrides={
            "containerOverrides": [
                {
                    "name": "worker",
                    "environment": [{"name": k, "value": v} for k, v in environment.items()],
                }
            ]
        },
        startedBy=started_by,
    )
    if resp.get("failures"):
        raise RuntimeError(f"run_task failed: {resp['failures']}")
    return resp["tasks"][0]["taskArn"]


def task_id(task_arn: str) -> str:
    return task_arn.rsplit("/", 1)[-1]


def wait_for_task(
    ecs: Any,
    cluster: str,
    task_arn: str,
    status: str,
    *,
    timeout_s: float = 1800,
    poll_s: float = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Poll until the task reaches status (RUNNING or STOPPED); fail early if it stops."""
    deadline = time.monotonic() + timeout_s
    while True:
        task = ecs.describe_tasks(cluster=cluster, tasks=[task_arn])["tasks"][0]
        if task["lastStatus"] == status:
            return task
        if task["lastStatus"] == "STOPPED":
            raise RuntimeError(f"task stopped: {task.get('stoppedReason', 'unknown reason')}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"task not {status} after {timeout_s:.0f} s")
        sleep(poll_s)


def task_instance_type(ecs: Any, cluster: str, task: dict) -> str:
    """The EC2 instance type the task landed on (ECS records it as an attribute)."""
    ci = ecs.describe_container_instances(
        cluster=cluster, containerInstances=[task["containerInstanceArn"]]
    )["containerInstances"][0]
    return next(a["value"] for a in ci["attributes"] if a["name"] == "ecs.instance-type")


# Worker logs and run metrics --------------------------------------------------------


def parse_log_events(messages: Sequence[str]) -> list[dict]:
    """The worker's JSON log lines ("event": "batch" / "startup"), in order."""
    events = []
    for line in messages:
        start = line.find('{"event":')
        if start < 0:
            continue
        try:
            events.append(json.loads(line[start:]))
        except json.JSONDecodeError:
            continue
    return events


def fetch_worker_events(logs: Any, log_group: str, task_arn: str) -> list[dict]:
    """Every JSON event the task logged (stream worker/worker/<task id>)."""
    messages: list[str] = []
    paginator = logs.get_paginator("filter_log_events")
    stream = f"worker/worker/{task_id(task_arn)}"
    for page in paginator.paginate(logGroupName=log_group, logStreamNames=[stream]):
        messages += [e["message"] for e in page["events"]]
    return parse_log_events(messages)


# us-west-2 On-Demand Linux prices, USD per hour (AWS price list, checked 2026-11-07).
HOURLY_USD = {"g4dn.xlarge": 0.526, "c6i.2xlarge": 0.34}
PRICES_CHECKED = "2026-11-07"


def run_metrics(batches: Sequence[dict], hourly_usd: float) -> dict[str, Any]:
    """One report row from a run's batch log lines.

    Time runs from the first batch's first receive to the last batch's commit, so worker
    startup and instance launch are excluded (the design doc's timing rule).
    """
    batches = [b for b in batches if b.get("event") == "batch" and b["processed"] + b["failed"]]
    if not batches:
        raise ValueError("no batches with work in the log")
    seconds = max(b["t_written"] for b in batches) - min(b["t_receive"] for b in batches)
    images = sum(b["processed"] + b["failed"] for b in batches)
    stages: dict[str, float] = {}
    for b in batches:
        for name, ms in b["stages_ms"].items():
            stages[name] = stages.get(name, 0.0) + ms
    gpu = [b["gpu"] for b in batches if b.get("gpu")]
    gpu_samples = sum(g["samples"] for g in gpu)
    return {
        "images": images,
        "processed": sum(b["processed"] for b in batches),
        "failed": sum(b["failed"] for b in batches),
        "skipped": sum(b["skipped"] for b in batches),
        "batches": len(batches),
        "mean_batch_images": round(images / len(batches), 2),
        "seconds": round(seconds, 2),
        "images_per_s": round(images / seconds, 2) if seconds > 0 else None,
        "cost_per_1000_usd": round(hourly_usd * seconds / 3600 / images * 1000, 5),
        "stage_ms_per_batch": {k: round(v / len(batches), 2) for k, v in stages.items()},
        "stage_ms_per_image": {k: round(v / images, 2) for k, v in stages.items()},
        "gpu_util_mean": (
            round(sum(g["util_mean"] * g["samples"] for g in gpu) / gpu_samples, 1)
            if gpu_samples
            else None
        ),
        "gpu_mem_mb_peak": max((g["mem_mb"] for g in gpu), default=None),
    }
