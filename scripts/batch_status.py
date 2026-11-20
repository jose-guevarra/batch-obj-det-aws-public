#!/usr/bin/env python3
"""Show a batch's counters plus the queues, workers and instances behind it.

uv run python scripts/batch_status.py coco-20261106-120000
uv run python scripts/batch_status.py            # queues and workers only
"""

from __future__ import annotations

import argparse
import os

import boto3

from detector.ops import load_deployment, queue_depth
from detector.store import BatchStore


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("batch_id", nargs="?")
    p.add_argument("--env", default="dev")
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    args = p.parse_args()

    session = boto3.Session(profile_name=args.profile)
    d = load_deployment(session.client("cloudformation"), args.env)

    if args.batch_id:
        batch = BatchStore(session.client("dynamodb"), d.table).get_batch(args.batch_id)
        if batch is None:
            p.error(f"no batch {args.batch_id}")
        done = batch["completed"] + batch["failed"]
        expected = batch.get("expected_count")
        pct = f" ({100 * done / expected:.0f}%)" if expected else ""
        print(f"batch {args.batch_id}: {done}/{expected if expected is not None else '?'}{pct}")
        print(
            f"  completed {batch['completed']}, failed {batch['failed']}, status {batch['status']}"
        )
        print(f"  created {batch['created_at']}, last update {batch.get('updated_at', '-')}")

    sqs = session.client("sqs")
    q, dlq = queue_depth(sqs, d.queue_url), queue_depth(sqs, d.dlq_url)
    print(
        f"queue: {q['visible']} waiting, {q['in_flight']} in flight;  dead-letter: {dlq['visible']}"
    )

    ecs, autoscaling = session.client("ecs"), session.client("autoscaling")
    for kind, w in d.workers.items():
        s = ecs.describe_services(cluster=d.cluster, services=[w.service])["services"][0]
        asg = autoscaling.describe_auto_scaling_groups(AutoScalingGroupNames=[w.asg])[
            "AutoScalingGroups"
        ][0]
        print(
            f"{kind} workers: {s['runningCount']} running / {s['desiredCount']} desired;"
            f"  instances: {len(asg['Instances'])} (desired {asg['DesiredCapacity']})"
        )
    bench = ecs.list_tasks(cluster=d.cluster, startedBy="benchmark")["taskArns"]
    if bench:
        print(f"benchmark tasks: {len(bench)}")


if __name__ == "__main__":
    main()
