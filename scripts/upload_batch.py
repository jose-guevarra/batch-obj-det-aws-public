#!/usr/bin/env python3
"""Create a batch record, then upload its images to input/{batch_id}/ with s5cmd.

    uv run python scripts/upload_batch.py --limit 100 --corrupt 3 --wait
    uv run python scripts/upload_batch.py --batch-id coco-20261106-120000 --limit 100 --rerun

The batch record (expected_count) is written before any image, so workers can track
completion. S3 queues one message per image; ECS scales workers from the queue backlog.
--rerun uploads the same files to an existing batch without a new record, which tests
that finished images are skipped.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import boto3

from detector.ops import (
    credentials_env,
    default_batch_id,
    list_images,
    load_deployment,
    run_s5cmd,
    s5cmd_commands,
    wait_for_batch,
    write_corrupt_files,
)
from detector.store import BatchStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--env", default="dev")
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    p.add_argument("--source", type=Path, default=ROOT / "data/datasets/coco/val2017")
    p.add_argument("--limit", type=int, default=None, help="first N images (sorted)")
    p.add_argument("--corrupt", type=int, default=0, help="also upload N garbage .jpg files")
    p.add_argument("--batch-id", default=None, help="default: coco-<UTC timestamp>")
    p.add_argument("--model", default="yolov8m")
    p.add_argument("--rerun", action="store_true", help="re-upload to an existing batch")
    p.add_argument("--wait", action="store_true", help="poll until every image is accounted for")
    p.add_argument("--timeout", type=float, default=3600, help="seconds, with --wait")
    args = p.parse_args()

    session = boto3.Session(profile_name=args.profile)
    deployment = load_deployment(session.client("cloudformation"), args.env)
    store = BatchStore(session.client("dynamodb"), deployment.table)
    batch_id = args.batch_id or default_batch_id()
    if args.rerun and not args.batch_id:
        p.error("--rerun needs --batch-id")

    images = list_images(args.source, args.limit)
    if not images:
        p.error(f"no images in {args.source}")

    with tempfile.TemporaryDirectory() as tmp:
        files = images + write_corrupt_files(Path(tmp), args.corrupt)
        if args.rerun:
            if store.get_batch(batch_id) is None:
                p.error(f"batch {batch_id} does not exist")
        else:
            store.create_batch(batch_id, len(files), args.model)
        print(f"batch {batch_id}: uploading {len(files)} files to s3://{deployment.bucket}/input/")
        start = time.monotonic()
        run_s5cmd(s5cmd_commands(files, deployment.bucket, batch_id), credentials_env(session))
        print(f"uploaded in {time.monotonic() - start:.1f} s")

    if args.wait:

        def progress(batch: dict) -> None:
            done = batch["completed"] + batch["failed"]
            elapsed = time.monotonic() - start
            print(
                f"  {elapsed:7.0f} s  {done}/{batch['expected_count']}"
                f"  (completed {batch['completed']}, failed {batch['failed']})",
                flush=True,
            )

        batch = wait_for_batch(store, batch_id, timeout_s=args.timeout, on_progress=progress)
        print(
            f"done in {time.monotonic() - start:.0f} s from first upload: "
            f"{batch['completed']} completed, {batch['failed']} failed"
        )
    print(f"status: uv run python scripts/batch_status.py {batch_id}")


if __name__ == "__main__":
    main()
