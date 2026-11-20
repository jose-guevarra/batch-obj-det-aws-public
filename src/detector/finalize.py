"""Finalize a batch once every image is accounted for: combine its part files, summarize.

Runs as a Lambda function (handler: detector.finalize.handler), invoked asynchronously by
the worker whose commit completes the batch, and by GET /batches/{id} as a safety net.
Several invocations can race; BatchStore.claim_finalize lets exactly one of them run.

Part files can hold the same image twice: a worker that crashes after writing its part
file but before committing leaves a line for an image that is processed again later.
The combined file keeps one line per image, and where duplicates disagree, the image's
committed DynamoDB record decides.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime
from typing import Any

from detector.coco import COCO_NAMES
from detector.messages import combined_results_key, part_prefix, summary_key
from detector.ops import HOURLY_USD
from detector.store import BatchStore

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

GPU_INSTANCE = "g4dn.xlarge"
MAX_FAILURES_LISTED = 1000


def _read_parts(s3: Any, bucket: str, batch_id: str) -> dict[str, list[dict]]:
    lines: dict[str, list[dict]] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=part_prefix(batch_id)):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode()
            for raw in body.splitlines():
                if raw.strip():
                    line = json.loads(raw)
                    lines.setdefault(line["image_name"], []).append(line)
    return lines


def _pick(candidates: list[dict], record: dict) -> dict | None:
    """The part line that matches the committed record, else one with the same status."""
    for line in candidates:
        if line["status"] == record["status"] and line.get("worker_id") == record.get("worker_id"):
            return line
    for line in candidates:
        if line["status"] == record["status"]:
            return line
    return None


def _parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def combine(
    store: BatchStore, s3: Any, bucket: str, batch_id: str
) -> tuple[list[dict], list[dict]]:
    """(one result line per committed image, the committed records), sorted by image name."""
    parts = _read_parts(s3, bucket, batch_id)
    records = sorted(store.all_images(batch_id), key=lambda r: r["image_name"])
    combined = []
    for record in records:
        line = _pick(parts.get(record["image_name"], []), record)
        if line is None:
            # No matching part line (shouldn't happen): rebuild it from DynamoDB.
            item = store.get_image(batch_id, record["image_name"]) or record
            line = {"batch_id": batch_id, "image_name": record["image_name"], **_line(item)}
        combined.append(line)
    return combined, records


def _line(item: dict) -> dict:
    out = {"status": item["status"], "worker_id": item.get("worker_id")}
    if item["status"] == "done":
        out["detections"] = item.get("detections", [])
    else:
        out["error"] = item.get("error")
    return out


def summarize(batch: dict, combined: list[dict], records: list[dict]) -> dict:
    per_class: Counter[str] = Counter()
    scores: list[float] = []
    failures = []
    for line in combined:
        if line["status"] == "done":
            for d in line["detections"]:
                per_class[COCO_NAMES[int(d["class_id"])]] += 1
                scores.append(float(d["score"]))
        else:
            failures.append({"image_name": line["image_name"], "error": line.get("error")})

    created = _parse_time(batch.get("created_at"))
    finished = _parse_time(batch.get("updated_at"))
    processed = [t for t in (_parse_time(r.get("processed_at")) for r in records) if t]
    workers = {r.get("worker_id") for r in records if r.get("worker_id")}
    processing_s = (max(processed) - min(processed)).total_seconds() if processed else 0.0
    # Instance time while processing, at the On-Demand price; cold starts aren't included.
    cost = processing_s / 3600 * HOURLY_USD[GPU_INSTANCE] * max(1, len(workers))
    done = sum(line["status"] == "done" for line in combined)
    return {
        "batch_id": batch["pk"].removeprefix("BATCH#"),
        "images": len(combined),
        "completed": done,
        "failed": len(combined) - done,
        "detections": sum(per_class.values()),
        "detections_per_class": dict(per_class.most_common()),
        "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
        "failures": failures[:MAX_FAILURES_LISTED],
        "created_at": batch.get("created_at"),
        "last_result_at": batch.get("updated_at"),
        "total_s": round((finished - created).total_seconds(), 1) if created and finished else None,
        "processing_s": round(processing_s, 1),
        "workers": len(workers),
        "estimated_cost_usd": round(cost, 5),
        "cost_note": f"{GPU_INSTANCE} On-Demand price x processing time x workers; an estimate",
    }


def finalize(store: BatchStore, s3: Any, bucket: str, batch_id: str) -> dict | None:
    """Finalize the batch if this call wins the claim. Returns the summary, or None."""
    if not store.claim_finalize(batch_id):
        log.info(json.dumps({"event": "finalize_skipped", "batch_id": batch_id}))
        return None
    batch = store.get_batch(batch_id)
    assert batch is not None
    combined, records = combine(store, s3, bucket, batch_id)
    summary = summarize(batch, combined, records)
    body = "".join(json.dumps(line) + "\n" for line in combined)
    s3.put_object(
        Bucket=bucket,
        Key=combined_results_key(batch_id),
        Body=body.encode(),
        ContentType="application/x-ndjson",
    )
    s3.put_object(
        Bucket=bucket,
        Key=summary_key(batch_id),
        Body=json.dumps(summary, indent=1).encode(),
        ContentType="application/json",
    )
    store.finish_finalize(
        batch_id,
        {
            "detections": summary["detections"],
            "mean_score": summary["mean_score"] or 0.0,
            "estimated_cost_usd": summary["estimated_cost_usd"],
        },
    )
    log.info(json.dumps({"event": "finalized", "batch_id": batch_id, "images": len(combined)}))
    return summary


def invoke_async(lambda_client: Any, function_name: str, batch_id: str) -> None:
    """Start finalize without waiting for it (InvocationType=Event)."""
    lambda_client.invoke(
        FunctionName=function_name,
        InvocationType="Event",
        Payload=json.dumps({"batch_id": batch_id}).encode(),
    )


def handler(event: dict, _context: Any) -> dict:
    import boto3

    store = BatchStore(boto3.client("dynamodb"), os.environ["TABLE"])
    summary = finalize(store, boto3.client("s3"), os.environ["BUCKET"], event["batch_id"])
    return {"finalized": summary is not None}
