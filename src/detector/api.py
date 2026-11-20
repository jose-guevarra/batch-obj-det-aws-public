"""Batch status and on-demand rendering API (FastAPI; on Lambda through Mangum).

    GET  /health                                   liveness, touches no AWS service
    POST /batches                                  create a batch, get presigned upload URLs
    GET  /batches/{batch_id}                       status, progress, summary when done
    GET  /batches/{batch_id}/images                one page of image results
    GET  /batches/{batch_id}/images/{name}         one image's detections, original image URL
    GET  /batches/{batch_id}/images/{name}/annotated   annotated image URL, rendered once

Access is limited by API Gateway's resource policy (source IP allowlist), not here.
Uploads and downloads go straight to S3 through presigned URLs this API signs.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import PurePosixPath
from typing import Annotated, Any

from botocore.exceptions import ClientError
from fastapi import Depends, FastAPI, HTTPException, Query
from mangum import Mangum
from pydantic import BaseModel, Field

from detector.coco import COCO_NAMES
from detector.finalize import invoke_async
from detector.messages import (
    BATCH_ID_RE,
    combined_results_key,
    input_key,
    rendered_key,
    summary_key,
    upload_key,
)
from detector.store import BatchExists, BatchStore, is_complete
from detector.unzip import IMAGE_TYPES

MAX_FILES = 1000
URL_TTL_S = 3600
MODEL = "yolov8m"


@dataclass
class Backend:
    store: BatchStore
    s3: Any  # client for reads and writes
    presign: Any  # client for presigned URLs (regional endpoint, SigV4)
    bucket: str
    finalize: Any = None  # Callable[[str], None] | None


def presign_client(region: str) -> Any:
    """SigV4 on the regional endpoint: browser PUTs to a new bucket aren't redirected."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=f"https://s3.{region}.amazonaws.com",
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )


@cache
def aws_backend() -> Backend:
    import boto3

    presign = presign_client(os.environ.get("AWS_REGION", "us-west-2"))
    function = os.environ.get("FINALIZE_FUNCTION", "")
    lam = boto3.client("lambda") if function else None
    return Backend(
        store=BatchStore(boto3.client("dynamodb"), os.environ["TABLE"]),
        s3=boto3.client("s3"),
        presign=presign,
        bucket=os.environ["BUCKET"],
        finalize=(lambda batch_id: invoke_async(lam, function, batch_id)) if lam else None,
    )


B = Annotated[Backend, Depends(aws_backend)]


class BatchCreate(BaseModel):
    files: list[str] | None = Field(default=None, description="image file names (not paths)")
    zip: bool = False
    batch_id: str | None = None


def _valid_name(name: str) -> bool:
    return (
        0 < len(name) <= 255
        and "/" not in name
        and "\\" not in name
        and not name.startswith(".")
        and PurePosixPath(name).suffix.lower() in IMAGE_TYPES
    )


def _check_batch_id(batch_id: str) -> str:
    if not BATCH_ID_RE.match(batch_id):
        raise HTTPException(422, "batch_id: letters, digits, - and _ only, at most 64")
    return batch_id


def _new_batch_id() -> str:
    return f"ui-{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def _url(b: Backend, method: str, key: str, **params: Any) -> str:
    return b.presign.generate_presigned_url(
        method, Params={"Bucket": b.bucket, "Key": key, **params}, ExpiresIn=URL_TTL_S
    )


def _exists(b: Backend, key: str) -> bool:
    try:
        b.s3.head_object(Bucket=b.bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def _batch_or_404(b: Backend, batch_id: str) -> dict:
    batch = b.store.get_batch(_check_batch_id(batch_id))
    if batch is None:
        raise HTTPException(404, f"no batch {batch_id!r}")
    return batch


def _image_or_404(b: Backend, batch_id: str, name: str) -> dict:
    _batch_or_404(b, batch_id)
    item = b.store.get_image(batch_id, name)
    if item is None:
        raise HTTPException(404, f"no result for {name!r} in batch {batch_id!r} (yet)")
    return item


def _top_classes(detections: list[dict], n: int = 3) -> list[str]:
    counts = Counter(COCO_NAMES[int(d["class_id"])] for d in detections)
    return [name for name, _ in counts.most_common(n)]


def _named(detections: list[dict]) -> list[dict]:
    return [{**d, "class_name": COCO_NAMES[int(d["class_id"])]} for d in detections]


app = FastAPI(title="Detector batch API", version="1")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/batches", status_code=201)
def create_batch(body: BatchCreate, b: B) -> dict:
    if body.zip == bool(body.files):
        raise HTTPException(422, "send either files (a list of names) or zip: true")
    batch_id = _check_batch_id(body.batch_id) if body.batch_id else _new_batch_id()
    if body.files:
        if len(body.files) > MAX_FILES:
            raise HTTPException(
                422, f"at most {MAX_FILES} files; use scripts/upload_batch.py for more"
            )
        bad = [n for n in body.files if not _valid_name(n)]
        if bad:
            raise HTTPException(422, f"not image file names: {bad[:5]}")
        if len({n.lower() for n in body.files}) != len(body.files):
            raise HTTPException(422, "file names must be unique")
    try:
        b.store.create_batch(
            batch_id,
            None if body.zip else len(body.files or []),
            MODEL,
            source="zip" if body.zip else "ui",
        )
    except BatchExists:
        raise HTTPException(409, f"batch {batch_id!r} already exists") from None
    if body.zip:
        key = upload_key(batch_id)
        return {
            "batch_id": batch_id,
            "uploads": [{"name": "batch.zip", "url": _url(b, "put_object", key)}],
        }
    uploads = [
        {"name": n, "url": _url(b, "put_object", input_key(batch_id, n))} for n in body.files or []
    ]
    return {"batch_id": batch_id, "uploads": uploads}


@app.get("/batches/{batch_id}")
def get_batch(batch_id: str, b: B) -> dict:
    batch = _batch_or_404(b, batch_id)
    expected = batch.get("expected_count")
    done = batch["completed"] + batch["failed"]
    out: dict[str, Any] = {
        "batch_id": batch_id,
        "status": batch["status"],
        "source": batch.get("source"),
        "expected_count": expected,
        "completed": batch["completed"],
        "failed": batch["failed"],
        "percent": round(100 * done / expected, 1)
        if expected
        else (100.0 if expected == 0 else None),
        "created_at": batch.get("created_at"),
        "updated_at": batch.get("updated_at"),
        "finalized_at": batch.get("finalized_at"),
        "error": batch.get("error"),
    }
    if batch["status"] == "pending" and is_complete(batch) and b.finalize:
        # Safety net: the worker normally starts finalize; the claim makes repeats harmless.
        b.finalize(batch_id)
        out["finalize_requested"] = True
    if batch["status"] == "done":
        body = b.s3.get_object(Bucket=b.bucket, Key=summary_key(batch_id))["Body"].read()
        out["summary"] = json.loads(body)
        out["results_url"] = _url(b, "get_object", combined_results_key(batch_id))
    return out


@app.get("/batches/{batch_id}/images")
def list_images(
    batch_id: str,
    b: B,
    cursor: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 24,
) -> dict:
    _batch_or_404(b, batch_id)
    items, next_cursor = b.store.list_images(batch_id, limit=limit, cursor=cursor)
    return {
        "images": [
            {
                "image_name": i["image_name"],
                "status": i["status"],
                "num_detections": i.get("num_detections", 0),
                "top_classes": _top_classes(i.get("detections", [])),
                "classes": sorted(
                    {COCO_NAMES[int(d["class_id"])] for d in i.get("detections", [])}
                ),
                "error": i.get("error"),
            }
            for i in items
        ],
        "next_cursor": next_cursor,
    }


@app.get("/batches/{batch_id}/images/{name}")
def get_image(batch_id: str, name: str, b: B) -> dict:
    item = _image_or_404(b, batch_id, name)
    content_type = IMAGE_TYPES.get(PurePosixPath(name).suffix.lower(), "application/octet-stream")
    return {
        "image_name": name,
        "status": item["status"],
        "detections": _named(item.get("detections", [])),
        "error": item.get("error"),
        "worker_id": item.get("worker_id"),
        "processed_at": item.get("processed_at"),
        "original_url": _url(
            b, "get_object", input_key(batch_id, name), ResponseContentType=content_type
        ),
    }


@app.get("/batches/{batch_id}/images/{name}/annotated")
def annotated(batch_id: str, name: str, b: B) -> dict:
    key = rendered_key(batch_id, name)
    item = _image_or_404(b, batch_id, name)
    if _exists(b, key):
        return {"url": _url(b, "get_object", key), "cached": True, "render_ms": 0}
    if item["status"] != "done":
        raise HTTPException(409, f"{name!r} failed ({item.get('error')}); nothing to draw")
    from detector.render import render_jpeg  # OpenCV: imported only when rendering

    started = time.perf_counter()
    data = b.s3.get_object(Bucket=b.bucket, Key=input_key(batch_id, name))["Body"].read()
    try:
        jpeg = render_jpeg(data, item.get("detections", []))
    except ValueError as e:
        raise HTTPException(409, f"can't render {name!r}: {e}") from None
    b.s3.put_object(Bucket=b.bucket, Key=key, Body=jpeg, ContentType="image/jpeg")
    render_ms = round(1000 * (time.perf_counter() - started))
    return {"url": _url(b, "get_object", key), "cached": False, "render_ms": render_ms}


handler = Mangum(app, lifespan="off")  # the Lambda entry point
